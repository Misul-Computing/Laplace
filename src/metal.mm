#include "tensor.h"
#include <Metal/Metal.h>
#include <mutex>
#include <sys/mman.h>
#include <unordered_map>
#include <vector>

namespace Laplace {

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLBuffer> g_xbuf;
static id<MTLBuffer> g_ybuf;
static std::mutex g_xy_mtx;
static std::once_flag g_init;
static std::mutex g_wb_mtx;
static std::unordered_map<const void*, id<MTLBuffer>> g_wb_cache;
static std::mutex g_pipe_mtx;
static std::unordered_map<int, id<MTLComputePipelineState>> g_pipes;
static std::unordered_map<int, id<MTLComputePipelineState>> g_gemm_pipes;
static id<MTLLibrary> g_lib;
static id<MTLLibrary> g_gemm_lib;
static std::once_flag g_gemm_init;

// Zero-copy mmap registration: one or more large Metal buffers covering the
// mmap'd GGUF file. Tensors within the mmap are resolved to (buffer, offset)
// at dispatch time, avoiding the copy in get_weight_buf.
struct MmapBuf {
    id<MTLBuffer> buf;
    const uint8_t* base;
    size_t size;
};
static std::vector<MmapBuf> g_mmap_bufs;

static const char* src_gemv = R"METAL(
#include <metal_stdlib>
using namespace metal;
#define QK 32
#define QK_K 256

enum : int { T_F32=0, T_F16=1, T_BF16=30, T_Q4_0=2, T_Q4_1=3, T_Q5_0=6, T_Q5_1=7, T_Q8_0=8, T_Q2_K=10, T_Q3_K=11, T_Q4_K=12, T_Q5_K=13, T_Q6_K=14 };

struct q4_K_blk { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct q8_0_blk { half d; int8_t qs[32]; };
struct q6_K_blk { uchar ql[128]; uchar qh[64]; int8_t scales[16]; half d; };

static inline float dequant_f32(device const uchar* row, int k) { return ((device const float*)row)[k]; }
static inline float dequant_f16(device const uchar* row, int k) { return float(((device const half*)row)[k]); }
static inline float dequant_bf16(device const uchar* row, int k) { ushort v = ((device const ushort*)row)[k]; return as_type<float>((uint32_t)v << 16); }

static inline float deq_q4_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; uchar qs[16]; }* blk = (device const decltype(blk))row;
    float d = float(blk[b].d); uint8_t q = blk[b].qs[l%16];
    return d * float((l < 16 ? (q & 0xF) : (q >> 4)) - 8);
}
static inline float deq_q4_1(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; half m; uchar qs[16]; }* blk = (device const decltype(blk))row;
    float d = float(blk[b].d), m = float(blk[b].m); uint8_t q = blk[b].qs[l%16];
    return d * float(l < 16 ? (q & 0xF) : (q >> 4)) + m;
}
static inline float deq_q5_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; uchar qh[4]; uchar qs[16]; }* blk = (device const decltype(blk))row;
    float d = float(blk[b].d); uint qh = (uint)blk[b].qh[0]|((uint)blk[b].qh[1]<<8)|((uint)blk[b].qh[2]<<16)|((uint)blk[b].qh[3]<<24);
    uint8_t q = blk[b].qs[l%16]; int val = (l < 16 ? (q & 0xF) : (q >> 4)) | (((qh >> l) & 1) << 4);
    return d * float(val - 16);
}
static inline float deq_q5_1(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; half m; uchar qh[4]; uchar qs[16]; }* blk = (device const decltype(blk))row;
    float d = float(blk[b].d), m = float(blk[b].m); uint qh = (uint)blk[b].qh[0]|((uint)blk[b].qh[1]<<8)|((uint)blk[b].qh[2]<<16)|((uint)blk[b].qh[3]<<24);
    uint8_t q = blk[b].qs[l%16]; int val = (l < 16 ? (q & 0xF) : (q >> 4)) | (((qh >> l) & 1) << 4);
    return d * float(val) + m;
}
static inline float deq_q8_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; int8_t qs[32]; }* blk = (device const decltype(blk))row;
    return float(blk[b].d) * float(blk[b].qs[l]);
}
static inline uchar2 get_scm(int j, device const uchar* q) {
    return j < 4 ? uchar2(q[j]&63, q[j+4]&63) : uchar2((q[j+4]&0xF)|((q[j-4]>>6)<<4), (q[j+4]>>4)|((q[j]>>6)<<4));
}
static inline float deq_q4_K(device const uchar* row, int k) {
    int b = k/QK_K, off = k%QK_K; device const struct { half d; half dmin; uchar scales[12]; uchar qs[128]; }* blk = (device const decltype(blk))row;
    int jb = off/64, l = off%64; int is = jb*2; int lo = l%32, hi = l>=32; uint8_t q = blk[b].qs[jb*32 + lo];
    uchar2 s = hi ? get_scm(is+1, blk[b].scales) : get_scm(is, blk[b].scales);
    float val = hi ? float(q >> 4) : float(q & 0xF);
    return float(blk[b].d) * s.x * val - float(blk[b].dmin) * s.y;
}
static inline float deq_q6_K(device const uchar* row, int k) {
    int b = k/QK_K, off = k%QK_K; device const struct { uchar ql[128]; uchar qh[64]; int8_t scales[16]; half d; }* blk = (device const decltype(blk))row;
    int n = off/128, lh = off%128, sg = lh/32, l = lh%32, is = l/16;
    int qlb = n*64 + l + (sg%2)*32, qls = sg < 2 ? 0 : 4;
    int q = (int)((blk[b].ql[qlb] >> qls) & 0xF) | (((blk[b].qh[n*32 + l] >> (sg*2)) & 3) << 4);
    return float(blk[b].d) * float(blk[b].scales[n*8 + is + sg*2]) * float(q - 32);
}

static inline float deq_q2_K(device const uchar* row, int k) {
    int b = k/QK_K, off = k%QK_K; device const struct { uchar scales[16]; uchar qs[64]; half d; half dmin; }* blk = (device const decltype(blk))row;
    int j = off/16, l = off%16;
    int hf = j/8, jj = j%8, group = jj/2, lo = jj%2;
    uint8_t q = (blk[b].qs[hf*32 + lo*16 + l] >> (group*2)) & 3;
    float sc = blk[b].scales[j] & 0xF, mn = blk[b].scales[j] >> 4;
    return float(blk[b].d) * sc * float(q) - float(blk[b].dmin) * mn;
}

static inline float deq_q3_K(device const uchar* row, int k) {
    int b = k/QK_K, off = k%QK_K; device const struct { uchar hmask[32]; uchar qs[64]; uchar scales[12]; half d; }* blk = (device const decltype(blk))row;
    device const uchar* sc = blk[b].scales;
    const uint kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint a0 = (uint)sc[0]|((uint)sc[1]<<8)|((uint)sc[2]<<16)|((uint)sc[3]<<24);
    uint a1 = (uint)sc[4]|((uint)sc[5]<<8)|((uint)sc[6]<<16)|((uint)sc[7]<<24);
    uint a2 = (uint)sc[8]|((uint)sc[9]<<8)|((uint)sc[10]<<16)|((uint)sc[11]<<24);
    uint tmp = a2;
    a2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    uint a3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    a0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
    a1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int j = off/16;
    uint sv = j < 4 ? a0 : (j < 8 ? a1 : (j < 12 ? a2 : a3));
    float scale = float((sv >> ((j % 4) * 8)) & 0x3F) - 32;
    int g = off/32, l = off%32;
    uint8_t q = (blk[b].qs[(g/4)*32 + l] >> ((g%4)*2)) & 3;
    int hbit = (blk[b].hmask[l] >> g) & 1;
    int a = (int)q - 4 + 4*hbit;
    return float(blk[b].d) * scale * float(a);
}

static inline float deq_q5_K(device const uchar* row, int k) {
    int b = k/QK_K, off = k%QK_K; device const struct { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; }* blk = (device const decltype(blk))row;
    int jb = off/64, l = off%64; int is = jb*2; int hi = l>=32;
    uchar2 s = hi ? get_scm(is+1, blk[b].scales) : get_scm(is, blk[b].scales);
    uint8_t q = blk[b].qs[jb*32 + (l%32)];
    int bit = hi ? (2*jb+1) : (2*jb);
    int val = hi ? (q >> 4) : (q & 0xF);
    val += 16 * ((blk[b].qh[l%32] >> bit) & 1);
    return float(blk[b].d) * s.x * float(val) - float(blk[b].dmin) * s.y;
}

static inline float dequant(device const uchar* row, int k, int type) {
    switch (type) {
        case T_F32: return dequant_f32(row, k);
        case T_F16: return dequant_f16(row, k);
        case T_BF16: return dequant_bf16(row, k);
        case T_Q4_0: return deq_q4_0(row, k);
        case T_Q4_1: return deq_q4_1(row, k);
        case T_Q5_0: return deq_q5_0(row, k);
        case T_Q5_1: return deq_q5_1(row, k);
        case T_Q8_0: return deq_q8_0(row, k);
        case T_Q4_K: return deq_q4_K(row, k);
        case T_Q6_K: return deq_q6_K(row, k);
        case T_Q2_K: return deq_q2_K(row, k);
        case T_Q3_K: return deq_q3_K(row, k);
        case T_Q5_K: return deq_q5_K(row, k);
        default: return 0;
    }
}

// Function constant: the quant type is baked in at pipeline creation time.
// The compiler eliminates the switch in dequant() and dead-code-eliminates
// unused branches in the kernel, producing code as efficient as a hand-written
// specialized kernel. Pipelines are created on demand for whatever types the
// model actually uses.
constant int QUANT_TYPE [[function_constant(0)]];

kernel void gemv(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    uint j = tgid % (uint)N;
    uint m = tgid / (uint)N;
    if (j >= (uint)N || m >= (uint)M) return;
    device const uchar* row = W + (uint64_t)j * rb;
    device const float* xm = x + (uint64_t)m * K;
    float acc = 0;

    if (QUANT_TYPE == T_Q4_K) {
        int nb = K / 256;
        device const q4_K_blk* wr = (device const q4_K_blk*)row;
        for (int b = tiisg; b < nb; b += 32) {
            float d = float(wr[b].d), dmin = float(wr[b].dmin);
            device const uchar* q = wr[b].qs; int is = 0;
            for (int jb = 0; jb < 4; jb++) {
                auto s0 = get_scm(is, wr[b].scales), s1 = get_scm(is+1, wr[b].scales);
                float d1=d*s0.x, m1=dmin*s0.y, d2=d*s1.x, m2=dmin*s1.y;
                float4 s1v4=float4(0), s2v4=float4(0), x14=float4(0), x24=float4(0);
                device const float* xp1 = xm + b*256 + jb*64;
                device const float* xp2 = xp1 + 32;
                for (int l = 0; l < 32; l += 4) {
                    float4 a1 = *(device const float4*)(xp1 + l);
                    float4 a2 = *(device const float4*)(xp2 + l);
                    uchar4 qq = *(device const uchar4*)(q + l);
                    float4 lo = float4(qq & (uchar)0xF);
                    float4 hi = float4(qq >> (uchar)4);
                    s1v4 += a1 * lo; x14 += a1;
                    s2v4 += a2 * hi; x24 += a2;
                }
                float s1v = s1v4.x + s1v4.y + s1v4.z + s1v4.w;
                float x1 = x14.x + x14.y + x14.z + x14.w;
                float s2v = s2v4.x + s2v4.y + s2v4.z + s2v4.w;
                float x2 = x24.x + x24.y + x24.z + x24.w;
                acc += d1*s1v - m1*x1 + d2*s2v - m2*x2;
                q += 32; is += 2;
            }
        }
    } else if (QUANT_TYPE == T_Q8_0) {
        int nb = K / 32;
        device const q8_0_blk* wr = (device const q8_0_blk*)row;
        for (int b = tiisg; b < nb; b += 32) {
            float d0 = float(wr[b].d);
            float4 v0 = float4(0);
            for (int i = 0; i < 8; i++) {
                float4 a0 = *(device const float4*)(xm + b*32 + i*4);
                char4 cq = *(device const char4*)(wr[b].qs + i*4);
                float4 q0 = float4(float(cq.x), float(cq.y), float(cq.z), float(cq.w));
                v0 += a0 * q0;
            }
            acc += d0 * (v0.x+v0.y+v0.z+v0.w);
        }
    } else if (QUANT_TYPE == T_Q6_K) {
        int nb = K / 256;
        device const q6_K_blk* wr = (device const q6_K_blk*)row;
        for (int b = tiisg; b < nb; b += 32) {
            float d = float(wr[b].d);
            device const uchar* ql = wr[b].ql; device const uchar* qh = wr[b].qh;
            device const int8_t* sc = wr[b].scales;
            for (int n_off = 0; n_off < 256; n_off += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l/16;
                    int q1 = (int)((ql[l]&0xF) | (((qh[l]>>0)&3)<<4)) - 32;
                    int q2 = (int)((ql[l+32]&0xF) | (((qh[l]>>2)&3)<<4)) - 32;
                    int q3 = (int)((ql[l]>>4) | (((qh[l]>>4)&3)<<4)) - 32;
                    int q4 = (int)((ql[l+32]>>4) | (((qh[l]>>6)&3)<<4)) - 32;
                    acc += xm[b*256+n_off+l] * d*float(sc[is]) * float(q1);
                    acc += xm[b*256+n_off+l+32] * d*float(sc[is+2]) * float(q2);
                    acc += xm[b*256+n_off+l+64] * d*float(sc[is+4]) * float(q3);
                    acc += xm[b*256+n_off+l+96] * d*float(sc[is+6]) * float(q4);
                }
                ql += 64; qh += 32; sc += 8;
            }
        }
    } else if (QUANT_TYPE == T_Q4_0) {
        int nb = K / 32;
        device const struct { half d; uchar qs[16]; }* wr = (device const decltype(wr))row;
        for (int b = tiisg; b < nb; b += 32) {
            float d = float(wr[b].d);
            device const uint16_t* q4 = (device const uint16_t*)wr[b].qs;
            float sum_qx = 0, sum_x = 0;
            for (int i = 0; i < 8; i++) {
                uint16_t v = q4[i];
                float ql0 = float(v & 0x000F);
                float qh0 = float((v & 0x00F0) >> 4);
                float ql1 = float((v & 0x0F00) >> 8);
                float qh1 = float((v & 0xF000) >> 12);
                int base = b*32 + i*2;
                sum_qx += ql0 * xm[base] + qh0 * xm[base+16] + ql1 * xm[base+1] + qh1 * xm[base+17];
                sum_x += xm[base] + xm[base+16] + xm[base+1] + xm[base+17];
            }
            acc += d * (sum_qx - 8.0f * sum_x);
        }
    } else if (QUANT_TYPE == T_F16) {
        device const half* wr = (device const half*)row;
        int K4 = (K / 4) * 4;
        for (int k = tiisg * 4; k < K4; k += 128) {
            float4 wv = float4(*(device const half4*)(wr + k));
            float4 xv = *(device const float4*)(xm + k);
            acc += dot(wv, xv);
        }
        for (int k = K4 + tiisg; k < K; k += 32) acc += xm[k] * float(wr[k]);
    } else if (QUANT_TYPE == T_F32) {
        device const float* wr = (device const float*)row;
        int K4 = (K / 4) * 4;
        for (int k = tiisg * 4; k < K4; k += 128) {
            float4 wv = *(device const float4*)(wr + k);
            float4 xv = *(device const float4*)(xm + k);
            acc += dot(wv, xv);
        }
        for (int k = K4 + tiisg; k < K; k += 32) acc += xm[k] * wr[k];
    } else {
        int epb = (QUANT_TYPE >= 10 && QUANT_TYPE <= 14) ? 256 : (QUANT_TYPE < 2 || QUANT_TYPE == 30 ? 1 : 32);
        int K_aligned = (K / epb) * epb;
        for (int k = tiisg; k < K_aligned; k += 32) acc += xm[k] * dequant(row, k, QUANT_TYPE);
    }

    acc = simd_sum(acc);
    if (tiisg == 0) y[(uint64_t)m * N + j] = acc;
}
)METAL";

// GEMM source: compiled lazily on first metal_gemm(M>1) call.
// Kept separate because metal_simdgroup_matrix increases compile time
// significantly, and we don't want to block init() for a kernel that
// may never be used (decode-only sessions never hit the GEMM path).
static const char* src_gemm = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define QK 32
#define QK_K 256

enum : int { T_F32=0, T_F16=1, T_BF16=30, T_Q4_0=2, T_Q4_1=3, T_Q5_0=6, T_Q5_1=7, T_Q8_0=8, T_Q2_K=10, T_Q3_K=11, T_Q4_K=12, T_Q5_K=13, T_Q6_K=14 };

struct q4_K_blk { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct q8_0_blk { half d; int8_t qs[32]; };
struct q6_K_blk { uchar ql[128]; uchar qh[64]; int8_t scales[16]; half d; };

static inline float dequant_f32(device const uchar* row, int k) { return ((device const float*)row)[k]; }
static inline float dequant_f16(device const uchar* row, int k) { return float(((device const half*)row)[k]); }
static inline float dequant_bf16(device const uchar* row, int k) { ushort v = ((device const ushort*)row)[k]; return as_type<float>((uint32_t)v << 16); }

static inline float deq_q4_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; uchar qs[16]; }* blk = (device const decltype(blk))row;
    float d = float(blk[b].d); uint8_t q = blk[b].qs[l%16];
    return d * float((l < 16 ? (q & 0xF) : (q >> 4)) - 8);
}
static inline float deq_q4_1(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; half m; uchar qs[16]; }* blk = (device const decltype(blk))row;
    float d = float(blk[b].d), m = float(blk[b].m); uint8_t q = blk[b].qs[l%16];
    return d * float(l < 16 ? (q & 0xF) : (q >> 4)) + m;
}
static inline float deq_q5_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; uchar qh[4]; uchar qs[16]; }* blk = (device const decltype(blk))row;
    float d = float(blk[b].d); uint qh = (uint)blk[b].qh[0]|((uint)blk[b].qh[1]<<8)|((uint)blk[b].qh[2]<<16)|((uint)blk[b].qh[3]<<24);
    uint8_t q = blk[b].qs[l%16]; int val = (l < 16 ? (q & 0xF) : (q >> 4)) | (((qh >> l) & 1) << 4);
    return d * float(val - 16);
}
static inline float deq_q5_1(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; half m; uchar qh[4]; uchar qs[16]; }* blk = (device const decltype(blk))row;
    float d = float(blk[b].d), m = float(blk[b].m); uint qh = (uint)blk[b].qh[0]|((uint)blk[b].qh[1]<<8)|((uint)blk[b].qh[2]<<16)|((uint)blk[b].qh[3]<<24);
    uint8_t q = blk[b].qs[l%16]; int val = (l < 16 ? (q & 0xF) : (q >> 4)) | (((qh >> l) & 1) << 4);
    return d * float(val) + m;
}
static inline float deq_q8_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const q8_0_blk* blk = (device const q8_0_blk*)row;
    return float(blk[b].d) * float(blk[b].qs[l]);
}
static inline float deq_q2_K(device const uchar* row, int k) {
    int b = k/QK_K, l = k%QK_K; device const struct { uchar scales[16]; uchar qs[64]; half d; }* blk = (device const decltype(blk))row;
    uchar sc = blk[b].scales[l/16]; uchar q = blk[b].qs[l/4];
    int shift = (l % 4) * 2; int qv = (q >> shift) & 3;
    return float(blk[b].d) * float(qv - 2) * float(sc);
}
static inline float deq_q3_K(device const uchar* row, int k) {
    int b = k/QK_K, l = k%QK_K; device const struct { half d; uchar scales[12]; uchar qs[64]; }* blk = (device const decltype(blk))row;
    int is = l/16; uchar sc = blk[b].scales[is] & 0x3F; int sign = (blk[b].scales[is] >> 6) & 1;
    uchar q = blk[b].qs[l/4]; int shift = (l % 4) * 2; int qv = (q >> shift) & 3;
    return float(blk[b].d) * float(qv - (sign ? 4 : 0)) * float(sc);
}
static inline float deq_q4_K(device const uchar* row, int k) {
    int b = k/QK_K, l = k%QK_K; device const q4_K_blk* blk = (device const q4_K_blk*)row;
    uchar sc = blk[b].scales[l/16]; uchar q = blk[b].qs[l/2 + (l>=128?64:0)];
    int shift = (l % 32 < 16) ? 0 : 4; int qv = (q >> shift) & 0xF;
    return float(blk[b].d) * float(qv - 8) * float(sc);
}
static inline float deq_q5_K(device const uchar* row, int k) {
    int b = k/QK_K, l = k%QK_K; device const struct { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; }* blk = (device const decltype(blk))row;
    uchar sc = blk[b].scales[l/16]; uchar qh_v = blk[b].qh[l/8]; uchar q = blk[b].qs[l/2 + (l>=128?64:0)];
    int shift = (l % 32 < 16) ? 0 : 4; int qv = (q >> shift) & 0xF; int qh_bit = (qh_v >> (l%8)) & 1;
    return float(blk[b].d) * float(qv + qh_bit*16 - 8) * float(sc);
}
static inline float deq_q6_K(device const uchar* row, int k) {
    int b = k/QK_K, l = k%QK_K; device const q6_K_blk* blk = (device const q6_K_blk*)row;
    int is = l/16; int8_t sc = blk[b].scales[is]; uchar ql = blk[b].ql[l/2 + (l>=128?64:0)];
    uchar qh = blk[b].qh[l/4]; int shift = (l % 32 < 16) ? 0 : 4;
    int qv = ((ql >> shift) & 0xF) | (((qh >> (l%4)) & 1) << 4);
    return float(blk[b].d) * float(qv - 32) * float(sc);
}

constant int QUANT_TYPE [[function_constant(0)]];

static inline float dequant(device const uchar* row, int k) {
    switch (QUANT_TYPE) {
        case T_F32: return dequant_f32(row, k);
        case T_F16: return dequant_f16(row, k);
        case T_BF16: return dequant_bf16(row, k);
        case T_Q4_0: return deq_q4_0(row, k);
        case T_Q4_1: return deq_q4_1(row, k);
        case T_Q5_0: return deq_q5_0(row, k);
        case T_Q5_1: return deq_q5_1(row, k);
        case T_Q8_0: return deq_q8_0(row, k);
        case T_Q2_K: return deq_q2_K(row, k);
        case T_Q3_K: return deq_q3_K(row, k);
        case T_Q4_K: return deq_q4_K(row, k);
        case T_Q5_K: return deq_q5_K(row, k);
        case T_Q6_K: return deq_q6_K(row, k);
        default: return 0;
    }
}

// GEMM kernel for M > 1 (prefill). Uses simdgroup_matrix 8x8 tiles.
// Threadgroup: 128 threads = 4 SIMD groups in 2x2, covering 32x32 output.
// Each SIMD group handles 16x16 output via 2x2 simdgroup_matrix tiles.
// Weights are dequantized to threadgroup shared memory, then loaded into
// simdgroup_matrix tiles for hardware-accelerated multiply-accumulate.
kernel void gemm_simdgroup(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    constant int& n_tg_x [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    const int BM = 32, BN = 32, BK = 32;
    int row_start = (int)(tgid / (uint)n_tg_x) * BM;
    int col_start = (int)(tgid % (uint)n_tg_x) * BN;
    if (row_start >= M || col_start >= N) return;
    int sg_m = (int)(sgid / 2);
    int sg_n = (int)(sgid % 2);

    threadgroup float Xs[BM][BK];
    threadgroup float Ws[BN][BK];
    threadgroup float Ys[BM][BN];

    simdgroup_matrix<float, 8, 8> Ymat[2][2];

    for (int ko = 0; ko < K; ko += BK) {
        // Load X tile [BM, BK] from device (float, already contiguous)
        for (int idx = (int)tid; idx < BM * BK; idx += 128) {
            int r = idx / BK, c = idx % BK;
            int mr = row_start + r, ki = ko + c;
            Xs[r][c] = (mr < M && ki < K) ? x[(uint64_t)mr * K + ki] : 0;
        }
        // Dequant W tile [BN, BK] to threadgroup shared memory
        for (int idx = (int)tid; idx < BN * BK; idx += 128) {
            int r = idx / BK, c = idx % BK;
            int ni = col_start + r, ki = ko + c;
            device const uchar* wrow = W + (uint64_t)ni * rb;
            Ws[r][c] = (ni < N && ki < K) ? dequant(wrow, ki) : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply-accumulate in 8x8 simdgroup_matrix tiles
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<float, 8, 8> Xa[2], Wa[2];
            simdgroup_load(Xa[0], (threadgroup float*)&Xs[sg_m*16][kk], BK, ulong2(0,0));
            simdgroup_load(Xa[1], (threadgroup float*)&Xs[sg_m*16 + 8][kk], BK, ulong2(0,0));
            // W is stored as [N, K]; we need W^T[K, N] for the multiply,
            // so load with transpose=true.
            simdgroup_load(Wa[0], (threadgroup float*)&Ws[sg_n*16][kk], BK, ulong2(0,0), true);
            simdgroup_load(Wa[1], (threadgroup float*)&Ws[sg_n*16 + 8][kk], BK, ulong2(0,0), true);
            simdgroup_multiply_accumulate(Ymat[0][0], Xa[0], Wa[0], Ymat[0][0]);
            simdgroup_multiply_accumulate(Ymat[0][1], Xa[0], Wa[1], Ymat[0][1]);
            simdgroup_multiply_accumulate(Ymat[1][0], Xa[1], Wa[0], Ymat[1][0]);
            simdgroup_multiply_accumulate(Ymat[1][1], Xa[1], Wa[1], Ymat[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results through threadgroup buffer for bounds-safe writes
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            simdgroup_store(Ymat[i][j], (threadgroup float*)&Ys[sg_m*16 + i*8][sg_n*16 + j*8], BN, ulong2(0,0));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = (int)tid; idx < BM * BN; idx += 128) {
        int r = idx / BN, c = idx % BN;
        int mr = row_start + r, nc = col_start + c;
        if (mr < M && nc < N) y[(uint64_t)mr * N + nc] = Ys[r][c];
    }
}
)METAL";

static void init() {
    std::call_once(g_init, []{
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return;
        g_q = [g_dev newCommandQueue];
        g_xbuf = [g_dev newBufferWithLength:128*1024 options:MTLResourceStorageModeShared];
        g_ybuf = [g_dev newBufferWithLength:32*1024*1024 options:MTLResourceStorageModeShared];
        NSError* err = nil;
        g_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src_gemv] options:nil error:&err];
        if (!g_lib) { fprintf(stderr, "[metal] %s\n", err ? [[err localizedDescription] UTF8String] : "?"); g_dev = nil; return; }
    });
}

// Lazily compile the GEMM library (simdgroup_matrix). Called on first
// metal_gemm(M>1) call, not during init(), to avoid blocking decode-only
// sessions with a slow compile.
static void init_gemm() {
    std::call_once(g_gemm_init, []{
        if (!g_dev) return;
        NSError* err = nil;
        g_gemm_lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:src_gemm] options:nil error:&err];
        if (!g_gemm_lib) { fprintf(stderr, "[metal] gemm lib: %s\n", err ? [[err localizedDescription] UTF8String] : "?"); }
    });
}

// Create a specialized pipeline for a given quant type using function constants.
// The Metal compiler bakes in QUANT_TYPE, eliminating the dequant switch and
// dead-code-eliminating unused kernel branches. Called on first encounter of
// each type, then cached in g_pipes.
static id<MTLComputePipelineState> get_pipe(int type) {
    {
        std::lock_guard<std::mutex> lk(g_pipe_mtx);
        auto it = g_pipes.find(type);
        if (it != g_pipes.end()) return it->second;
    }
    if (!g_lib) return nil;
    MTLFunctionConstantValues* vals = [MTLFunctionConstantValues new];
    int t = type;
    [vals setConstantValue:&t type:MTLDataTypeInt atIndex:0];
    NSError* err = nil;
    id<MTLFunction> f = [g_lib newFunctionWithName:@"gemv" constantValues:vals error:&err];
    if (!f) { fprintf(stderr, "[metal] function for type %d: %s\n", type, err ? [[err localizedDescription] UTF8String] : "?"); return nil; }
    id<MTLComputePipelineState> pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!pipe) { fprintf(stderr, "[metal] pipeline for type %d: %s\n", type, err ? [[err localizedDescription] UTF8String] : "?"); return nil; }
    std::lock_guard<std::mutex> lk(g_pipe_mtx);
    g_pipes[type] = pipe;
    return pipe;
}

// Pipeline for the gemm_simdgroup kernel. Uses the same QUANT_TYPE function
// constant to specialize the dequant function used inside the kernel.
static id<MTLComputePipelineState> get_gemm_pipe(int type) {
    {
        std::lock_guard<std::mutex> lk(g_pipe_mtx);
        auto it = g_gemm_pipes.find(type);
        if (it != g_gemm_pipes.end()) return it->second;
    }
    init_gemm();
    if (!g_gemm_lib) return nil;
    MTLFunctionConstantValues* vals = [MTLFunctionConstantValues new];
    int t = type;
    [vals setConstantValue:&t type:MTLDataTypeInt atIndex:0];
    NSError* err = nil;
    id<MTLFunction> f = [g_gemm_lib newFunctionWithName:@"gemm_simdgroup" constantValues:vals error:&err];
    if (!f) { fprintf(stderr, "[metal] gemm function for type %d: %s\n", type, err ? [[err localizedDescription] UTF8String] : "?"); return nil; }
    id<MTLComputePipelineState> pipe = [g_dev newComputePipelineStateWithFunction:f error:&err];
    if (!pipe) { fprintf(stderr, "[metal] gemm pipeline for type %d: %s\n", type, err ? [[err localizedDescription] UTF8String] : "?"); return nil; }
    std::lock_guard<std::mutex> lk(g_pipe_mtx);
    g_gemm_pipes[type] = pipe;
    return pipe;
}

bool metal_available() { init(); return g_dev != nil; }

// Create zero-copy Metal buffers covering the mmap'd GGUF file. The pointer
// is page-aligned (mmap guarantees this). The length is rounded up to page
// size. If the mmap exceeds maxBufferLength, it is split into multiple
// buffers. Tensors are later resolved to (buffer, offset) at dispatch time.
static void metal_register_mmap(const void* base, size_t size) {
    if (!g_dev || !base || size == 0) return;
    long page_size = sysconf(_SC_PAGESIZE);
    size_t aligned = (size + (size_t)page_size - 1) & ~((size_t)page_size - 1);
    NSUInteger max_buf = g_dev.maxBufferLength;
    const uint8_t* ptr = (const uint8_t*)base;
    size_t remaining = aligned;
    int n_bufs = 0;
    while (remaining > 0) {
        size_t chunk = remaining < (size_t)max_buf ? remaining : (size_t)max_buf;
        chunk &= ~((size_t)page_size - 1);
        if (chunk == 0) chunk = (size_t)page_size;
        id<MTLBuffer> buf = [g_dev newBufferWithBytesNoCopy:(void*)ptr
                                                    length:chunk
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        if (buf) {
            std::lock_guard<std::mutex> lk(g_wb_mtx);
            g_mmap_bufs.push_back({buf, ptr, chunk});
            n_bufs++;
        } else {
            fprintf(stderr, "[metal] mmap buffer failed at %p size %zu\n", ptr, chunk);
        }
        ptr += chunk;
        remaining -= chunk;
    }
    fprintf(stderr, "[metal] model %.1f GB, %d zero-copy buffer(s)\n", size/1e9, n_bufs);
}

void metal_register_weights(const void* base, size_t size) {
    init();
    metal_register_mmap(base, size);
}

// Look up a tensor pointer in the registered mmap buffers. Returns the Metal
// buffer and sets offset to the byte offset within that buffer. If the tensor
// spans two buffers (at a split boundary), returns nil so the caller falls
// back to the copy path.
static id<MTLBuffer> find_mmap_buf_locked(const void* ptr, size_t len, size_t& offset) {
    const uint8_t* p = (const uint8_t*)ptr;
    for (const auto& mb : g_mmap_bufs) {
        if (p >= mb.base && p + len <= mb.base + mb.size) {
            offset = (size_t)(p - mb.base);
            return mb.buf;
        }
    }
    return nil;
}

static id<MTLBuffer> get_weight_buf(const void* ptr, size_t len, size_t& offset) {
    offset = 0;
    {
        std::lock_guard<std::mutex> lk(g_wb_mtx);
        // Try zero-copy mmap path first
        id<MTLBuffer> mb = find_mmap_buf_locked(ptr, len, offset);
        if (mb) return mb;
        // Try copy cache
        auto it = g_wb_cache.find(ptr);
        if (it != g_wb_cache.end()) return it->second;
    }
    // Fall back to copy for tensors not in the registered mmap
    madvise((void*)ptr, len, MADV_WILLNEED);
    id<MTLBuffer> buf = [g_dev newBufferWithBytes:ptr length:len options:MTLResourceStorageModeShared];
    if (buf) {
        std::lock_guard<std::mutex> lk(g_wb_mtx);
        g_wb_cache[ptr] = buf;
        madvise((void*)ptr, len, MADV_DONTNEED);
    }
    return buf;
}

static bool supported(GGMLType t) {
    switch (t) {
        case GGMLType::F32: case GGMLType::F16: case GGMLType::BF16:
        case GGMLType::Q4_0: case GGMLType::Q4_1: case GGMLType::Q5_0:
        case GGMLType::Q5_1: case GGMLType::Q8_0: case GGMLType::Q4_K:
        case GGMLType::Q6_K: case GGMLType::Q2_K: case GGMLType::Q3_K:
        case GGMLType::Q5_K: return true;
        default: return false;
    }
}

bool metal_gemm(const float* x, const Tensor& w, float* y, int M, int K, int N) {
    init();
    if (!g_dev || !g_lib || N <= 0 || K <= 0 || M <= 0) return false;
    if (!supported(w.type)) return false;
    int type = (int)w.type;
    uint64_t rb = ((uint64_t)K + elements_per_block(w.type) - 1) / elements_per_block(w.type) * bytes_per_block(w.type);

    if (M > 1) {
        // GEMM path: simdgroup_matrix kernel for prefill
        id<MTLComputePipelineState> pipe = get_gemm_pipe(type);
        if (!pipe) return false;
        @autoreleasepool {
            std::lock_guard<std::mutex> lk(g_xy_mtx);
            size_t wbytes = (size_t)N * rb;
            size_t w_off = 0;
            id<MTLBuffer> wb = get_weight_buf(w.data, wbytes, w_off);
            if (!wb) {
                wb = [g_dev newBufferWithBytes:w.data length:wbytes options:MTLResourceStorageModeShared];
                w_off = 0;
            }
            size_t xbytes = (size_t)M * K * 4, ybytes = (size_t)M * N * 4;
            id<MTLBuffer> xb = [g_dev newBufferWithBytes:x length:xbytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> yb = [g_dev newBufferWithLength:ybytes options:MTLResourceStorageModeShared];
            if (!wb || !xb || !yb) return false;
            int n_tg_x = (N + 31) / 32;
            int n_tg_y = (M + 31) / 32;
            id<MTLCommandBuffer> cmd = [g_q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:wb offset:w_off atIndex:0];
            [enc setBuffer:xb offset:0 atIndex:1];
            [enc setBuffer:yb offset:0 atIndex:2];
            [enc setBytes:&K length:4 atIndex:3];
            [enc setBytes:&N length:4 atIndex:4];
            [enc setBytes:&rb length:8 atIndex:5];
            [enc setBytes:&M length:4 atIndex:6];
            [enc setBytes:&n_tg_x length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(n_tg_x * n_tg_y, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
            memcpy(y, [yb contents], ybytes);
        }
        return true;
    }

    // GEMV path: M == 1
    id<MTLComputePipelineState> pipe = get_pipe(type);
    if (!pipe) return false;
    @autoreleasepool {
        std::lock_guard<std::mutex> lk(g_xy_mtx);
        size_t wbytes = (size_t)N * rb;
        size_t w_off = 0;
        id<MTLBuffer> wb = get_weight_buf(w.data, wbytes, w_off);
        if (!wb) {
            wb = [g_dev newBufferWithBytes:w.data length:wbytes options:MTLResourceStorageModeShared];
            w_off = 0;
        }
        size_t xbytes = (size_t)M * K * 4, ybytes = (size_t)M * N * 4;
        id<MTLBuffer> xb = nil, yb = nil;
        if (g_xbuf && xbytes <= (size_t)g_xbuf.length) {
            memcpy([g_xbuf contents], x, xbytes);
            xb = g_xbuf;
        } else {
            xb = [g_dev newBufferWithBytes:x length:xbytes options:MTLResourceStorageModeShared];
        }
        if (g_ybuf && ybytes <= (size_t)g_ybuf.length) {
            yb = g_ybuf;
        } else {
            yb = [g_dev newBufferWithLength:ybytes options:MTLResourceStorageModeShared];
        }
        if (!wb || !xb || !yb) return false;
        id<MTLCommandBuffer> cmd = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:wb offset:w_off atIndex:0];
        [enc setBuffer:xb offset:0 atIndex:1];
        [enc setBuffer:yb offset:0 atIndex:2];
        [enc setBytes:&K length:4 atIndex:3];
        [enc setBytes:&N length:4 atIndex:4];
        [enc setBytes:&rb length:8 atIndex:5];
        [enc setBytes:&M length:4 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(N*M, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        memcpy(y, [yb contents], ybytes);
    }
    return true;
}

bool metal_gemv(const float* x, const Tensor& w, float* y, int K, int N) {
    return metal_gemm(x, w, y, 1, K, N);
}

} // namespace Laplace
