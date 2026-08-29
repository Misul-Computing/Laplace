// Decode GEMV used by the Hello-safe 17:48 binary. 1 row / TG, 32 threads.
static const char* src_gemv = R"METAL(
#include <metal_stdlib>
using namespace metal;
#define QK 32
#define QK_K 256

enum : int { T_F32=0, T_F16=1, T_BF16=30, T_Q4_0=2, T_Q4_1=3, T_Q5_0=6, T_Q5_1=7, T_Q8_0=8, T_Q2_K=10, T_Q3_K=11, T_Q4_K=12, T_Q5_K=13, T_Q6_K=14, T_IQ2_XXS=16, T_IQ1_S=19 };

struct q4_K_blk { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct q2_K_blk { uchar scales[16]; uchar qs[64]; half d; half dmin; };
struct q8_0_blk { half d; int8_t qs[32]; };
struct q6_K_blk { uchar ql[128]; uchar qh[64]; int8_t scales[16]; half d; };
struct iq2_xxs_blk { half d; ushort qs[32]; };
struct iq1_s_blk { half d; uchar qs[32]; ushort qh[8]; };

// ggml-org/llama.cpp 4d19b287691e8f47fc303be420f630c40ec45684.
constant uchar iq2xxs_signs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
constant ulong iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};


static inline float dequant_f32(device const uchar* row, int k) { return ((device const float*)row)[k]; }
static inline float dequant_f16(device const uchar* row, int k) { return float(((device const half*)row)[k]); }
static inline float dequant_bf16(device const uchar* row, int k) { ushort v = ((device const ushort*)row)[k]; return as_type<float>((uint32_t)v << 16); }

static inline float deq_q4_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; uchar qs[16]; }* blk = (decltype(blk))row;
    float d = float(blk[b].d); uint8_t q = blk[b].qs[l%16];
    return d * float((l < 16 ? (q & 0xF) : (q >> 4)) - 8);
}
static inline float deq_q4_1(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; half m; uchar qs[16]; }* blk = (decltype(blk))row;
    float d = float(blk[b].d), m = float(blk[b].m); uint8_t q = blk[b].qs[l%16];
    return d * float(l < 16 ? (q & 0xF) : (q >> 4)) + m;
}
static inline float deq_q5_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; uchar qh[4]; uchar qs[16]; }* blk = (decltype(blk))row;
    float d = float(blk[b].d); uint qh = (uint)blk[b].qh[0]|((uint)blk[b].qh[1]<<8)|((uint)blk[b].qh[2]<<16)|((uint)blk[b].qh[3]<<24);
    uint8_t q = blk[b].qs[l%16]; int val = (l < 16 ? (q & 0xF) : (q >> 4)) | (((qh >> l) & 1) << 4);
    return d * float(val - 16);
}
static inline float deq_q5_1(device const uchar* row, int k) {
    int b = k/QK, l = k%QK; device const struct { half d; half m; uchar qh[4]; uchar qs[16]; }* blk = (decltype(blk))row;
    float d = float(blk[b].d), m = float(blk[b].m); uint qh = (uint)blk[b].qh[0]|((uint)blk[b].qh[1]<<8)|((uint)blk[b].qh[2]<<16)|((uint)blk[b].qh[3]<<24);
    uint8_t q = blk[b].qs[l%16]; int val = (l < 16 ? (q & 0xF) : (q >> 4)) | (((qh >> l) & 1) << 4);
    return d * float(val) + m;
}
static inline float deq_q8_0(device const uchar* row, int k) {
    int b = k/QK, l = k%QK;
    device const uchar* block = row + b * 34;
    int q = int(block[2 + l]);
    return float(*(device const half*)block) *
           float(q < 128 ? q : q - 256);
}
static inline uchar2 get_scm(int j, device const uchar* q) {
    return j < 4 ? uchar2(q[j]&63, q[j+4]&63) : uchar2((q[j+4]&0xF)|((q[j-4]>>6)<<4), (q[j+4]>>4)|((q[j]>>6)<<4));
}
static inline float deq_q4_K(device const uchar* row, int k) {
    int b = k/QK_K, off = k%QK_K; device const struct { half d; half dmin; uchar scales[12]; uchar qs[128]; }* blk = (decltype(blk))row;
    int jb = off/64, l = off%64; int is = jb*2; int lo = l%32, hi = l>=32; uint8_t q = blk[b].qs[jb*32 + lo];
    uchar2 s = hi ? get_scm(is+1, blk[b].scales) : get_scm(is, blk[b].scales);
    float val = hi ? float(q >> 4) : float(q & 0xF);
    return float(blk[b].d) * s.x * val - float(blk[b].dmin) * s.y;
}
static inline float deq_q6_K(device const uchar* row, int k) {
    int b = k/QK_K, off = k%QK_K; device const struct { uchar ql[128]; uchar qh[64]; int8_t scales[16]; half d; }* blk = (decltype(blk))row;
    int n = off/128, lh = off%128, sg = lh/32, l = lh%32, is = l/16;
    int qlb = n*64 + l + (sg%2)*32, qls = sg < 2 ? 0 : 4;
    int q = (int)((blk[b].ql[qlb] >> qls) & 0xF) | (((blk[b].qh[n*32 + l] >> (sg*2)) & 3) << 4);
    return float(blk[b].d) * float(blk[b].scales[n*8 + is + sg*2]) * float(q - 32);
}
static inline float deq_q2_K(device const uchar* row, int k) {
    int b = k/QK_K, off = k%QK_K; device const struct { uchar scales[16]; uchar qs[64]; half d; half dmin; }* blk = (decltype(blk))row;
    int j = off/16, l = off%16;
    int hf = j/8, jj = j%8, group = jj/2, lo = jj%2;
    uint8_t q = (blk[b].qs[hf*32 + lo*16 + l] >> (group*2)) & 3;
    float sc = blk[b].scales[j] & 0xF, mn = blk[b].scales[j] >> 4;
    return float(blk[b].d) * sc * float(q) - float(blk[b].dmin) * mn;
}
static inline float deq_q3_K(device const uchar* row, int k) {
    int b = k/QK_K, off = k%QK_K; device const struct { uchar hmask[32]; uchar qs[64]; uchar scales[12]; half d; }* blk = (decltype(blk))row;
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
    int b = k/QK_K, off = k%QK_K; device const struct { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; }* blk = (decltype(blk))row;
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

constant int QUANT_TYPE [[function_constant(0)]];
constant bool SPARSE_AFFINE [[function_constant(1)]];

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
        for (int b = tiisg; b < nb; b += 32) {
            device const uchar* block = row + b * 34;
            float d0 = float(*(device const half*)block);
            float sum = 0.0f;
            for (int i = 0; i < 32; ++i) {
                int q = int(block[2 + i]);
                if (q >= 128) q -= 256;
                sum += xm[b * 32 + i] * float(q);
            }
            acc += d0 * sum;
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
    } else if (QUANT_TYPE == T_Q2_K) {
        int nb = K / 256;
        device const struct { uchar scales[16]; uchar qs[64]; half d; half dmin; }* wr =
            (decltype(wr))row;
        for (int b = tiisg; b < nb; b += 32) {
            float d = float(wr[b].d), dmin = float(wr[b].dmin);
            for (int group = 0; group < 16; ++group) {
                int half_index = group / 8, local_group = group % 8;
                int packed_group = local_group / 2, packed_half = local_group % 2;
                int shift = packed_group * 2;
                device const uchar* q = wr[b].qs + half_index * 32 + packed_half * 16;
                device const float* xp = xm + b * 256 + group * 16;
                float4 qsum4 = float4(0), xsum4 = float4(0);
                for (int lane = 0; lane < 16; lane += 4) {
                    float4 xv = *(device const float4*)(xp + lane);
                    uchar4 qv = (*(device const uchar4*)(q + lane) >> (uchar)shift) & (uchar)3;
                    qsum4 += xv * float4(qv);
                    xsum4 += xv;
                }
                float qsum = qsum4.x + qsum4.y + qsum4.z + qsum4.w;
                float xsum = xsum4.x + xsum4.y + xsum4.z + xsum4.w;
                uchar sm = wr[b].scales[group];
                acc += d * float(sm & 0x0F) * qsum - dmin * float(sm >> 4) * xsum;
            }
        }
    } else if (QUANT_TYPE == T_Q4_0) {
        int nb = K / 32;
        device const struct { half d; uchar qs[16]; }* wr = (decltype(wr))row;
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

kernel void gemv_id(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& x_mul [[buffer(6)]],
    device const int* ids [[buffer(7)]],
    constant uint64_t& expert_stride [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    uint j = tgpig.x;
    uint slot = tgpig.z;
    if (j >= (uint)N) return;
    int eid = ids[slot];
    device const uchar* row = W + (uint64_t)eid * expert_stride + (uint64_t)j * rb;
    device const float* xm = x + (uint64_t)slot * (uint)K * (uint)x_mul;
    float acc = 0;
    int epb = (QUANT_TYPE >= 10 && QUANT_TYPE <= 14) ? 256
        : (QUANT_TYPE < 2 || QUANT_TYPE == 30 ? 1 : 32);
    int K_aligned = (K / epb) * epb;
    for (int k = tiisg; k < K_aligned; k += 32)
        acc += xm[k] * dequant(row, k, QUANT_TYPE);
    acc = simd_sum(acc);
    if (tiisg == 0) y[(uint64_t)slot * (uint)N + j] = acc;
}

// Test-only gathered down projection and weighted reduction. Each output row
// walks the selected worklist in order, so reduction order is deterministic.
// QUANT_TYPE is specialized at pipeline creation for the exact source codec.
kernel void moe_down_reduce(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& row_bytes [[buffer(5)]],
    constant uint64_t& expert_stride [[buffer(6)]],
    device const uint* ids [[buffer(7)]],
    device const float* route_weights [[buffer(8)]],
    device const float* expert_scales [[buffer(9)]],
    constant int& selected_count [[buffer(10)]],
    constant bool& has_scales [[buffer(11)]],
    uint j [[thread_position_in_grid]]) {
    if (j >= (uint)N) return;
    float reduced = 0.0f;
    for (int slot = 0; slot < selected_count; ++slot) {
        const uint eid = ids[slot];
        device const uchar* row = W + (uint64_t)eid * expert_stride +
                                  (uint64_t)j * row_bytes;
        device const float* xslot = x + (uint64_t)slot * (uint)K;
        float dot = 0.0f;
        for (int k = 0; k < K; ++k)
            dot += xslot[k] * dequant(row, k, QUANT_TYPE);
        const float expert_scale = has_scales ? expert_scales[eid] : 1.0f;
        reduced += route_weights[slot] * expert_scale * dot;
    }
    y[j] = reduced;
}

// llama.cpp kernel_mul_mv_q4_K_f32 launch: 2 rows / simdgroup, 2 simdgroups /
// TG. Threads split K-blocks (ix = tiisg/8) so all 32 lanes work on short K.
// simd_sum is per row. Do not shuffle across a mixed-row group.
kernel void gemv_q4k(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    (void)M;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = K / 256;
    const int first_row = (int)(tgid * 2u + sgitg) * 2;
    if (first_row >= N) return;

    float yl[16], yh[16];
    float sumf[2] = {0.f, 0.f};
    device const float* y4 = x + ix * 256 + 64 * iq + 8 * ir;
    uint16_t sc16[4];
    thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        for (short row = 0; row < 2; ++row) {
            if (first_row + row >= N) break;
            device const q4_K_blk* blk =
                (device const q4_K_blk*)(W + (uint64_t)(first_row + row) * rb);
            device const uint16_t* sc = (device const uint16_t*)blk[ib].scales + iq;
            device const uint16_t* q1 = (device const uint16_t*)blk[ib].qs + 16 * iq + 4 * ir;
            device const half* dh = &blk[ib].d;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const uint16_t* q2 = q1 + 32;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            sumf[row] += float(dh[0]) * ((acc1[0] + acc1[1] * (1.f / 256.f)) * sc8[0] +
                                         (acc1[2] + acc1[3] * (1.f / 256.f)) * sc8[1] * (1.f / 16.f) +
                                         (acc2[0] + acc2[1] * (1.f / 256.f)) * sc8[4] +
                                         (acc2[2] + acc2[3] * (1.f / 256.f)) * sc8[5] * (1.f / 16.f)) -
                         float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                         sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
        y4 += 4 * 256;
    }
    for (int row = 0; row < 2 && first_row + row < N; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) y[first_row + row] = sum_all;
    }
}

// Derived affine UInt2 M=1: one FP16 scale and bias per 256 input values,
// four output rows per SIMD group, and two SIMD groups per threadgroup.
kernel void gemv_affine_u2_256(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* biases [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& N [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    constexpr int rows_per_simd = 4;
    constexpr int values_per_lane = 16;
    constexpr int k_tile = 512;
    const int first_row = int(tgid) * 8 + int(sgitg) * rows_per_simd;
    if (first_row >= N) return;
    const int row_bytes = K / 4;
    const int blocks_per_row = K / 256;
    float sums[rows_per_simd] = {0.f};
    for (int k = 0; k < K; k += k_tile) {
        float4 xv[values_per_lane / 4];
        float sum_x = 0.f;
        device const float4* lane_x = (device const float4*)(
            x + k + int(tiisg) * values_per_lane);
        for (int i = 0; i < values_per_lane / 4; ++i) {
            xv[i] = lane_x[i];
            sum_x += xv[i].x + xv[i].y + xv[i].z + xv[i].w;
        }
        const int block = k / 256 + int(tiisg) / 16;
        for (int row = 0; row < rows_per_simd && first_row + row < N; ++row) {
            device const uchar* lane_weights =
                packed + (first_row + row) * row_bytes + k / 4 + int(tiisg) * 4;
            float sum_qx = 0.f;
            for (int i = 0; i < values_per_lane / 4; ++i) {
                const uchar bits = lane_weights[i];
                const float4 q = float4(bits & 3u, (bits >> 2) & 3u,
                                        (bits >> 4) & 3u, bits >> 6);
                sum_qx += dot(xv[i], q);
            }
            const int plane_index = (first_row + row) * blocks_per_row + block;
            sums[row] += float(scales[plane_index]) * sum_qx +
                         float(biases[plane_index]) * sum_x;
        }
    }
    for (int row = 0; row < rows_per_simd && first_row + row < N; ++row) {
        const float total = simd_sum(sums[row]);
        if (tiisg == 0) y[first_row + row] = total;
    }
}

// Upstream-compatible IQ2_XXS M=1 mapping: four rows per SIMD group,
// two SIMD groups per threadgroup, and one 32-value block per lane.
kernel void gemv_iq2_xxs(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    threadgroup uchar* scratch [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    (void)M;
    constexpr int rows_per_simd = 4;
    const int first_row = (int)(tgid * 2u + sgitg) * rows_per_simd;
    threadgroup ulong* grids = (threadgroup ulong*)scratch;
    threadgroup uchar* signs_table = (threadgroup uchar*)(grids + 256);
    const int grid_pos = (32 * (int)sgitg + (int)tiisg) * 4;
    for (int i = 0; i < 4; ++i) grids[grid_pos + i] = iq2xxs_grid[grid_pos + i];
    const int sign_pos = (32 * (int)sgitg + (int)tiisg) * 2;
    for (int i = 0; i < 2; ++i) signs_table[sign_pos + i] = iq2xxs_signs[sign_pos + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (first_row >= N) return;

    const int blocks32 = (K / 256) * 8;
    device const float* x32 = x + 32 * tiisg;
    float values[32];
    float sums[rows_per_simd] = {0.f};
    for (int block32 = (int)tiisg; block32 < blocks32; block32 += 32) {
        for (int i = 0; i < 32; ++i) values[i] = x32[i];
        const int block256 = block32 / 8;
        const int group32 = block32 % 8;
        for (int row = 0; row < rows_per_simd && first_row + row < N; ++row) {
            device const iq2_xxs_blk* blocks =
                (device const iq2_xxs_blk*)(W + (uint64_t)(first_row + row) * rb);
            device const iq2_xxs_blk& block = blocks[block256];
            device const ushort* packed = block.qs + 4 * group32;
            const uint grid_words = (uint)packed[0] | ((uint)packed[1] << 16);
            const uint signs_and_scale = (uint)packed[2] | ((uint)packed[3] << 16);
            thread const uchar* grid_indices = (thread const uchar*)&grid_words;
            const float scale = float(block.d) *
                (0.5f + float(signs_and_scale >> 28));
            float sum = 0.f;
            for (int group8 = 0; group8 < 4; ++group8) {
                threadgroup const uchar* grid =
                    (threadgroup const uchar*)(grids + grid_indices[group8]);
                const uchar signs = signs_table[(signs_and_scale >> (7 * group8)) & 127u];
                for (int lane = 0; lane < 8; ++lane)
                    sum += values[group8 * 8 + lane] * float(grid[lane]) *
                        ((signs & (1u << lane)) ? -1.f : 1.f);
            }
            sums[row] += scale * sum;
        }
        x32 += 32 * 32;
    }
    for (int row = 0; row < rows_per_simd && first_row + row < N; ++row) {
        const float total = 0.25f * simd_sum(sums[row]);
        if (tiisg == 0) y[first_row + row] = total;
    }
}

// IQ1_S M=1 mapping from the upstream GGML physical contract: four rows per
// SIMD group, two SIMD groups per threadgroup, and eight lanes per row.
kernel void gemv_iq1_s(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    constant const uint* grid [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    (void)M;
    constexpr ushort lanes_per_row = 8;
    const ushort tx = tiisg % lanes_per_row;
    const ushort ty = tiisg / lanes_per_row;
    const int row = int(tgid) * 8 + int(sgitg) * 4 + int(ty);
    if (row >= N) return;

    device const iq1_s_blk* blocks =
        (device const iq1_s_blk*)(W + uint64_t(row) * rb);
    device const float4x4* input = (device const float4x4*)x;
    float sum = 0.f;
    for (int chunk = int(tx); 16 * chunk < K; chunk += lanes_per_row) {
        const int block_index = chunk / 16;
        const int within_block = chunk % 16;
        const int group32 = within_block / 2;
        const int half_index = within_block % 2;
        device const iq1_s_blk& block = blocks[block_index];
        const ushort high = block.qh[group32];
        const ushort shifted = high >> (6 * half_index);
        device const uchar* indices = block.qs + 4 * group32 + 2 * half_index;
        constant const uchar* grid1 = (constant const uchar*)(grid +
            (uint(indices[0]) | ((uint(shifted) << 8) & 0x700u)));
        constant const uchar* grid2 = (constant const uchar*)(grid +
            (uint(indices[1]) | ((uint(shifted) << 5) & 0x700u)));
        const float scale = float(block.d) *
            float(2u * ((uint(high) >> 12) & 7u) + 1u);
        const float minimum = scale * ((high & 0x8000u) ? -1.125f : -0.875f);
        float4x4 weights;
        for (int lane = 0; lane < 4; ++lane) {
            weights[0][lane] = scale * float(grid1[lane] & 0x0fu) + minimum;
            weights[1][lane] = scale * float(grid1[lane] >> 4) + minimum;
            weights[2][lane] = scale * float(grid2[lane] & 0x0fu) + minimum;
            weights[3][lane] = scale * float(grid2[lane] >> 4) + minimum;
        }
        const float4x4 values = input[chunk];
        sum += dot(weights[0], values[0]) + dot(weights[1], values[1]) +
               dot(weights[2], values[2]) + dot(weights[3], values[3]);
    }
    sum += simd_shuffle_down(sum, 4);
    sum += simd_shuffle_down(sum, 2);
    sum += simd_shuffle_down(sum, 1);
    if (tx == 0) y[row] = sum;
}

// Four rows per SIMD group and two SIMD groups per threadgroup. Threads split
// the K blocks four ways so every lane works when K/256 is small.
kernel void gemv_q2k(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    (void)M;
    constexpr int rows_per_simd = 4;
    const int first_row = (int)(tgid * 2u + sgitg) * rows_per_simd;
    if (first_row >= N) return;
    const int nb = K / 256;
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    device const q2_K_blk* rows = (device const q2_K_blk*)(W + (uint64_t)first_row * rb);
    device const float* x4 = x + ix * 256 + 128 * iq + 8 * ir;
    float yl[32];
    float sumf[rows_per_simd] = {0.f};

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = float4(0);
        for (short i = 0; i < 8; ++i) {
            yl[i] = x4[i];           sumy[0] += yl[i];
            yl[i + 8] = x4[i + 32];  sumy[1] += yl[i + 8];
            yl[i + 16] = x4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = x4[i + 96]; sumy[3] += yl[i + 24];
        }
        device const uchar* sc = rows[ib].scales + 8 * iq + is;
        device const ushort* qs = (device const ushort*)rows[ib].qs + 16 * iq + 4 * ir;
        device const half* dh = &rows[ib].d;
        for (short row = 0; row < rows_per_simd && first_row + row < N; ++row) {
            float4 acc1 = float4(0), acc2 = float4(0);
            for (short i = 0; i < 8; i += 2) {
                acc1[0] += yl[i]      * (qs[i / 2] & 0x0003);
                acc2[0] += yl[i + 1]  * (qs[i / 2] & 0x0300);
                acc1[1] += yl[i + 8]  * (qs[i / 2] & 0x000c);
                acc2[1] += yl[i + 9]  * (qs[i / 2] & 0x0c00);
                acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
            }
            float d = dh[0], dmin = dh[1] * (1.f / 16.f);
            sumf[row] += d * ((acc1[0] + acc2[0] * (1.f / 256.f)) * (sc[0] & 0x0F) +
                               (acc1[1] + acc2[1] * (1.f / 256.f)) * (sc[2] & 0x0F) * (1.f / 4.f) +
                               (acc1[2] + acc2[2] * (1.f / 256.f)) * (sc[4] & 0x0F) * (1.f / 16.f) +
                               (acc1[3] + acc2[3] * (1.f / 256.f)) * (sc[6] & 0x0F) * (1.f / 64.f)) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                 sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
            qs += rb / sizeof(ushort);
            sc += rb;
            dh += rb / sizeof(half);
        }
        x4 += 4 * 256;
    }
    for (short row = 0; row < rows_per_simd && first_row + row < N; ++row) {
        float sum = simd_sum(sumf[row]);
        if (tiisg == 0) y[first_row + row] = sum;
    }
}

// Test-only occupancy falsifier for gemv_q2k. The storage, unpacking, and
// accumulation are identical; each SIMD group carries two output rows.
kernel void gemv_q2k_two_row(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    (void)M;
    constexpr int rows_per_simd = 2;
    const int first_row = (int)(tgid * 2u + sgitg) * rows_per_simd;
    if (first_row >= N) return;
    const int nb = K / 256;
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    device const q2_K_blk* rows = (device const q2_K_blk*)(W + (uint64_t)first_row * rb);
    device const float* x4 = x + ix * 256 + 128 * iq + 8 * ir;
    float yl[32];
    float sumf[rows_per_simd] = {0.f};

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = float4(0);
        for (short i = 0; i < 8; ++i) {
            yl[i] = x4[i];           sumy[0] += yl[i];
            yl[i + 8] = x4[i + 32];  sumy[1] += yl[i + 8];
            yl[i + 16] = x4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = x4[i + 96]; sumy[3] += yl[i + 24];
        }
        device const uchar* sc = rows[ib].scales + 8 * iq + is;
        device const ushort* qs = (device const ushort*)rows[ib].qs + 16 * iq + 4 * ir;
        device const half* dh = &rows[ib].d;
        for (short row = 0; row < rows_per_simd && first_row + row < N; ++row) {
            float4 acc1 = float4(0), acc2 = float4(0);
            for (short i = 0; i < 8; i += 2) {
                acc1[0] += yl[i]      * (qs[i / 2] & 0x0003);
                acc2[0] += yl[i + 1]  * (qs[i / 2] & 0x0300);
                acc1[1] += yl[i + 8]  * (qs[i / 2] & 0x000c);
                acc2[1] += yl[i + 9]  * (qs[i / 2] & 0x0c00);
                acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
            }
            float d = dh[0], dmin = dh[1] * (1.f / 16.f);
            sumf[row] += d * ((acc1[0] + acc2[0] * (1.f / 256.f)) * (sc[0] & 0x0F) +
                               (acc1[1] + acc2[1] * (1.f / 256.f)) * (sc[2] & 0x0F) * (1.f / 4.f) +
                               (acc1[2] + acc2[2] * (1.f / 256.f)) * (sc[4] & 0x0F) * (1.f / 16.f) +
                               (acc1[3] + acc2[3] * (1.f / 256.f)) * (sc[6] & 0x0F) * (1.f / 64.f)) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                 sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
            qs += rb / sizeof(ushort);
            sc += rb;
            dh += rb / sizeof(half);
        }
        x4 += 4 * 256;
    }
    for (short row = 0; row < rows_per_simd && first_row + row < N; ++row) {
        float sum = simd_sum(sumf[row]);
        if (tiisg == 0) y[first_row + row] = sum;
    }
}

// Test-only register-pressure falsifier for gemv_q2k. It preserves the
// four-row launch and Q2_K equation while loading one eight-value subgroup.
kernel void gemv_q2k_streamed(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    (void)M;
    constexpr int rows_per_simd = 4;
    const int first_row = (int)(tgid * 2u + sgitg) * rows_per_simd;
    if (first_row >= N) return;
    const int nb = K / 256;
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    device const q2_K_blk* rows = (device const q2_K_blk*)(W + (uint64_t)first_row * rb);
    device const float* x4 = x + ix * 256 + 128 * iq + 8 * ir;
    float sumf[rows_per_simd] = {0.f};

    for (int ib = ix; ib < nb; ib += 4) {
        float quant_sum[rows_per_simd] = {0.f};
        float min_sum[rows_per_simd] = {0.f};
        for (short subgroup = 0; subgroup < 4; ++subgroup) {
            float values[8];
            float value_sum = 0.f;
            for (short i = 0; i < 8; ++i) {
                values[i] = x4[subgroup * 32 + i];
                value_sum += values[i];
            }
            const ushort low_mask = (ushort)(0x0003u << (2 * subgroup));
            const ushort high_mask = (ushort)(0x0300u << (2 * subgroup));
            const float mask_scale = 1.f / float(1u << (2 * subgroup));
            device const uchar* sc = rows[ib].scales + 8 * iq + is;
            device const ushort* qs = (device const ushort*)rows[ib].qs + 16 * iq + 4 * ir;
            for (short row = 0; row < rows_per_simd && first_row + row < N; ++row) {
                float low = 0.f, high = 0.f;
                for (short i = 0; i < 8; i += 2) {
                    low += values[i] * (qs[i / 2] & low_mask);
                    high += values[i + 1] * (qs[i / 2] & high_mask);
                }
                quant_sum[row] += (low + high * (1.f / 256.f)) *
                                  float(sc[2 * subgroup] & 0x0F) * mask_scale;
                min_sum[row] += value_sum * float(sc[2 * subgroup] & 0xF0);
                qs += rb / sizeof(ushort);
                sc += rb;
            }
        }
        device const half* dh = &rows[ib].d;
        for (short row = 0; row < rows_per_simd && first_row + row < N; ++row) {
            sumf[row] += float(dh[0]) * quant_sum[row] -
                         float(dh[1]) * (1.f / 16.f) * min_sum[row];
            dh += rb / sizeof(half);
        }
        x4 += 4 * 256;
    }
    for (short row = 0; row < rows_per_simd && first_row + row < N; ++row) {
        float sum = simd_sum(sumf[row]);
        if (tiisg == 0) y[first_row + row] = sum;
    }
}

// Same 4-row / 2-SG launch as gemv_q4k. Each of 32 lanes owns one Q8_0
// value in every 32-wide block so short-K downs stay fully occupied.
kernel void gemv_q8(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    (void)M;
    const int first_row = (int)(tgid * 2u + sgitg) * 2;
    if (first_row >= N) return;
    int nb = K / 32;
    device const q8_0_blk* wr0 =
        (device const q8_0_blk*)(W + (uint64_t)first_row * rb);
    bool row1_ok = (first_row + 1) < N;
    device const q8_0_blk* wr1 = row1_ok
        ? (device const q8_0_blk*)(W + (uint64_t)(first_row + 1) * rb) : wr0;
    float acc0 = 0, acc1 = 0;
    for (int b = 0; b < nb; ++b) {
        float xv = x[b * 32 + (int)tiisg];
        acc0 += xv * float(wr0[b].qs[tiisg]) * float(wr0[b].d);
        acc1 += xv * float(wr1[b].qs[tiisg]) * float(wr1[b].d);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (tiisg == 0) {
        y[first_row] = acc0;
        if (row1_ok) y[first_row + 1] = acc1;
    }
}

kernel void gemv_q8_id(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    device const int* ids [[buffer(7)]],
    constant uint64_t& expert_stride [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    (void)M;
    const int first_row = (int)(tgpig.x * 2u + sgitg) * 2;
    if (first_row >= N) return;
    const int eid = ids[tgpig.z];
    device const uchar* We = W + (uint64_t)eid * expert_stride;
    device float* ye = y + (uint64_t)tgpig.z * (uint)N;
    device const float* xslot = x + (uint64_t)tgpig.z * (uint)K * (uint)M;
    int nb = K / 32;
    device const q8_0_blk* wr0 = (device const q8_0_blk*)(We + (uint64_t)first_row * rb);
    bool row1_ok = (first_row + 1) < N;
    device const q8_0_blk* wr1 = row1_ok
        ? (device const q8_0_blk*)(We + (uint64_t)(first_row + 1) * rb) : wr0;
    float acc0 = 0, acc1 = 0;
    for (int b = 0; b < nb; ++b) {
        float xv = xslot[b * 32 + (int)tiisg];
        acc0 += xv * float(wr0[b].qs[tiisg]) * float(wr0[b].d);
        acc1 += xv * float(wr1[b].qs[tiisg]) * float(wr1[b].d);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (tiisg == 0) {
        ye[first_row] = acc0;
        if (row1_ok) ye[first_row + 1] = acc1;
    }
}

kernel void gemv_q4k_id(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    device const int* ids [[buffer(7)]],
    constant uint64_t& expert_stride [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    (void)M;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = K / 256;
    const int first_row = (int)(tgpig.x * 2u + sgitg) * 2;
    if (first_row >= N) return;
    const int eid = ids[tgpig.z];
    device const uchar* We = W + (uint64_t)eid * expert_stride;
    device float* ye = y + (uint64_t)tgpig.z * (uint)N;
    device const float* xslot = x + (uint64_t)tgpig.z * (uint)K * (uint)M;

    float yl[16], yh[16];
    float sumf[2] = {0.f, 0.f};
    device const float* y4 = xslot + ix * 256 + 64 * iq + 8 * ir;
    uint16_t sc16[4];
    thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        for (short row = 0; row < 2; ++row) {
            if (first_row + row >= N) break;
            device const q4_K_blk* blk =
                (device const q4_K_blk*)(We + (uint64_t)(first_row + row) * rb);
            device const uint16_t* sc = (device const uint16_t*)blk[ib].scales + iq;
            device const uint16_t* q1 = (device const uint16_t*)blk[ib].qs + 16 * iq + 4 * ir;
            device const half* dh = &blk[ib].d;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const uint16_t* q2 = q1 + 32;
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            sumf[row] += float(dh[0]) * ((acc1[0] + acc1[1] * (1.f / 256.f)) * sc8[0] +
                                         (acc1[2] + acc1[3] * (1.f / 256.f)) * sc8[1] * (1.f / 16.f) +
                                         (acc2[0] + acc2[1] * (1.f / 256.f)) * sc8[4] +
                                         (acc2[2] + acc2[3] * (1.f / 256.f)) * sc8[5] * (1.f / 16.f)) -
                         float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                         sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
        y4 += 4 * 256;
    }
    for (int row = 0; row < 2 && first_row + row < N; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) ye[first_row + row] = sum_all;
    }
}

inline float gelu_tanh_exact(float g) {
    const float c = 0.7978845608028654f;
    const float c2 = 0.044715f;
    const float z = c * (g + c2 * g * g * g);
    // Evaluate tanh(z) through its equivalent signed exponential form. On
    // Apple GPU math, tanh can return NaN for large finite arguments even
    // though the GELU-tanh equation is well-defined there. Selecting the
    // sign before forming the exponential also avoids exp overflow for z<0.
    const float e = z >= 0.0f ? exp(-2.0f * z) : exp(2.0f * z);
    const float t = z >= 0.0f ? (1.0f - e) / (1.0f + e)
                              : (e - 1.0f) / (e + 1.0f);
    return 0.5f * g * (1.0f + t);
}

kernel void act_glu(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& n [[buffer(3)]],
    constant int& swiglu [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= (uint)n) return;
    float g = gate[i], u = up[i];
    if (swiglu) {
        out[i] = (g / (1.0f + exp(-g))) * u;
    } else {
        out[i] = gelu_tanh_exact(g) * u;
    }
}

kernel void act_glu_experts(
    device const float* gate_up [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& intermediate [[buffer(2)]],
    constant int& selected [[buffer(3)]],
    constant int& swiglu [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    if (position.x >= (uint)intermediate || position.y >= (uint)selected) return;
    const uint64_t gate_base = uint64_t(position.y) * uint64_t(intermediate) * 2u;
    const uint64_t output_index = uint64_t(position.y) * uint64_t(intermediate) + position.x;
    const float g = gate_up[gate_base + position.x];
    const float u = gate_up[gate_base + uint64_t(intermediate) + position.x];
    if (swiglu) {
        out[output_index] = (g / (1.0f + exp(-g))) * u;
    } else {
        out[output_index] = gelu_tanh_exact(g) * u;
    }
}

kernel void rmsnorm_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]
) {
    float acc = 0;
    for (uint i = tid; i < (uint)n; i += ntg) acc += x[i] * x[i];
    acc = simd_sum(acc);
    threadgroup float sh[8];
    uint sg = tid / 32;
    if ((tid & 31) == 0) sh[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ms = 0;
    if (tid == 0) {
        uint nsg = (ntg + 31) / 32;
        for (uint i = 0; i < nsg; i++) ms += sh[i];
        sh[0] = 1.0f / sqrt(ms / float(n) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = sh[0];
    for (uint i = tid; i < (uint)n; i += ntg) y[i] = x[i] * inv * w[i];
}

kernel void rmsnorm_noscale(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]
) {
    float acc = 0;
    for (uint i = tid; i < (uint)n; i += ntg) acc += x[i] * x[i];
    acc = simd_sum(acc);
    threadgroup float sh[8];
    if ((tid & 31) == 0) sh[tid / 32] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float ms = 0;
        uint nsg = (ntg + 31) / 32;
        for (uint i = 0; i < nsg; i++) ms += sh[i];
        sh[0] = 1.0f / sqrt(ms / float(n) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = sh[0];
    for (uint i = tid; i < (uint)n; i += ntg) y[i] = x[i] * inv;
}

kernel void rmsnorm_rows_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& cols [[buffer(3)]],
    constant int& rows [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]
) {
    if (tgid >= (uint)rows) return;
    device const float* xr = x + (uint64_t)tgid * cols;
    device float* yr = y + (uint64_t)tgid * cols;
    float acc = 0;
    for (uint i = tid; i < (uint)cols; i += ntg) acc += xr[i] * xr[i];
    acc = simd_sum(acc);
    threadgroup float sh[8];
    if ((tid & 31) == 0) sh[tid / 32] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float ms = 0;
        uint nsg = (ntg + 31) / 32;
        for (uint i = 0; i < nsg; i++) ms += sh[i];
        sh[0] = 1.0f / sqrt(ms / float(cols) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = sh[0];
    for (uint i = tid; i < (uint)cols; i += ntg) yr[i] = xr[i] * inv * w[i];
}

kernel void vec_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* o [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < (uint)n) o[i] = a[i] + b[i];
}

kernel void activation_importance_accumulate(
    device const float* input [[buffer(0)]],
    device float* sum_squares [[buffer(1)]],
    device atomic_uint* sample_count [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < width) {
        const float value = input[i];
        sum_squares[i] += value * value;
    }
    if (i == 0) atomic_fetch_add_explicit(sample_count, 1u, memory_order_relaxed);
}

kernel void vec_scale(
    device float* x [[buffer(0)]],
    constant float& s [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < (uint)n) x[i] *= s;
}

kernel void vec_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* o [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < (uint)n) o[i] = a[i] * b[i];
}

kernel void vec_copy(
    device const float* a [[buffer(0)]],
    device float* o [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < (uint)n) o[i] = a[i];
}

kernel void embedding_f16(
    device const half* table [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant int& width [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < (uint)width) output[i] = float(table[(uint64_t)token * (uint)width + i]) * scale;
}

// GGML Q6_K is not a generic int6 plane. Each 256-value block has two
// 128-value halves with independently placed low bits, high-bit pairs, and
// signed scale groups. Keep this direct unpack exact so token embeddings can
// stay in their mapped Q6_K representation.
kernel void embedding_q6k(
    device const q6_K_blk* table [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant int& width [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= (uint)width) return;
    const uint block = i / 256;
    const uint element = i % 256;
    const device q6_K_blk& source = table[(uint64_t)token * (uint)(width / 256) + block];
    const uint half_index = element / 128;
    const uint lane = element % 128;
    const uint group = lane / 32;
    const uint within = lane % 32;
    const uint low_index = half_index * 64 + (group == 1 || group == 3 ? 32 : 0) + within;
    const uint high_index = half_index * 32 + within;
    const uchar low = source.ql[low_index];
    const uchar high = source.qh[high_index];
    const int low_bits = (group < 2) ? int(low & 0x0f) : int(low >> 4);
    const int high_bits = int((high >> (group * 2)) & 3);
    const int scale_index = int(half_index * 8 + (within / 16) + group * 2);
    output[i] = float(source.d) * float(source.scales[scale_index]) *
                float((low_bits | (high_bits << 4)) - 32) * scale;
}

// GGML Q4_K uses packed scale/min pairs for four 64-value groups. This is a
// direct selected-row decode. It does not reuse Q6_K or expand the table.
kernel void embedding_q4k(
    device const q4_K_blk* table [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant int& width [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= (uint)width) return;
    const uint block = i / 256;
    const uint element = i % 256;
    const device q4_K_blk& source = table[(uint64_t)token * (uint)(width / 256) + block];
    const uint group = element / 64;
    const uint lane = element % 64;
    const uchar2 scale_min = get_scm(int(group * 2 + (lane >= 32 ? 1 : 0)), source.scales);
    const uchar packed = source.qs[group * 32 + lane % 32];
    const float value = float(lane < 32 ? (packed & 0x0f) : (packed >> 4));
    output[i] = (float(source.d) * float(scale_min.x) * value -
                 float(source.dmin) * float(scale_min.y)) * scale;
}

kernel void rope_f32(
    device float* qk [[buffer(0)]],
    constant int& n_heads [[buffer(1)]],
    constant int& head_dim [[buffer(2)]],
    constant int& rope_pairs [[buffer(3)]],
    constant float& rope_base [[buffer(4)]],
    constant int& pos [[buffer(5)]],
    constant int& use_pf [[buffer(6)]],
    device const float* pf [[buffer(7)]],
    constant int& frequency_dimension [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (rope_pairs < 1 || frequency_dimension < rope_pairs * 2) return;
    uint n = (uint)n_heads * (uint)rope_pairs;
    if (gid >= n) return;
    uint h = gid / (uint)rope_pairs;
    uint p = gid % (uint)rope_pairs;
    device float* v = qk + (uint64_t)h * head_dim;
    float inverse = pow(rope_base, -2.0f * float(p) / float(frequency_dimension));
    if (use_pf) inverse *= pf[p];
    float ang = float(pos) * inverse;
    float c = cos(ang), s = sin(ang);
    float a = v[p], b = v[p + (uint)rope_pairs];
    v[p] = a * c - b * s;
    v[p + (uint)rope_pairs] = a * s + b * c;
}

kernel void rope_interleaved_f32(
    device float* qk [[buffer(0)]],
    constant int& n_heads [[buffer(1)]],
    constant int& head_dim [[buffer(2)]],
    constant int& rope_pairs [[buffer(3)]],
    constant float& rope_base [[buffer(4)]],
    constant int& pos [[buffer(5)]],
    constant int& use_pf [[buffer(6)]],
    device const float* pf [[buffer(7)]],
    constant int& frequency_dimension [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (rope_pairs < 1 || frequency_dimension < rope_pairs * 2) return;
    uint n = (uint)n_heads * (uint)rope_pairs;
    if (gid >= n) return;
    uint h = gid / (uint)rope_pairs;
    uint p = gid % (uint)rope_pairs;
    device float* v = qk + (uint64_t)h * head_dim;
    float inverse = pow(rope_base, -2.0f * float(p) / float(frequency_dimension));
    if (use_pf) inverse *= pf[p];
    float ang = float(pos) * inverse;
    float c = cos(ang), s = sin(ang);
    uint first = p * 2;
    float a = v[first], b = v[first + 1];
    v[first] = a * c - b * s;
    v[first + 1] = a * s + b * c;
}

kernel void rope_multisection_f32(
    device float* qk [[buffer(0)]],
    constant int& n_heads [[buffer(1)]],
    constant int& head_dim [[buffer(2)]],
    constant int& rope_pairs [[buffer(3)]],
    constant float& rope_base [[buffer(4)]],
    constant int4& positions [[buffer(5)]],
    constant uint4& sections [[buffer(6)]],
    constant int& use_pf [[buffer(7)]],
    device const float* pf [[buffer(8)]],
    constant int& frequency_dimension [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (rope_pairs < 1 || frequency_dimension < rope_pairs * 2 ||
        sections.x + sections.y + sections.z + sections.w != (uint)rope_pairs) return;
    uint n = (uint)n_heads * (uint)rope_pairs;
    if (gid >= n) return;
    uint h = gid / (uint)rope_pairs;
    uint p = gid % (uint)rope_pairs;
    uint section = p < sections.x ? 0 : p < sections.x + sections.y ? 1 :
                   p < sections.x + sections.y + sections.z ? 2 : 3;
    device float* v = qk + (uint64_t)h * head_dim;
    float inverse = pow(rope_base, -2.0f * float(p) / float(frequency_dimension));
    if (use_pf) inverse *= pf[p];
    float ang = float(positions[section]) * inverse;
    float c = cos(ang), s = sin(ang);
    float a = v[p], b = v[p + (uint)rope_pairs];
    v[p] = a * c - b * s;
    v[p + (uint)rope_pairs] = a * s + b * c;
}

kernel void axis_split_f32(
    device const float* input [[buffer(0)]],
    device float* first [[buffer(1)]],
    device float* second [[buffer(2)]],
    constant int& rows [[buffer(3)]],
    constant int& first_width [[buffer(4)]],
    constant int& second_width [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    const int width = first_width + second_width;
    const int count = rows * width;
    if (index >= (uint)count || first_width < 1 || second_width < 1) return;
    const int row = (int)index / width;
    const int column = (int)index - row * width;
    if (column < first_width) first[(uint64_t)row * first_width + column] = input[index];
    else second[(uint64_t)row * second_width + column - first_width] = input[index];
}

kernel void gated_attention_f32(
    device const float* context [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= (uint)count) return;
    const float value = gate[index];
    const float sigmoid = value >= 0.0f ? 1.0f / (1.0f + exp(-value))
                                        : exp(value) / (1.0f + exp(value));
    output[index] = context[index] * sigmoid;
}

kernel void kv_write(
    device const float* src [[buffer(0)]],
    device float* cache [[buffer(1)]],
    constant int& pos [[buffer(2)]],
    constant int& max_seq [[buffer(3)]],
    constant ulong& cache_base [[buffer(4)]],
    constant int& n [[buffer(5)]],
    constant int& stride [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= (uint)n) return;
    cache[cache_base + (uint64_t)pos * (uint)stride + i] = src[i];
}

kernel void attn_decode(
    device const float* Q [[buffer(0)]],
    device const float* Kc [[buffer(1)]],
    device const float* Vc [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int& Hq [[buffer(4)]],
    constant int& Hk [[buffer(5)]],
    constant int& Dh [[buffer(6)]],
    constant int& pos [[buffer(7)]],
    constant int& window [[buffer(8)]],
    constant int& max_seq [[buffer(9)]],
    constant ulong& cache_base [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant int& kn_stride [[buffer(12)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]]
) {
    if (h >= (uint)Hq || Hk < 1 || Dh < 1) return;
    device float* o = out + (uint64_t)h * Dh;
    int start = window > 0 ? max(0, pos - window + 1) : 0;
    int T = pos + 1 - start;
    if (T < 1 || Dh % 32 != 0 || Dh > 512) {
        for (int i = (int)tid; i < Dh; i += 32) o[i] = 0;
        return;
    }
    int kvh = (int)((uint)h * (uint)Hk / (uint)Hq);
    device const float* q = Q + (uint64_t)h * Dh;
    device const float* kbase = Kc + cache_base + (uint64_t)kvh * Dh;
    device const float* vbase = Vc + cache_base + (uint64_t)kvh * Dh;
    const int DN = Dh / 32;
    float acc[16];
    for (int j = 0; j < 16; j++) acc[j] = 0;
    float mx = -1e30f;
    float sum = 0;
    // Online softmax: one pass over K/V. Each lane sums a Dh/32 slice of
    // the dot product, simd_sum broadcasts the score, and each lane keeps
    // its own Dh/32 output accumulator.
    for (int t = 0; t < T; t++) {
        device const float* k = kbase + (uint64_t)(start + t) * (uint)kn_stride;
        device const float* v = vbase + (uint64_t)(start + t) * (uint)kn_stride;
        float part = 0;
        for (int j = (int)tid; j < Dh; j += 32) part += q[j] * k[j];
        float d = simd_sum(part) * scale;
        float m_new = max(mx, d);
        float r_old = exp(mx - m_new);
        float r_new = exp(d - m_new);
        sum = sum * r_old + r_new;
        mx = m_new;
        for (int j = 0; j < DN; j++) acc[j] = acc[j] * r_old + r_new * v[(int)tid + j * 32];
    }
    float inv = (sum > 0.f && isfinite(sum)) ? (1.0f / sum) : 0.f;
    for (int j = 0; j < DN; j++) o[(int)tid + j * 32] = acc[j] * inv;
}

kernel void apply_down_scale(
    device int* ids [[buffer(0)]],
    device float* wt [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    constant int& top_k [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < (uint)top_k) wt[i] *= scale[ids[i]];
}

kernel void moe_combine(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant int& H [[buffer(2)]],
    constant int& n [[buffer(3)]],
    device const float* wt [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= (uint)H) return;
    float acc = 0;
    for (int k = 0; k < n; k++) acc += wt[k] * src[(uint64_t)k * (uint)H + i];
    dst[i] = acc;
}

kernel void router_topk(
    device const float* logits [[buffer(0)]],
    device int* ids [[buffer(1)]],
    device float* wt [[buffer(2)]],
    constant int& n_exp [[buffer(3)]],
    constant int& top_k [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]
) {
    const int n = min(n_exp, 512);
    threadgroup float loc[512];
    for (uint i = tid; i < (uint)n; i += ntg)
        loc[i] = logits[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;
    float mx = loc[0];
    for (int e = 1; e < n; e++) mx = max(mx, loc[e]);
    float sum = 0;
    for (int e = 0; e < n; e++) {
        loc[e] = exp(loc[e] - mx);
        sum += loc[e];
    }
    float inv = 1.0f / sum;
    for (int e = 0; e < n; e++) loc[e] *= inv;
    for (int k = 0; k < top_k; k++) {
        int best = 0;
        float bv = loc[0];
        for (int e = 1; e < n; e++) if (loc[e] > bv) { bv = loc[e]; best = e; }
        ids[k] = best;
        wt[k] = bv;
        loc[best] = -1e30f;
    }
    float wsum = 0;
    for (int k = 0; k < top_k; k++) wsum += wt[k];
    float iw = 1.0f / wsum;
    for (int k = 0; k < top_k; k++) wt[k] *= iw;
}

// 4-row Q6_K for the tied LM head. Same launch as gemv_q4k.
kernel void gemv_q6k(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant int& M [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    (void)M;
    const int first_row = (int)(tgid * 2u + sgitg) * 2;
    if (first_row >= N) return;
    int nb = K / 256;
    device const q6_K_blk* wr0 =
        (device const q6_K_blk*)(W + (uint64_t)first_row * rb);
    bool row1_ok = (first_row + 1) < N;
    device const q6_K_blk* wr1 = row1_ok
        ? (device const q6_K_blk*)(W + (uint64_t)(first_row + 1) * rb) : wr0;
    // Every lane owns one element slot l = tiisg in each 32-wide sub-group.
    // Q6_K stores the four 32-wide groups as low ql[0..31], low ql[32..63],
    // high ql[0..31], high ql[32..63], with qh pairs 0,2,4,6 respectively.
    const int l = (int)tiisg;
    const int is = l / 16;
    float acc0 = 0, acc1 = 0;
    for (int b = 0; b < nb; ++b) {
        float d0 = float(wr0[b].d), d1 = float(wr1[b].d);
        for (int hb = 0; hb < 2; ++hb) {
            device const uchar* ql0 = wr0[b].ql + hb * 64;
            device const uchar* qh0 = wr0[b].qh + hb * 32;
            device const int8_t* sc0 = wr0[b].scales + hb * 8;
            device const uchar* ql1 = wr1[b].ql + hb * 64;
            device const uchar* qh1 = wr1[b].qh + hb * 32;
            device const int8_t* sc1 = wr1[b].scales + hb * 8;
            int base = b * 256 + hb * 128;
            for (int g = 0; g < 4; ++g) {
                int qi = (g & 1) ? 32 + l : l;
                int shift = 2 * g;
                int lo0 = (g >= 2) ? (ql0[qi] >> 4) : (ql0[qi] & 0xF);
                int lo1 = (g >= 2) ? (ql1[qi] >> 4) : (ql1[qi] & 0xF);
                int q0 = (lo0 | (((qh0[l] >> shift) & 3) << 4)) - 32;
                int q1 = (lo1 | (((qh1[l] >> shift) & 3) << 4)) - 32;
                float xv = x[base + g * 32 + l];
                acc0 += xv * d0 * float(sc0[g * 2 + is]) * float(q0);
                acc1 += xv * d1 * float(sc1[g * 2 + is]) * float(q1);
            }
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (tiisg == 0) {
        y[first_row] = acc0;
        if (row1_ok) y[first_row + 1] = acc1;
    }
}

// Selected output rows remain in their original quantized matrix. The device
// block table maps compact output slots to source rows without a copied bank.
kernel void sparse_proxy_block_scores(
    device const float* input [[buffer(0)]],
    device const float* coefficients [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& input_blocks [[buffer(3)]],
    constant uint& output_blocks [[buffer(4)]],
    uint output_block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (output_block >= output_blocks) return;
    float score = 0.0f;
    for (uint input_block = 0; input_block != input_blocks; ++input_block) {
        float energy = 0.0f;
        const uint base = input_block * 256;
        for (uint offset = lane; offset < 256; offset += 32) {
            const float value = input[base + offset];
            energy += value * value;
        }
        energy = simd_sum(energy);
        if (lane == 0)
            score += energy * coefficients[output_block * input_blocks + input_block];
    }
    if (lane == 0) scores[output_block] = score;
}

// Test-only dense oracle scoring. The dense gate/up projections and SwiGLU
// have already produced the exact intermediate values. This kernel measures
// their actual per-block energy before the retained window is recomputed by
// the sparse source-format kernels.
kernel void sparse_exact_swiglu_block_scores(
    device const float* values [[buffer(0)]],
    device float* scores [[buffer(1)]],
    constant uint& output_blocks [[buffer(2)]],
    uint output_block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (output_block >= output_blocks) return;
    float score = 0.0f;
    const uint base = output_block * 256;
    for (uint offset = lane; offset < 256; offset += 32) {
        const float value = values[base + offset];
        score += value * value;
    }
    score = simd_sum(score);
    if (lane == 0) scores[output_block] = score;
}

kernel void sparse_record_window(
    constant const uint* run [[buffer(0)]],
    device uint* starts [[buffer(1)]],
    constant uint& index [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane == 0) starts[index] = run[0];
}

kernel void sparse_record_pair(
    constant const uint* block_ids [[buffer(0)]],
    device uint* starts [[buffer(1)]],
    constant uint& index [[buffer(2)]],
    constant uint& run_blocks [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane == 0) {
        starts[index] = block_ids[0];
        starts[index + 1] = block_ids[run_blocks];
    }
}

kernel void sparse_prefix_block_contributions(
    device const float* contributions [[buffer(0)]],
    device float* prefix [[buffer(1)]],
    constant uint& total_blocks [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= hidden) return;
    float sum = 0.0f;
    prefix[index] = 0.0f;
    for (uint block = 0; block < total_blocks; ++block) {
        sum += contributions[block * hidden + index];
        prefix[(block + 1) * hidden + index] = sum;
    }
}

kernel void sparse_downstream_window_scores(
    device const float* prefix [[buffer(0)]],
    device float* scores [[buffer(1)]],
    constant uint& total_blocks [[buffer(2)]],
    constant uint& selected_blocks [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    uint candidate [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[2];
    float error = 0.0f;
    for (uint index = tid; index < hidden; index += 64) {
        const float dense = prefix[total_blocks * hidden + index];
        const float retained = prefix[(candidate + selected_blocks) * hidden + index] -
                               prefix[candidate * hidden + index];
        const float omitted = dense - retained;
        error += omitted * omitted;
    }
    error = simd_sum(error);
    if (lane == 0) partial[simd_index] = error;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) scores[candidate] = -(partial[0] + partial[1]);
}

kernel void sparse_select_best_candidate(
    device const float* scores [[buffer(0)]],
    device uint* run [[buffer(1)]],
    constant uint& candidate_count [[buffer(2)]],
    constant uint& selected_blocks [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup float best_scores[32];
    threadgroup uint best_starts[32];
    float best_score = -INFINITY;
    uint best_start = UINT_MAX;
    for (uint candidate = lane; candidate < candidate_count; candidate += 32) {
        const float score = scores[candidate];
        if (score > best_score || (score == best_score && candidate < best_start)) {
            best_score = score;
            best_start = candidate;
        }
    }
    best_scores[lane] = best_score;
    best_starts[lane] = best_start;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0) return;
    for (uint candidate = 1; candidate != 32; ++candidate) {
        if (best_scores[candidate] > best_score ||
            (best_scores[candidate] == best_score && best_starts[candidate] < best_start)) {
            best_score = best_scores[candidate];
            best_start = best_starts[candidate];
        }
    }
    run[0] = best_start == UINT_MAX ? 0 : best_start;
    run[1] = best_start == UINT_MAX ? 0 : selected_blocks;
}

kernel void sparse_downstream_pair_scores(
    device const float* prefix [[buffer(0)]],
    device const uint2* pairs [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& total_blocks [[buffer(3)]],
    constant uint& run_blocks [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    uint candidate [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[2];
    const uint2 pair = pairs[candidate];
    float error = 0.0f;
    for (uint index = tid; index < hidden; index += 64) {
        const float dense = prefix[total_blocks * hidden + index];
        const float first = prefix[(pair.x + run_blocks) * hidden + index] -
                            prefix[pair.x * hidden + index];
        const float second = prefix[(pair.y + run_blocks) * hidden + index] -
                             prefix[pair.y * hidden + index];
        const float omitted = dense - first - second;
        error += omitted * omitted;
    }
    error = simd_sum(error);
    if (lane == 0) partial[simd_index] = error;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) scores[candidate] = -(partial[0] + partial[1]);
}

kernel void sparse_select_best_pair(
    device const float* scores [[buffer(0)]],
    device const uint2* pairs [[buffer(1)]],
    device uint* block_ids [[buffer(2)]],
    constant uint& candidate_count [[buffer(3)]],
    constant uint& run_blocks [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup float best_scores[32];
    threadgroup uint best_indices[32];
    float best_score = -INFINITY;
    uint best_index = UINT_MAX;
    for (uint candidate = lane; candidate < candidate_count; candidate += 32) {
        const float score = scores[candidate];
        if (score > best_score || (score == best_score && candidate < best_index)) {
            best_score = score;
            best_index = candidate;
        }
    }
    best_scores[lane] = best_score;
    best_indices[lane] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0) return;
    for (uint candidate = 1; candidate != 32; ++candidate) {
        if (best_scores[candidate] > best_score ||
            (best_scores[candidate] == best_score && best_indices[candidate] < best_index)) {
            best_score = best_scores[candidate];
            best_index = best_indices[candidate];
        }
    }
    if (best_index == UINT_MAX) return;
    const uint2 pair = pairs[best_index];
    for (uint offset = 0; offset < run_blocks; ++offset) {
        block_ids[offset] = pair.x + offset;
        block_ids[run_blocks + offset] = pair.y + offset;
    }
}

kernel void sparse_select_contiguous_window(
    device const float* scores [[buffer(0)]],
    device uint* run [[buffer(1)]],
    constant uint& total_blocks [[buffer(2)]],
    constant uint& selected_blocks [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup float best_scores[32];
    threadgroup uint best_starts[32];
    float best_score = -INFINITY;
    uint best_start = UINT_MAX;
    if (selected_blocks > 0 && selected_blocks <= total_blocks) {
        const uint last_start = total_blocks - selected_blocks;
        for (uint first = lane; first <= last_start; first += 32) {
            float energy = 0.0f;
            for (uint block = 0; block != selected_blocks; ++block)
                energy += scores[first + block];
            if (energy > best_score || (energy == best_score && first < best_start)) {
                best_score = energy;
                best_start = first;
            }
        }
    }
    best_scores[lane] = best_score;
    best_starts[lane] = best_start;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0) return;
    for (uint candidate = 1; candidate != 32; ++candidate) {
        if (best_scores[candidate] > best_score ||
            (best_scores[candidate] == best_score && best_starts[candidate] < best_start)) {
            best_score = best_scores[candidate];
            best_start = best_starts[candidate];
        }
    }
    run[0] = best_start == UINT_MAX ? 0 : best_start;
    run[1] = best_start == UINT_MAX ? 0 : selected_blocks;
}

kernel void gemv_sparse_rows(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& K [[buffer(3)]],
    constant int& packed_N [[buffer(4)]],
    constant uint64_t& rb [[buffer(5)]],
    constant const uint* block_ids [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint first_compact_row = (tgid * 2u + sgitg) * 2u;
    if (first_compact_row >= (uint)packed_N) return;
    const bool row1_ok = first_compact_row + 1 < (uint)packed_N;
    const uint compact_block = first_compact_row >> 8;
    const uint source_block = SPARSE_AFFINE ? block_ids[0] + compact_block
                                             : block_ids[compact_block];
    const uint source_row0 = (source_block << 8) | (first_compact_row & 255u);
    const uint source_row1 = row1_ok ? source_row0 + 1 : source_row0;

    if (QUANT_TYPE == T_Q4_K) {
        constexpr uint16_t kmask1 = 0x3f3f;
        constexpr uint16_t kmask2 = 0x0f0f;
        constexpr uint16_t kmask3 = 0xc0c0;
        const short ix = lane / 8;
        const short it = lane % 8;
        const short iq = it / 4;
        const short ir = it % 4;
        const int nb = K / 256;
        float yl[16], yh[16];
        float sumf[2] = {0.f, 0.f};
        device const float* y4 = x + ix * 256 + 64 * iq + 8 * ir;
        uint16_t sc16[4];
        thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
        for (int ib = ix; ib < nb; ib += 4) {
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }
            for (short pair_row = 0; pair_row < 2; ++pair_row) {
                if (pair_row == 1 && !row1_ok) break;
                const uint source_row = pair_row == 0 ? source_row0 : source_row1;
                device const q4_K_blk* blk =
                    (device const q4_K_blk*)(W + (uint64_t)source_row * rb);
                device const uint16_t* sc = (device const uint16_t*)blk[ib].scales + iq;
                device const uint16_t* q1 = (device const uint16_t*)blk[ib].qs + 16 * iq + 4 * ir;
                device const half* dh = &blk[ib].d;
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                device const uint16_t* q2 = q1 + 32;
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                }
                sumf[pair_row] += float(dh[0]) *
                    ((acc1[0] + acc1[1] * (1.f / 256.f)) * sc8[0] +
                     (acc1[2] + acc1[3] * (1.f / 256.f)) * sc8[1] * (1.f / 16.f) +
                     (acc2[0] + acc2[1] * (1.f / 256.f)) * sc8[4] +
                     (acc2[2] + acc2[3] * (1.f / 256.f)) * sc8[5] * (1.f / 16.f)) -
                    float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                    sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }
            y4 += 4 * 256;
        }
        for (short pair_row = 0; pair_row < 2; ++pair_row) {
            if (pair_row == 1 && !row1_ok) break;
            const float sum = simd_sum(sumf[pair_row]);
            if (lane == 0) y[first_compact_row + pair_row] = sum;
        }
        return;
    }

    if (QUANT_TYPE == T_Q6_K) {
        const int nb = K / 256;
        device const q6_K_blk* wr0 =
            (device const q6_K_blk*)(W + (uint64_t)source_row0 * rb);
        device const q6_K_blk* wr1 =
            (device const q6_K_blk*)(W + (uint64_t)source_row1 * rb);
        const int l = (int)lane;
        const int is = l / 16;
        float acc0 = 0.f, acc1 = 0.f;
        for (int b = 0; b < nb; ++b) {
            const float d0 = float(wr0[b].d), d1 = float(wr1[b].d);
            for (int hb = 0; hb < 2; ++hb) {
                device const uchar* ql0 = wr0[b].ql + hb * 64;
                device const uchar* qh0 = wr0[b].qh + hb * 32;
                device const int8_t* sc0 = wr0[b].scales + hb * 8;
                device const uchar* ql1 = wr1[b].ql + hb * 64;
                device const uchar* qh1 = wr1[b].qh + hb * 32;
                device const int8_t* sc1 = wr1[b].scales + hb * 8;
                const int base = b * 256 + hb * 128;
                for (int g = 0; g < 4; ++g) {
                    const int qi = (g & 1) ? 32 + l : l;
                    const int shift = 2 * g;
                    const int lo0 = (g >= 2) ? (ql0[qi] >> 4) : (ql0[qi] & 0xF);
                    const int lo1 = (g >= 2) ? (ql1[qi] >> 4) : (ql1[qi] & 0xF);
                    const int q0 = (lo0 | (((qh0[l] >> shift) & 3) << 4)) - 32;
                    const int q1 = (lo1 | (((qh1[l] >> shift) & 3) << 4)) - 32;
                    const float xv = x[base + g * 32 + l];
                    acc0 += xv * d0 * float(sc0[g * 2 + is]) * float(q0);
                    acc1 += xv * d1 * float(sc1[g * 2 + is]) * float(q1);
                }
            }
        }
        acc0 = simd_sum(acc0);
        acc1 = simd_sum(acc1);
        if (lane == 0) {
            y[first_compact_row] = acc0;
            if (row1_ok) y[first_compact_row + 1] = acc1;
        }
    }
}

// The down projection reads selected K blocks from each original source row.
// The activation is compact, while the matrix row stride remains unchanged.
kernel void gemv_sparse_columns(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& full_K [[buffer(3)]],
    constant int& packed_K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant uint64_t& rb [[buffer(6)]],
    constant const uint* block_ids [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint first_row = (tgid * 2u + sgitg) * 2u;
    if (first_row >= (uint)N || full_K <= 0 || packed_K <= 0) return;
    const uint compact_blocks = (uint)packed_K / 256;
    const bool row1_ok = first_row + 1 < (uint)N;

    if (QUANT_TYPE == T_Q4_K) {
        constexpr uint16_t kmask1 = 0x3f3f;
        constexpr uint16_t kmask2 = 0x0f0f;
        constexpr uint16_t kmask3 = 0xc0c0;
        const short ix = lane / 8;
        const short it = lane % 8;
        const short iq = it / 4;
        const short ir = it % 4;
        float yl[16], yh[16];
        float sumf[2] = {0.f, 0.f};
        uint16_t sc16[4];
        thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
        for (uint compact_block = ix; compact_block < compact_blocks; compact_block += 4) {
            const uint source_block = SPARSE_AFFINE ? block_ids[0] + compact_block
                                                     : block_ids[compact_block];
            if (source_block >= (uint)full_K / 256) return;
            device const float* y4 = x + compact_block * 256 + 64 * iq + 8 * ir;
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }
            for (short pair_row = 0; pair_row < 2; ++pair_row) {
                if (pair_row == 1 && !row1_ok) break;
                device const q4_K_blk* row = (device const q4_K_blk*)
                    (W + (uint64_t)(first_row + pair_row) * rb);
                device const uint16_t* sc = (device const uint16_t*)row[source_block].scales + iq;
                device const uint16_t* q1 = (device const uint16_t*)row[source_block].qs + 16 * iq + 4 * ir;
                device const half* dh = &row[source_block].d;
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                device const uint16_t* q2 = q1 + 32;
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                }
                sumf[pair_row] += float(dh[0]) *
                    ((acc1[0] + acc1[1] * (1.f / 256.f)) * sc8[0] +
                     (acc1[2] + acc1[3] * (1.f / 256.f)) * sc8[1] * (1.f / 16.f) +
                     (acc2[0] + acc2[1] * (1.f / 256.f)) * sc8[4] +
                     (acc2[2] + acc2[3] * (1.f / 256.f)) * sc8[5] * (1.f / 16.f)) -
                    float(dh[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                    sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }
        }
        for (short pair_row = 0; pair_row < 2; ++pair_row) {
            if (pair_row == 1 && !row1_ok) break;
            const float sum = simd_sum(sumf[pair_row]);
            if (lane == 0) y[first_row + pair_row] = sum;
        }
        return;
    }

    if (QUANT_TYPE == T_Q6_K) {
        device const q6_K_blk* row0 =
            (device const q6_K_blk*)(W + (uint64_t)first_row * rb);
        device const q6_K_blk* row1 = row1_ok
            ? (device const q6_K_blk*)(W + (uint64_t)(first_row + 1) * rb) : row0;
        const int l = (int)lane;
        const int is = l / 16;
        float acc0 = 0.f, acc1 = 0.f;
        for (uint compact_block = 0; compact_block < compact_blocks; ++compact_block) {
            const uint source_block = SPARSE_AFFINE ? block_ids[0] + compact_block
                                                     : block_ids[compact_block];
            if (source_block >= (uint)full_K / 256) return;
            const float d0 = float(row0[source_block].d), d1 = float(row1[source_block].d);
            for (int hb = 0; hb < 2; ++hb) {
                device const uchar* ql0 = row0[source_block].ql + hb * 64;
                device const uchar* qh0 = row0[source_block].qh + hb * 32;
                device const int8_t* sc0 = row0[source_block].scales + hb * 8;
                device const uchar* ql1 = row1[source_block].ql + hb * 64;
                device const uchar* qh1 = row1[source_block].qh + hb * 32;
                device const int8_t* sc1 = row1[source_block].scales + hb * 8;
                const int base = compact_block * 256 + hb * 128;
                for (int g = 0; g < 4; ++g) {
                    const int qi = (g & 1) ? 32 + l : l;
                    const int shift = 2 * g;
                    const int lo0 = (g >= 2) ? (ql0[qi] >> 4) : (ql0[qi] & 0xF);
                    const int lo1 = (g >= 2) ? (ql1[qi] >> 4) : (ql1[qi] & 0xF);
                    const int q0 = (lo0 | (((qh0[l] >> shift) & 3) << 4)) - 32;
                    const int q1 = (lo1 | (((qh1[l] >> shift) & 3) << 4)) - 32;
                    const float xv = x[base + g * 32 + l];
                    acc0 += xv * d0 * float(sc0[g * 2 + is]) * float(q0);
                    acc1 += xv * d1 * float(sc1[g * 2 + is]) * float(q1);
                }
            }
        }
        acc0 = simd_sum(acc0);
        acc1 = simd_sum(acc1);
        if (lane == 0) {
            y[first_row] = acc0;
            if (row1_ok) y[first_row + 1] = acc1;
        }
    }
}

// The recurrent core keeps both input and output state bindings
// distinct. The first production lowering can therefore abandon a failed
// command without publishing partially updated state.
kernel void dnet_conv_silu(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* history_input [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* history_output [[buffer(4)]],
    constant int& channels [[buffer(5)]],
    constant int& kernel_size [[buffer(6)]],
    uint channel [[thread_position_in_grid]]
) {
    if (channel >= (uint)channels || kernel_size < 2) return;
    const int history = kernel_size - 1;
    device const float* wc = weight + (uint64_t)channel * kernel_size;
    device const float* hc = history_input + (uint64_t)channel * history;
    device float* ho = history_output + (uint64_t)channel * history;
    float sum = wc[kernel_size - 1] * input[channel];
    for (int tap = 0; tap < history; ++tap) sum += wc[tap] * hc[tap];
    output[channel] = sum / (1.0f + exp(-sum));
    for (int tap = 0; tap + 1 < history; ++tap) ho[tap] = hc[tap + 1];
    ho[history - 1] = input[channel];
}

kernel void dnet_l2_qk(
    device float* qkv [[buffer(0)]],
    constant int& qk_heads [[buffer(1)]],
    constant int& head_dimension [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (qk_heads < 1 || head_dimension < 1 || head_dimension > 128 ||
        head_dimension % 32 != 0 || group >= (uint)(2 * qk_heads)) return;
    const int part = (int)group / qk_heads;
    const int head = (int)group % qk_heads;
    device float* vector = qkv + ((uint64_t)part * qk_heads + head) * head_dimension;
    float sum = 0.0f;
    for (int column = (int)lane; column < head_dimension; column += 32)
        sum += vector[column] * vector[column];
    float inv = rsqrt(simd_sum(sum) + epsilon);
    if (part == 0) inv *= rsqrt(float(head_dimension));
    for (int column = (int)lane; column < head_dimension; column += 32)
        vector[column] *= inv;
}

// One simdgroup owns one value-head and four output columns per lane for D=128.
// State is [value_head][key_row][output_column], so reads are S^T times
// the key/query vectors.
inline float dnet_softplus(float value) {
    return value > 0.0f ? value + log(1.0f + exp(-value)) : log(1.0f + exp(value));
}

kernel void dnet_update(
    device const float* qkv [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* beta_projection [[buffer(2)]],
    device const float* alpha_projection [[buffer(3)]],
    device const float* dt_bias [[buffer(4)]],
    device const float* decay_weight [[buffer(5)]],
    device const float* norm_weight [[buffer(6)]],
    device const float* state_input [[buffer(7)]],
    device float* state_output [[buffer(8)]],
    device float* output [[buffer(9)]],
    constant int& qk_heads [[buffer(10)]],
    constant int& value_heads [[buffer(11)]],
    constant int& head_dimension [[buffer(12)]],
    constant float& rms_epsilon [[buffer(13)]],
    uint value_head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (qk_heads < 1 || value_heads < 1 || value_heads % qk_heads != 0 ||
        head_dimension < 1 || head_dimension > 128 || head_dimension % 32 != 0 ||
        value_head >= (uint)value_heads) return;
    const int key_head = (int)value_head % qk_heads;
    device const float* query = qkv + (uint64_t)key_head * head_dimension;
    device const float* key = qkv + (uint64_t)(qk_heads + key_head) * head_dimension;
    device const float* value = qkv + (uint64_t)(2 * qk_heads + value_head) * head_dimension;
    const float decay = exp(decay_weight[value_head] *
                            dnet_softplus(alpha_projection[value_head] + dt_bias[value_head]));
    const float beta = 1.0f / (1.0f + exp(-beta_projection[value_head]));
    const uint64_t state_base = (uint64_t)value_head * head_dimension * head_dimension;
    threadgroup float raw[128];
    for (int output_column = (int)lane; output_column < head_dimension; output_column += 32) {
        float retrieved = 0.0f;
        for (int key_row = 0; key_row < head_dimension; ++key_row) {
            const uint64_t index = state_base + (uint64_t)key_row * head_dimension + output_column;
            retrieved += state_input[index] * decay * key[key_row];
        }
        const float delta = beta * (value[output_column] - retrieved);
        float output_value = 0.0f;
        for (int key_row = 0; key_row < head_dimension; ++key_row) {
            const uint64_t index = state_base + (uint64_t)key_row * head_dimension + output_column;
            const float updated = state_input[index] * decay + key[key_row] * delta;
            state_output[index] = updated;
            output_value += updated * query[key_row];
        }
        raw[output_column] = output_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean_square_part = 0.0f;
    for (int column = (int)lane; column < head_dimension; column += 32)
        mean_square_part += raw[column] * raw[column];
    const float inv = rsqrt(simd_sum(mean_square_part) / float(head_dimension) + rms_epsilon);
    const uint64_t output_base = (uint64_t)value_head * head_dimension;
    for (int column = (int)lane; column < head_dimension; column += 32) {
        const float z = gate[output_base + column];
        output[output_base + column] = raw[column] * inv * norm_weight[column] * z / (1.0f + exp(-z));
    }
}

// Output-block-major affine UInt2 GEMV with exact zero-column elision.
// The selector, partial GEMV, and reduction are encoded in the token's one
// command buffer. The physical group index is output_block * K + column;
// this is deliberately distinct from gemv_affine_u2_256's row-major ABI.
kernel void column_grouped_affine_u2_skip_256_select(
    device const float* input [[buffer(0)]],
    device atomic_uint* active_count [[buffer(1)]],
    device uint* active_columns [[buffer(2)]],
    device atomic_uint* numerical_error [[buffer(3)]],
    device ulong* selected_encoded_bytes [[buffer(4)]],
    constant uint& logical_k [[buffer(5)]],
    constant uint& output_blocks [[buffer(6)]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uint offsets[256];
    const uint begin = uint((ulong(logical_k) * ulong(thread_index)) / 256ul);
    const uint end = uint((ulong(logical_k) * ulong(thread_index + 1u)) / 256ul);
    uint local_count = 0u;
    for (uint column = begin; column != end; ++column) {
        const float value = input[column];
        if (!isfinite(value))
            atomic_store_explicit(numerical_error, 1u, memory_order_relaxed);
        else if (value != 0.0f)
            ++local_count;
    }
    offsets[thread_index] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        const uint index = (thread_index + 1u) * (stride << 1u) - 1u;
        if (index < 256u) offsets[index] += offsets[index - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) offsets[255] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        const uint index = (thread_index + 1u) * (stride << 1u) - 1u;
        if (index < 256u) {
            const uint left = offsets[index - stride];
            offsets[index - stride] = offsets[index];
            offsets[index] += left;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint output_index = offsets[thread_index];
    for (uint column = begin; column != end; ++column) {
        const float value = input[column];
        if (isfinite(value) && value != 0.0f) active_columns[output_index++] = column;
    }
    if (thread_index == 255u) {
        const uint count = offsets[255] + local_count;
        atomic_store_explicit(active_count, count, memory_order_relaxed);
        *selected_encoded_bytes += ulong(count) * ulong(output_blocks) * 68ul;
    }
}

inline void column_grouped_affine_u2_skip_256_accumulate(
    device const uchar* values, device const uint* metadata,
    device const float* input, device const uint* active_columns,
    uint index, uint logical_k, uint group_base, uint simd_lane,
    thread float* sums0, thread float* sums1, thread float* bias_sum) {
    const uint column = active_columns[index];
    if (column >= logical_k) return;
    const uint group = group_base + column;
    const uint packed = uint(reinterpret_cast<device const ushort*>(
        values + ulong(group) * 64ul)[simd_lane]);
    const uint packed0 = packed & 255u;
    const uint packed1 = packed >> 8u;
    const float input_value = input[column];
    const half2 scale_bias = as_type<half2>(metadata[group]);
    const float scaled_input = input_value * float(scale_bias.x);
    *bias_sum += input_value * float(scale_bias.y);
    sums0[0] = fma(scaled_input, float(packed0 & 3u), sums0[0]);
    sums0[1] = fma(scaled_input, float((packed0 >> 2u) & 3u), sums0[1]);
    sums0[2] = fma(scaled_input, float((packed0 >> 4u) & 3u), sums0[2]);
    sums0[3] = fma(scaled_input, float(packed0 >> 6u), sums0[3]);
    sums1[0] = fma(scaled_input, float(packed1 & 3u), sums1[0]);
    sums1[1] = fma(scaled_input, float((packed1 >> 2u) & 3u), sums1[1]);
    sums1[2] = fma(scaled_input, float((packed1 >> 4u) & 3u), sums1[2]);
    sums1[3] = fma(scaled_input, float(packed1 >> 6u), sums1[3]);
}

kernel void column_grouped_affine_u2_skip_256_partial(
    device const uchar* values [[buffer(0)]],
    device const uint* metadata [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* partial_output [[buffer(3)]],
    device const atomic_uint* active_count [[buffer(4)]],
    device const uint* active_columns [[buffer(5)]],
    constant uint& logical_k [[buffer(6)]],
    constant uint& logical_n [[buffer(7)]],
    constant uint& split_count [[buffer(8)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[512];
    const uint output_block = threadgroup_index / split_count;
    const uint split = threadgroup_index - output_block * split_count;
    if (thread_index >= 64u || output_block >= (logical_n >> 8u)) return;
    uint count = 0u;
    if (simd_lane == 0u)
        count = min(atomic_load_explicit(active_count, memory_order_relaxed), logical_k);
    count = simd_broadcast_first(count);
    const uint simd_count = split_count * 2u;
    const uint simd_global = split * 2u + simd_index;
    const uint begin = (count * simd_global) / simd_count;
    const uint end = (count * (simd_global + 1u)) / simd_count;
    const uint group_base = output_block * logical_k;
    float sums0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sums1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float bias_sum = 0.0f;
    uint index = begin;
    for (; index + 3u < end; index += 4u) {
        const uint4 columns = uint4(active_columns[index], active_columns[index + 1u],
                                    active_columns[index + 2u], active_columns[index + 3u]);
        if (any(columns >= uint4(logical_k))) {
            for (uint tail = 0u; tail != 4u; ++tail)
                column_grouped_affine_u2_skip_256_accumulate(
                    values, metadata, input, active_columns, index + tail, logical_k,
                    group_base, simd_lane, sums0, sums1, &bias_sum);
            continue;
        }
        const uint4 groups = uint4(group_base) + columns;
        const float4 input_values = float4(input[columns.x], input[columns.y],
                                           input[columns.z], input[columns.w]);
        const half2 sb0 = as_type<half2>(metadata[groups.x]);
        const half2 sb1 = as_type<half2>(metadata[groups.y]);
        const half2 sb2 = as_type<half2>(metadata[groups.z]);
        const half2 sb3 = as_type<half2>(metadata[groups.w]);
        const float4 scaled_inputs = input_values *
            float4(float(sb0.x), float(sb1.x), float(sb2.x), float(sb3.x));
        const float4 biased_inputs = input_values *
            float4(float(sb0.y), float(sb1.y), float(sb2.y), float(sb3.y));
        bias_sum += biased_inputs.x + biased_inputs.y + biased_inputs.z + biased_inputs.w;
        const uint4 packed = uint4(
            uint(reinterpret_cast<device const ushort*>(
                values + ulong(groups.x) * 64ul)[simd_lane]),
            uint(reinterpret_cast<device const ushort*>(
                values + ulong(groups.y) * 64ul)[simd_lane]),
            uint(reinterpret_cast<device const ushort*>(
                values + ulong(groups.z) * 64ul)[simd_lane]),
            uint(reinterpret_cast<device const ushort*>(
                values + ulong(groups.w) * 64ul)[simd_lane]));
        uint4 codes = packed;
        for (uint part = 0u; part != 4u; ++part) {
            sums0[part] += dot(scaled_inputs, float4(codes & uint4(3u)));
            codes >>= uint4(2u);
        }
        for (uint part = 0u; part != 4u; ++part) {
            sums1[part] += dot(scaled_inputs, float4(codes & uint4(3u)));
            codes >>= uint4(2u);
        }
    }
    for (; index < end; ++index)
        column_grouped_affine_u2_skip_256_accumulate(
            values, metadata, input, active_columns, index, logical_k,
            group_base, simd_lane, sums0, sums1, &bias_sum);
    const uint local_base = simd_index * 256u + simd_lane * 8u;
    for (uint part = 0u; part != 4u; ++part) {
        partials[local_base + part] = sums0[part] + bias_sum;
        partials[local_base + 4u + part] = sums1[part] + bias_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row_in_block = thread_index * 4u;
    const ulong output_base =
        (ulong(output_block) * ulong(split_count) + ulong(split)) * 256ul;
    for (uint part = 0u; part != 4u; ++part)
        partial_output[output_base + ulong(row_in_block + part)] =
            partials[row_in_block + part] + partials[256u + row_in_block + part];
}

kernel void column_grouped_affine_u2_skip_256_reduce(
    device const float* partial_output [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& logical_n [[buffer(2)]],
    constant uint& split_count [[buffer(3)]],
    uint row_in_block [[thread_index_in_threadgroup]],
    uint output_block [[threadgroup_position_in_grid]]) {
    if (row_in_block >= 256u || output_block >= (logical_n >> 8u)) return;
    float sum = 0.0f;
    const ulong base = ulong(output_block) * ulong(split_count) * 256ul +
                       ulong(row_in_block);
    for (uint split = 0u; split != split_count; ++split)
        sum += partial_output[base + ulong(split) * 256ul];
    output[(output_block << 8u) + row_in_block] = sum;
}
)METAL";
