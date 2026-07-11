#include "../src/tensor.h"
#include "../src/kernels.h"
#include "../src/matmul.h"
#include "../src/fp16.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

using namespace Laplace;
using namespace Laplace::kernels;
namespace Laplace { extern bool metal_available(); extern bool metal_gemv(const float* x, const Tensor& w, float* y, int K, int N); }

static void rng_fill(float* v, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) { s = s*1103515245 + 12345; v[i] = (float)((s >> 16) % 200 - 100) / 10.0f; }
}

static uint16_t f2h(float f) { return fp32_to_fp16(f); }

// ponytail: quantize using the actual block layouts from kernels.h.
// One function per block family, parameterized by nothing. Universal.
static void pack_q4_0(const float* w, int K, int N, uint8_t* out) {
    int nb = K/32; auto blk = (block_q4_0*)out;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*32; float amax = 0;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/8; blk[(uint64_t)j*nb+b].d = f2h(d);
        float id = d > 0 ? 1/d : 0;
        for (int i = 0; i < 16; i++) {
            int q0 = roundf(x[i]*id+8), q1 = roundf(x[i+16]*id+8);
            blk[(uint64_t)j*nb+b].qs[i] = (q0&0xF)|((q1&0xF)<<4);
        }
    }
}
static void pack_q5_0(const float* w, int K, int N, uint8_t* out) {
    int nb = K/32; auto blk = (block_q5_0*)out;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*32; float amax = 0;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/16; blk[(uint64_t)j*nb+b].d = f2h(d); blk[(uint64_t)j*nb+b].qh = 0;
        float id = d > 0 ? 1/d : 0; uint32_t qh = 0;
        for (int i = 0; i < 16; i++) {
            int q0 = roundf(x[i]*id+16), q1 = roundf(x[i+16]*id+16);
            blk[(uint64_t)j*nb+b].qs[i] = (q0&0xF)|((q1&0xF)<<4);
            qh |= ((q0>>4)&1)<<i; qh |= ((q1>>4)&1)<<(i+16);
        }
        blk[(uint64_t)j*nb+b].qh = qh;
    }
}
static void pack_q8_0(const float* w, int K, int N, uint8_t* out) {
    int nb = K/32; auto blk = (block_q8_0*)out;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*32; float amax = 0;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/127; blk[(uint64_t)j*nb+b].d = f2h(d);
        float id = d > 0 ? 1/d : 0;
        for (int i = 0; i < 32; i++) blk[(uint64_t)j*nb+b].qs[i] = (int8_t)fmaxf(-127, fminf(127, roundf(x[i]*id)));
    }
}
static void pack_q4_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q4_K*)out;
    unsigned rs = 999;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/15; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d); bb.dmin = f2h(amax/240);
        for (int s = 0; s < 12; s++) { rs = rs*1103515245+12345; bb.scales[s] = (uint8_t)(rs >> 16); }
        float id = d > 0 ? 1/d : 0;
        for (int i = 0; i < 128; i++) {
            int q0 = roundf(x[i]*id), q1 = roundf(x[i+128]*id);
            bb.qs[i] = (q0&0xF)|((q1&0xF)<<4);
        }
    }
}
static void pack_q6_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q6_K*)out;
    unsigned rs = 777;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/31; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d);
        memset(bb.ql, 0, 128); memset(bb.qh, 0, 64);
        for (int s = 0; s < 16; s++) { rs = rs*1103515245+12345; bb.scales[s] = (int8_t)((rs >> 16) % 64 - 32); }
        float id = d > 0 ? 1/d : 0;
        for (int n = 0; n < 256; n++) {
            int val = fmaxf(0, fminf(63, roundf(x[n]*id+32)));
            int lo = val & 0xF, hi = (val >> 4) & 3;
            int half = n/128, l = n%128;
            if (l < 64) { bb.ql[half*64 + l] = (bb.ql[half*64 + l] & 0xF0) | lo; }
            else { bb.ql[half*64 + l - 64] = (bb.ql[half*64 + l - 64] & 0x0F) | (lo << 4); }
            int qh_byte = half*32 + l%32;
            bb.qh[qh_byte] |= hi << ((l/16)*2);
        }
    }
}
static void pack_q2_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q2_K*)out;
    unsigned rs = 222;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/3; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d); bb.dmin = f2h(amax/15);
        for (int s = 0; s < 16; s++) { rs = rs*1103515245+12345; bb.scales[s] = (uint8_t)(rs >> 16); }
        float id = d > 0 ? 1/d : 0;
        memset(bb.qs, 0, 64);
        for (int e = 0; e < 256; e++) {
            int q = fmaxf(0, fminf(3, roundf(x[e]*id)));
            int sub = e/16, l = e%16;
            int half = sub/8, jj = sub%8, group = jj/2, lo = jj%2;
            bb.qs[half*32 + lo*16 + l] |= (uint8_t)(q << (group*2));
        }
    }
}
static void pack_q3_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q3_K*)out;
    unsigned rs = 333;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/4; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d);
        for (int s = 0; s < 12; s++) { rs = rs*1103515245+12345; bb.scales[s] = (uint8_t)(rs >> 16); }
        memset(bb.qs, 0, 64); memset(bb.hmask, 0, 32);
        float id = d > 0 ? 1/d : 0;
        for (int e = 0; e < 256; e++) {
            int q = (int)fmaxf(-4, fminf(3, roundf(x[e]*id)));
            int low2, hbit;
            if (q >= 0) { low2 = q; hbit = 1; } else { low2 = q + 4; hbit = 0; }
            int g = e/32, l = e%32;
            bb.qs[(g/4)*32 + l] |= (uint8_t)(low2 << ((g%4)*2));
            if (hbit) bb.hmask[l] |= (uint8_t)(1 << g);
        }
    }
}
static void pack_q5_K(const float* w, int K, int N, uint8_t* out) {
    int nb = K/256; auto blk = (block_q5_K*)out;
    unsigned rs = 444;
    for (int j = 0; j < N; j++) for (int b = 0; b < nb; b++) {
        const float* x = w + j*K + b*256; float amax = 0;
        for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[i]));
        float d = amax/31; auto& bb = blk[(uint64_t)j*nb+b]; bb.d = f2h(d); bb.dmin = f2h(amax/480);
        for (int s = 0; s < 12; s++) { rs = rs*1103515245+12345; bb.scales[s] = (uint8_t)(rs >> 16); }
        memset(bb.qs, 0, 128); memset(bb.qh, 0, 32);
        float id = d > 0 ? 1/d : 0;
        for (int e = 0; e < 256; e++) {
            int q = (int)fmaxf(0, fminf(31, roundf(x[e]*id)));
            int lo = q & 0xF, hi = (q >> 4) & 1;
            int jb = e/64, within = e%64;
            if (within < 32) {
                bb.qs[jb*32 + within] |= (uint8_t)lo;
                if (hi) bb.qh[within] |= (uint8_t)(1 << (2*jb));
            } else {
                int l = within - 32;
                bb.qs[jb*32 + l] |= (uint8_t)(lo << 4);
                if (hi) bb.qh[l] |= (uint8_t)(1 << (2*jb+1));
            }
        }
    }
}

int main() {
    if (!metal_available()) { fprintf(stderr, "SKIP: no Metal\n"); return 0; }
    fprintf(stderr, "[metal] device available\n");

    const int K = 2816, N = 512;
    std::vector<float> x(K), w(K*N), y_gpu(N), y_ref(N);
    rng_fill(x.data(), K, 42);
    rng_fill(w.data(), K*N, 123);

    struct TC { GGMLType type; const char* name; void (*pack)(const float*, int, int, uint8_t*); bool is_float; };
    TC tests[] = {
        {GGMLType::F32, "F32", nullptr, true},
        {GGMLType::Q4_0, "Q4_0", pack_q4_0, false},
        {GGMLType::Q5_0, "Q5_0", pack_q5_0, false},
        {GGMLType::Q8_0, "Q8_0", pack_q8_0, false},
        {GGMLType::Q4_K, "Q4_K", pack_q4_K, false},
        {GGMLType::Q6_K, "Q6_K", pack_q6_K, false},
        {GGMLType::Q2_K, "Q2_K", pack_q2_K, false},
        {GGMLType::Q3_K, "Q3_K", pack_q3_K, false},
        {GGMLType::Q5_K, "Q5_K", pack_q5_K, false},
    };

    int pass = 0, fail = 0;
    for (auto& tc : tests) {
        size_t bpb = bytes_per_block(tc.type), epb = elements_per_block(tc.type);
        size_t nblocks = ((size_t)K*N + epb - 1) / epb;
        std::vector<uint8_t> storage(nblocks * bpb, 0);
        Tensor wt; wt.type = tc.type; wt.n_dims = 2; wt.dims[0] = K; wt.dims[1] = N; wt.data = storage.data();
        if (tc.is_float) memcpy(storage.data(), w.data(), (size_t)K*N*4);
        else tc.pack(w.data(), K, N, storage.data());

        setenv("LAPLACE_NOSIMD", "1", 1); setenv("LAPLACE_NOMETAL", "1", 1);
        matmul_rows(x.data(), wt, y_ref.data(), 1, K, N);
        unsetenv("LAPLACE_NOSIMD"); unsetenv("LAPLACE_NOMETAL");

        if (!metal_gemv(x.data(), wt, y_gpu.data(), K, N)) {
            fprintf(stderr, "FAIL %s: not supported\n", tc.name); fail++; continue;
        }

        float max_err = 0, max_val = 0;
        for (int j = 0; j < N; j++) { max_err = fmaxf(max_err, fabsf(y_ref[j]-y_gpu[j])); max_val = fmaxf(max_val, fabsf(y_ref[j])); }
        float rel = max_val > 0 ? max_err/max_val : 0;
        if (rel > 0.01f) { fprintf(stderr, "FAIL %s: rel err %.6f (err=%.2f val=%.1f)\n", tc.name, rel, max_err, max_val); fail++; }
        else { fprintf(stderr, "PASS %s: rel err %.6f\n", tc.name, rel); pass++; }
    }
    fprintf(stderr, "%d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
