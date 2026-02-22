#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void k_fp32_to_fp16(const float * __restrict__ in,
                                __half * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

extern "C" void gpu_fp32_to_fp16(const float * in, void * out, int n,
                                  cudaStream_t stream) {
    k_fp32_to_fp16<<<(n + 255) / 256, 256, 0, stream>>>(in, (__half *)out, n);
}


// ── GPU argmax: find index of maximum value in FP32 array ──
// Single block, up to 1024 threads. For vocab_size <= 32768.
__global__ void k_argmax_f32(const float * __restrict__ in,
                              int32_t * __restrict__ out, int n) {
    __shared__ float s_max[32];
    __shared__ int s_idx[32];

    int tid = threadIdx.x;
    float local_max = -1e30f;
    int local_idx = 0;

    for (int i = tid; i < n; i += blockDim.x) {
        float v = in[i];
        if (v > local_max) { local_max = v; local_idx = i; }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        int oidx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
        if (other > local_max) { local_max = other; local_idx = oidx; }
    }

    int warp_id = tid >> 5;
    if ((tid & 31) == 0) {
        s_max[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // Final reduction in first warp
    int n_warps = (blockDim.x + 31) >> 5;
    if (tid < 32) {
        local_max = (tid < n_warps) ? s_max[tid] : -1e30f;
        local_idx = (tid < n_warps) ? s_idx[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            int oidx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other > local_max) { local_max = other; local_idx = oidx; }
        }
        if (tid == 0) *out = local_idx;
    }
}

extern "C" void gpu_argmax_f32(const float * in, int32_t * out, int n,
                                cudaStream_t stream) {
    int threads = (n < 1024) ? n : 1024;
    k_argmax_f32<<<1, threads, 0, stream>>>(in, out, n);
}

// ── GPU embedding lookup by GPU-resident token ID ──
// Reads token_id from device memory, copies one embedding row to output.
__global__ void k_embedding_lookup_gpu(const int32_t * __restrict__ token_id_ptr,
                                        const float * __restrict__ table,
                                        float * __restrict__ output,
                                        int embd_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < embd_dim) {
        output[i] = table[(*token_id_ptr) * embd_dim + i];
    }
}

extern "C" void gpu_embedding_lookup_by_gpu_id(const int32_t * token_id_ptr,
                                                const float * table,
                                                float * output,
                                                int embd_dim,
                                                cudaStream_t stream) {
    k_embedding_lookup_gpu<<<(embd_dim + 255) / 256, 256, 0, stream>>>(
        token_id_ptr, table, output, embd_dim);
}


// ── GPU top-k + temperature sampling from FP32 logits ──
// Single block, 1024 threads, vocab_size <= 2048 (2 elements/thread).
// All computation in shared memory. Serial CDF scan for final sampling.
__global__ void k_sample_topk_f32(const float * __restrict__ logits,
                                   const float * __restrict__ rand_val,
                                   int32_t * __restrict__ out,
                                   float temperature,
                                   int32_t top_k,
                                   int32_t vocab_size) {
    __shared__ float s[2048];
    __shared__ int32_t s_counts[2048];
    __shared__ float s_warp[32];

    int tid = threadIdx.x;
    int n = vocab_size;

    // Phase 1: Load logits and apply temperature scaling
    for (int i = tid; i < n; i += blockDim.x) {
        s[i] = logits[i] / temperature;
    }
    __syncthreads();

    // Phase 2: Top-k filtering
    if (top_k > 0 && top_k < n) {
        // Count how many elements are strictly greater (read-only on s[])
        for (int i = tid; i < n; i += blockDim.x) {
            int count = 0;
            float val = s[i];
            for (int j = 0; j < n; j++) {
                if (s[j] > val) count++;
            }
            s_counts[i] = count;
        }
        __syncthreads();
        // Mask elements outside top-k
        for (int i = tid; i < n; i += blockDim.x) {
            if (s_counts[i] >= top_k) s[i] = -1e30f;
        }
        __syncthreads();
    }

    // Phase 3: Find max for numerical stability (parallel warp reduction)
    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, s[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    if ((tid & 31) == 0) s_warp[tid >> 5] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = -1e30f;
        int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; i++) m = fmaxf(m, s_warp[i]);
        s_warp[0] = m;
    }
    __syncthreads();
    float global_max = s_warp[0];

    // Phase 4: Compute exp(logit - max) and sum (parallel warp reduction)
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = expf(s[i] - global_max);
        s[i] = val;
        local_sum += val;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if ((tid & 31) == 0) s_warp[tid >> 5] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; i++) total += s_warp[i];
        s_warp[0] = total;
    }
    __syncthreads();
    float total_sum = s_warp[0];

    // Phase 5: Serial CDF scan + sampling in thread 0
    // (2048 adds = ~2 us, negligible vs 4ms TRT step)
    if (tid == 0) {
        float target = (*rand_val) * total_sum;
        float cumsum = 0.0f;
        int result = n - 1;
        for (int i = 0; i < n; i++) {
            cumsum += s[i];
            if (cumsum >= target) { result = i; break; }
        }
        *out = result;
    }
}

extern "C" void gpu_sample_topk_f32(const float * logits, const float * rand_val,
                                     int32_t * out, float temperature,
                                     int32_t top_k, int32_t vocab_size,
                                     cudaStream_t stream) {
    int threads = (vocab_size < 1024) ? vocab_size : 1024;
    k_sample_topk_f32<<<1, threads, 0, stream>>>(logits, rand_val, out,
                                                  temperature, top_k, vocab_size);
}
