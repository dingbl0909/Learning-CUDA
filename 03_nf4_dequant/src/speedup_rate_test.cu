#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// -------------- 1. 对齐 bitsandbytes 的常量定义 --------------
#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// bitsandbytes 官方 NF4 查表值（必须对齐）
__constant__ float c_nf4[16] = {
    -1.0000f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07930000126361847f, 0.16093000769615173f, 0.24630000472068787f, 0.3379150033000702f,
    0.43870002031326294f, 0.5546000003814697f, 0.6961999535560608f, 1.0f
};

// -------------- 2. 你的 NF4 反量化核函数（保留原逻辑）--------------
__global__ void custom_nf4_dequantize_kernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const half*    __restrict__ absmax2,
    const half*    __restrict__ code2,
    float          offset,
    half*          __restrict__ output,
    int64_t        total_elements,
    int            blocksize,
    int            group_size)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t num_pairs = (total_elements + 1) / 2;
    if (tid >= num_pairs) return;

    int64_t elem0 = tid * 2;

    // Unpack two 4-bit NF4 indices from one byte
    uint8_t packed = packed_weights[tid];
    uint8_t idx_hi = (packed >> 4) & 0x0F;  // upper nibble -> even element
    uint8_t idx_lo = packed & 0x0F;        // lower nibble -> odd element

    // ── Recover block-level scale for element 0 ──
    int blk0  = static_cast<int>(elem0 / blocksize);
    int grp0  = blk0 / group_size;
    float sc0 = __half2float(code2[absmax_q[blk0]])
              * __half2float(absmax2[grp0])
              + offset;

    half h0 = __float2half(c_nf4[idx_hi] * sc0);

    // ── Element 1 (with boundary guard) ──
    if (elem0 + 1 < total_elements) {
        int blk1 = static_cast<int>((elem0 + 1) / blocksize);
        float sc1 = sc0;
        if (blk1 != blk0) {
            int grp1 = blk1 / group_size;
            sc1 = __half2float(code2[absmax_q[blk1]])
                * __half2float(absmax2[grp1])
                + offset;
        }
        half h1 = __float2half(c_nf4[idx_lo] * sc1);

        // Vectorized 32-bit store: two fp16 values packed into one uint32_t write.
        uint32_t packed_out = static_cast<uint32_t>(__half_as_ushort(h0))
                            | (static_cast<uint32_t>(__half_as_ushort(h1)) << 16);
        reinterpret_cast<uint32_t*>(output)[tid] = packed_out;
    } else {
        // Last element when total is odd — scalar store
        output[elem0] = h0;
    }
}

// -------------- 3. 调用 bitsandbytes 的 NF4 反量化 --------------
// 声明 bitsandbytes 的 NF4 反量化函数（适配你的输入格式）
extern "C" void nf4_dequantize_cuda_bnb(
    const uint8_t* packed_weights,
    const uint8_t* absmax_q,
    const half*    absmax2,
    const half*    code2,
    float          offset,
    half*          output,
    int64_t        total_elements,
    int            blocksize,
    int            group_size,
    cudaStream_t   stream);

// 封装 bnb 调用，适配你的参数
void bnb_nf4_dequantize(
    const uint8_t* packed_weights,
    const uint8_t* absmax_q,
    const half*    absmax2,
    const half*    code2,
    float          offset,
    half*          output,
    int64_t        total_elements,
    int            blocksize,
    int            group_size,
    cudaStream_t   stream = 0) {
    nf4_dequantize_cuda_bnb(packed_weights, absmax_q, absmax2, code2, offset,
                            output, total_elements, blocksize, group_size, stream);
}

// -------------- 4. 通用性能测试函数 --------------
float benchmark_kernel(
    const uint8_t* d_packed,
    const uint8_t* d_absmax_q,
    const half*    d_absmax2,
    const half*    d_code2,
    float          offset,
    half*          d_output,
    int64_t        total,
    int            blocksize,
    int            group_size,
    bool           use_custom,
    int            warmup,
    int            iters) {

    const int threads_per_block = 256;
    int64_t num_pairs = (total + 1) / 2;
    int grid_size = static_cast<int>((num_pairs + threads_per_block - 1) / threads_per_block);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 预热
    for (int i = 0; i < warmup; ++i) {
        if (use_custom) {
            custom_nf4_dequantize_kernel<<<grid_size, threads_per_block, 0, stream>>>(
                d_packed, d_absmax_q, d_absmax2, d_code2, offset,
                d_output, total, blocksize, group_size);
        } else {
            bnb_nf4_dequantize(
                d_packed, d_absmax_q, d_absmax2, d_code2, offset,
                d_output, total, blocksize, group_size, stream);
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 计时
    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventRecord(ev_start, stream));

    for (int i = 0; i < iters; ++i) {
        if (use_custom) {
            custom_nf4_dequantize_kernel<<<grid_size, threads_per_block, 0, stream>>>(
                d_packed, d_absmax_q, d_absmax2, d_code2, offset,
                d_output, total, blocksize, group_size);
        } else {
            bnb_nf4_dequantize(
                d_packed, d_absmax_q, d_absmax2, d_code2, offset,
                d_output, total, blocksize, group_size, stream);
        }
    }

    CHECK_CUDA(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    float avg_ms = elapsed_ms / iters;

    // 清理
    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return avg_ms;
}

// -------------- 5. 主函数（保留你的输入逻辑，增加对比）--------------
int main(int argc, char** argv) {
    const char* input_path  = get_arg(argc, argv, "--input");
    const char* output_path = get_arg(argc, argv, "--output");
    int group_size = get_int_arg(argc, argv, "--group_size", 256);
    int warmup     = get_int_arg(argc, argv, "--warmup", 3);
    int iters      = get_int_arg(argc, argv, "--iters", 10);

    if (!input_path || !output_path) {
        fprintf(stderr,
                "Usage: %s --input BIN --output BIN "
                "[--compute_type fp16|bf16] [--group_size N] "
                "[--warmup N] [--iters N]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    // ── 1. 读取输入（保留你的原逻辑） ────────────────────────────────────
    FILE* fin = fopen(input_path, "rb");
    if (!fin) { perror("fopen(input)"); return EXIT_FAILURE; }

    int64_t num_rows, num_cols;
    int32_t blocksize;
    fread(&num_rows,  sizeof(int64_t), 1, fin);
    fread(&num_cols,  sizeof(int64_t), 1, fin);
    fread(&blocksize, sizeof(int32_t), 1, fin);

    int64_t total     = num_rows * num_cols;
    int     num_blocks = static_cast<int>((total + blocksize - 1) / blocksize);
    int64_t num_packed = (total + 1) / 2;
    int     num_groups = (num_blocks + group_size - 1) / group_size;

    printf("Shape: %ldx%ld  blocksize=%d  group_size=%d\n",
           (long)num_rows, (long)num_cols, blocksize, group_size);
    printf("Elements=%ld  packed_bytes=%ld  blocks=%d  groups=%d\n",
           (long)total, (long)num_packed, num_blocks, num_groups);

    std::vector<uint8_t>  h_packed(num_packed);
    std::vector<uint8_t>  h_absmax_q(num_blocks);
    std::vector<uint16_t> h_absmax2(num_groups);
    std::vector<uint16_t> h_code2(256);
    float h_offset = 0.0f;

    fread(h_packed.data(),   1,              num_packed,  fin);
    fread(h_absmax_q.data(), 1,              num_blocks,  fin);
    fread(h_absmax2.data(),  sizeof(uint16_t), num_groups, fin);
    fread(h_code2.data(),    sizeof(uint16_t), 256,        fin);
    fread(&h_offset,         sizeof(float),    1,          fin);
    fclose(fin);

    printf("Offset: %.6f\n", h_offset);     

    // ── 2. 设备内存分配（保留原逻辑） ─────────────────────────────────────
    uint8_t* d_packed    = nullptr;
    uint8_t* d_absmax_q  = nullptr;
    half*    d_absmax2   = nullptr;
    half*    d_code2     = nullptr;
    half*    d_output_custom = nullptr;
    half*    d_output_bnb = nullptr;

    CHECK_CUDA(cudaMalloc(&d_packed,   num_packed));
    CHECK_CUDA(cudaMalloc(&d_absmax_q, num_blocks));
    CHECK_CUDA(cudaMalloc(&d_absmax2,  num_groups * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_code2,    256 * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_output_custom,   total * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_output_bnb,   total * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_packed,   h_packed.data(),   num_packed, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax_q, h_absmax_q.data(), num_blocks, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2,  h_absmax2.data(),  num_groups * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_code2,    h_code2.data(),    256 * sizeof(half), cudaMemcpyHostToDevice));

    // ── 3. 性能对比测试 ──────────────────────────────────────────────────
    printf("\n=== 性能对比 (自定义 vs bitsandbytes) ===\n");
    // 测试自定义实现
    float custom_avg_ms = benchmark_kernel(
        d_packed, d_absmax_q, d_absmax2, d_code2, h_offset,
        d_output_custom, total, blocksize, group_size,
        true, warmup, iters);
    // 测试 bitsandbytes 实现
    float bnb_avg_ms = benchmark_kernel(
        d_packed, d_absmax_q, d_absmax2, d_code2, h_offset,
        d_output_bnb, total, blocksize, group_size,
        false, warmup, iters);

    // 计算加速比和带宽
    float speedup = bnb_avg_ms / custom_avg_ms;
    double bytes_in  = static_cast<double>(num_packed + num_blocks + num_groups * 2 + 256 * 2);
    double bytes_out = static_cast<double>(total) * 2;
    double custom_bw = (bytes_in + bytes_out) / (custom_avg_ms * 1e6);
    double bnb_bw = (bytes_in + bytes_out) / (bnb_avg_ms * 1e6);

    // 输出对比结果
    printf("自定义实现：%.4f ms, 带宽 %.2f GB/s\n", custom_avg_ms, custom_bw);
    printf("bitsandbytes：%.4f ms, 带宽 %.2f GB/s\n", bnb_avg_ms, bnb_bw);
    printf("加速比 (自定义/bnb)：%.2f x\n", speedup);
    if (speedup > 1.0) {
        printf("✅ 自定义实现更快，比 bnb 快 %.2f 倍\n", speedup);
    } else {
        printf("❌ bnb 更快，比自定义快 %.2f 倍\n", 1.0/speedup);
    }

    // ── 4. 数值正确性验证 ────────────────────────────────────────────────
    std::vector<uint16_t> h_out_custom(total);
    std::vector<uint16_t> h_out_bnb(total);
    CHECK_CUDA(cudaMemcpy(h_out_custom.data(), d_output_custom, total * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_bnb.data(), d_output_bnb, total * sizeof(half), cudaMemcpyDeviceToHost));

    // 计算最大误差
    float max_error = 0.0f;
    for (int64_t i = 0; i < total; ++i) {
        half h_custom = __ushort_as_half(h_out_custom[i]);
        half h_bnb = __ushort_as_half(h_out_bnb[i]);
        float f_custom = __half2float(h_custom);
        float f_bnb = __half2float(h_bnb);
        max_error = fmax(max_error, fabs(f_custom - f_bnb));
    }
    printf("数值最大误差：%.6f (阈值 < 1e-6)\n", max_error);
    if (max_error < 1e-6) {
        printf("✅ 数值验证通过\n");
    } else {
        printf("❌ 数值误差过大，需检查逻辑\n");
    }

    // ── 5. 写输出（保留原逻辑） ──────────────────────────────────────────
    FILE* fout = fopen(output_path, "wb");
    if (!fout) { perror("fopen(output)"); return EXIT_FAILURE; }
    fwrite(h_out_custom.data(), sizeof(uint16_t), total, fout);
    fclose(fout);

    // ── 6. 清理 ───────────────────────────────────────────────────────────
    CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_absmax_q));
    CHECK_CUDA(cudaFree(d_absmax2));
    CHECK_CUDA(cudaFree(d_code2));
    CHECK_CUDA(cudaFree(d_output_custom));
    CHECK_CUDA(cudaFree(d_output_bnb));

    return 0;
}

// -------------- 6. 保留你的 CLI 辅助函数 --------------
static const char* get_arg(int argc, char** argv, const char* key,
                           const char* fallback = nullptr) {
    for (int i = 1; i < argc - 1; ++i)
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    return fallback;
}

static int get_int_arg(int argc, char** argv, const char* key, int fallback) {
    const char* v = get_arg(argc, argv, key);
    return v ? atoi(v) : fallback;
}