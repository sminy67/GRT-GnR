#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <THC/THCAtomics.cuh>
#include <mutex>
#include "grt_cuda_utils.cuh"

using namespace at;

inline void cuda_gemm_batched_fp32_fp32(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float* alpha,
    void** a_array,
    int lda,
    void** b_array,
    int ldb,
    float* beta,
    void** c_array,
    int ldc,
    int batch_count) {
        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(handle, c10::cuda::getCurrentCUDAStream());
        TORCH_CUDABLAS_CHECK(cublasGemmBatchedEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a_array,
            CUDA_R_32F,
            lda,
            b_array,
            CUDA_R_32F,
            ldb,
            beta,
            c_array,
            CUDA_R_32F,
            ldc,
            batch_count,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT));
    }

__global__ void init_batch_gemm_forward_2T_kernel(
    int32_t N,
    const int64_t* __restrict__ tt_idx_shapes,
    const int64_t* __restrict__ indices,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_0,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
        int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n < N) {
            auto idx = __ldg(&indices[n]);
            auto tt_idx_0 = idx / tt_idx_shapes[0];
            auto tt_idx_1 = idx % tt_idx_shapes[1];
            a_ptr[0 * N + n] = (float*)&(tt_cores_0[tt_idx_0][0]);
            b_ptr[0 * N + n] = (float*)&(tt_cores_1[tt_idx_1][0]);
            c_ptr[0 * N + n] = (float*)&(tr_0[n][0]);
        }
    }

__global__ void init_batch_gemm_forward_3T_kernel(
    int32_t N,
    const int64_t* __restrict__ tt_idx_shapes,
    const int64_t* __restrict__ indices,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_1,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
        int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n < N) {
            auto idx = __ldg(&indices[n]);
            auto tt_idx_0 = idx / tt_idx_shapes[0];
            idx = idx % tt_idx_shapes[0];
            auto tt_idx_1 = idx / tt_idx_shapes[1];
            auto tt_idx_2 = idx % tt_idx_shapes[1];
            float* tr_0_ptr = (float*)&(tr_0[n][0]);
            a_ptr[0 * N + n] = (float*)&(tt_cores_0[tt_idx_0][0]);
            b_ptr[0 * N + n] = (float*)&(tt_cores_1[tt_idx_1][0]);
            c_ptr[0 * N + n] = tr_0_ptr;
            a_ptr[1 * N + n] = tr_0_ptr;
            b_ptr[1 * N + n] = (float*)&(tt_cores_2[tt_idx_2][0]);
            c_ptr[1 * N + n] = (float*)&(tr_1[n][0]);
        }
    }

__global__ void init_grouped_batch_gemm_forward_2T_kernel(
    int32_t N,
    const int64_t* __restrict__ tt_idx_shapes,
    const int64_t* __restrict__ inter_group_indices,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> intra_group_output,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_0,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
        int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n < N) {
            auto tt_idx_0 = __ldg(&inter_group_indices[n]);
            a_ptr[0 * N + n] = (float*)&(tt_cores_0[tt_idx_0][0]);
            b_ptr[0 * N + n] = (float*)&(intra_group_output[n][0]);
            c_ptr[0 * N + n] = (float*)&(tr_0[n][0]);
        }
    }

__global__ void init_grouped_batch_gemm_forward_3T_kernel(
    int32_t N,
    const int64_t* __restrict__ tt_idx_shapes,
    const int64_t* __restrict__ inter_group_indices,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> intra_group_output,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> tr_1,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
        int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n < N) {
            auto gidx = __ldg(&inter_group_indices[n]);
            auto tt_idx_0 = gidx / tt_idx_shapes[0];
            auto tt_idx_1 = gidx % tt_idx_shapes[0];
            float* tr_0_ptr = (float*)&(tr_0[n][0]);
            a_ptr[0 * N + n] = (float*)&(tt_cores_0[tt_idx_0][0]);
            b_ptr[0 * N + n] = (float*)&(tt_cores_1[tt_idx_1][0]);
            c_ptr[0 * N + n] = tr_0_ptr;
            a_ptr[1 * N + n] = tr_0_ptr;
            b_ptr[1 * N + n] = (float*)&(intra_group_output[n][0]);
            c_ptr[1 * N + n] = (float*)&(tr_1[n][0]);
        }
    }

void init_batch_gemm_forward_cuda(
    int32_t num_cores,
    int32_t nnz,
    const int64_t* __restrict__ tt_idx_shapes,
    const int64_t* __restrict__ indices,
    const std::vector<Tensor>& tt_cores,
    const std::vector<Tensor>& tr,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
        int32_t threads = (nnz > 512 ? 512 : 256);
        int32_t num_blocks = ((nnz + threads - 1) / threads);
        if (num_cores == 2) {
            init_batch_gemm_forward_2T_kernel<<<num_blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                nnz,
                tt_idx_shapes,
                indices,
                tt_cores[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
                tt_cores[1].packed_accessor64<float, 2, RestrictPtrTraits>(),
                tr[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
                a_ptr,
                b_ptr,
                c_ptr);
        } else if (num_cores == 3) {
            init_batch_gemm_forward_3T_kernel<<<num_blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                nnz,
                tt_idx_shapes,
                indices,
                tt_cores[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
                tt_cores[1].packed_accessor64<float, 2, RestrictPtrTraits>(),
                tt_cores[2].packed_accessor64<float, 2, RestrictPtrTraits>(),
                tr[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
                tr[1].packed_accessor64<float, 2, RestrictPtrTraits>(),
                a_ptr,
                b_ptr,
                c_ptr);
        }
    }

void init_grouped_batch_gemm_forward_cuda(
    int32_t num_cores,
    int32_t inter_group_nnz,
    const int64_t* __restrict__ tt_idx_shapes,
    const int64_t* __restrict__ inter_group_indices,
    const std::vector<Tensor>& tt_cores,
    const std::vector<Tensor>& tr,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> intra_group_output,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
        int32_t threads = (inter_group_nnz > 512 ? 512: 256);
        int32_t num_blocks = ((inter_group_nnz + threads - 1) / threads);
        if (num_cores == 2) {
            init_grouped_batch_gemm_forward_2T_kernel<<<num_blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                inter_group_nnz,
                tt_idx_shapes,
                inter_group_indices,
                tt_cores[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
                intra_group_output,
                tr[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
                a_ptr,
                b_ptr,
                c_ptr);
        } else if (num_cores == 3) {
            init_grouped_batch_gemm_forward_3T_kernel<<<num_blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                inter_group_nnz,
                tt_idx_shapes,
                inter_group_indices,
                tt_cores[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
                tt_cores[1].packed_accessor64<float, 2, RestrictPtrTraits>(),
                intra_group_output,
                tr[0].packed_accessor64<float, 2, RestrictPtrTraits>(),
                tr[1].packed_accessor64<float, 2, RestrictPtrTraits>(),
                a_ptr,
                b_ptr,
                c_ptr);
        }
    }

__global__ void init_intra_group_reduce_forward_kernel(
    int32_t N,
    const int64_t* __restrict__ intra_group_indices,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> last_tt_core,
    float** __restrict__ intra_group_ptr
) {
    int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        auto idx = __ldg(&intra_group_indices[n]);
        intra_group_ptr[n] = (float*)&(last_tt_core[idx][0]);
    }
}

__global__ void intra_group_reduce_kernel(
    int32_t N,
    int32_t D,
    int32_t num_group_bags,
    const int64_t* __restrict__ intra_group_offsets,
    float** __restrict__ intra_group_tensor,
    float* __restrict__ intra_group_output
) {
    int64_t chunks_per_bag = div_round_up(D, (int64_t)blockDim.x);
    int64_t total_chunks = num_group_bags * chunks_per_bag;
    int64_t chunk_offset = blockIdx.x * blockDim.y + threadIdx.y;
    int64_t chunk_stride = gridDim.x * blockDim.y;

    for (int64_t chunk = chunk_offset; chunk < total_chunks; chunk += chunk_stride) {
        int64_t colidx = (chunk % chunks_per_bag) * blockDim.x + threadIdx.x;
        if (colidx < D) {
            int64_t bag = chunk / chunks_per_bag;
            int64_t begin = intra_group_offsets[bag];
            int64_t end = (bag < num_group_bags - 1) ? (intra_group_offsets[bag + 1]) : N;
            float weight_sum = 0;
            for (int64_t rowidx = begin; rowidx < end; rowidx++) {
                weight_sum += intra_group_tensor[rowidx][colidx];
            }
            intra_group_output[bag * D + colidx] = weight_sum;
        }
    }
}

__global__ void inter_group_reduce_kernel(
    int32_t N,
    int32_t D,
    int32_t num_bags,
    const int64_t* __restrict__ offsets,
    const float* __restrict__ tr_last,
    float* __restrict__ output) {
        int64_t chunks_per_bag = div_round_up(D, (int64_t)blockDim.x);
        int64_t total_chunks = num_bags * chunks_per_bag;
        int64_t chunk_offset = blockIdx.x * blockDim.y + threadIdx.y;
        int64_t chunk_stride = gridDim.x * blockDim.y;

        for (int64_t chunk = chunk_offset; chunk < total_chunks; chunk += chunk_stride) {
            int64_t colidx = (chunk % chunks_per_bag) * blockDim.x + threadIdx.x;
            if (colidx < D) {
                int64_t bag = chunk / chunks_per_bag;
                int64_t begin = offsets[bag];
                int64_t end = (bag < num_bags - 1) ? (offsets[bag + 1]) : N;
                float weight_feat_sum = 0;
                for (int64_t rowidx = begin; rowidx < end; rowidx++) {
                    weight_feat_sum += tr_last[rowidx * D + colidx];
                }
                output[bag * D + colidx] = weight_feat_sum;
            }
        }
    }

__global__ void reduce_output_kernel(
    int32_t N,
    int32_t D,
    int32_t num_bags,
    const int64_t* __restrict__ offsets,
    const float* __restrict__ tr_last,
    float* __restrict__ output) {
        int64_t chunks_per_bag = div_round_up(D, (int64_t)blockDim.x);
        int64_t total_chunks = num_bags * chunks_per_bag;
        int64_t chunk_offset = blockIdx.x * blockDim.y + threadIdx.y;
        int64_t chunk_stride = gridDim.x * blockDim.y;

        for (int64_t chunk = chunk_offset; chunk < total_chunks; chunk += chunk_stride) {
            int64_t colidx = (chunk % chunks_per_bag) * blockDim.x + threadIdx.x;
            if (colidx < D) {
                int64_t bag = chunk / chunks_per_bag;
                int64_t begin = offsets[bag];
                int64_t end = (bag < num_bags - 1) ? (offsets[bag + 1]) : N;
                float weight_feat_sum = 0;
                for (int64_t rowidx = begin; rowidx < end; rowidx++) {
                    weight_feat_sum += tr_last[rowidx * D + colidx];
                }
                output[bag * D + colidx] = weight_feat_sum;
            }
        }
    }

Tensor grt_embeddings_forward_cuda(
    int32_t num_group_bags,
    int32_t num_bags,
    int32_t emb_dims,
    const std::vector<int>& row_shapes,
    const std::vector<int>& emb_shapes,
    const std::vector<int>& tt_ranks,
    Tensor tt_idx_shapes,
    int32_t intra_group_nnz,
    Tensor intra_group_indices,
    Tensor intra_group_offsets,
    Tensor inter_group_indices,
    Tensor inter_group_offsets,
    const std::vector<Tensor>& tt_cores,
    Tensor output
) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(intra_group_indices.get_device());
    device_guard.set_index(intra_group_offsets.get_device());
    device_guard.set_index(inter_group_indices.get_device());
    device_guard.set_index(inter_group_offsets.get_device());

    int32_t num_cores = row_shapes.size();
    int32_t inter_group_nnz = num_group_bags;

    if (intra_group_nnz == 0) {
        return output;
    }

    TORCH_CHECK(emb_dims > 0);
    TORCH_CHECK(emb_dims % 4 == 0);
    TORCH_CHECK(num_cores > 0);

    /* Intra Group Reduce Operation */

    int32_t last_tt_shape = emb_shapes[num_cores - 1] * tt_ranks[num_cores - 1];
    auto intra_group_output = at::zeros({num_group_bags, last_tt_shape}, tt_cores[0].options().dtype(at::kFloat));

    auto intra_group_ptr_tensor = at::empty({intra_group_nnz}, tt_cores[0].options().dtype(at::kLong));

    float** intra_group_ptr = (float**)intra_group_ptr_tensor.data_ptr<int64_t>();

    init_intra_group_reduce_forward_kernel<<<div_round_up(intra_group_nnz, 512), 512, 0, c10::cuda::getCurrentCUDAStream()>>>(
        intra_group_nnz,
        intra_group_indices.data_ptr<int64_t>(),
        tt_cores[num_cores - 1].packed_accessor64<float, 2, RestrictPtrTraits>(),
        intra_group_ptr);

    int32_t tx = kWarpSize;
    int32_t ty = 1024 / tx;
    dim3 threads(tx, ty);
    
    intra_group_reduce_kernel<<<div_round_up(intra_group_nnz, ty), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        intra_group_nnz,
        last_tt_shape,
        num_group_bags,
        intra_group_offsets.data_ptr<int64_t>(),
        intra_group_ptr,
        intra_group_output.data_ptr<float>());

    /* Batched GEMM Operation */

    std::vector<int32_t> m(num_cores - 1);
    std::vector<int32_t> n(num_cores - 1);
    std::vector<int32_t> k(num_cores - 1);
    float alpha = 1.0;
    float beta = 0.0;

    int32_t m_ = emb_shapes[0];
    for (int32_t i = 0; i < num_cores - 1; ++i) {
        m[i] = m_;
        k[i] = tt_ranks[i + 1];
        n[i] = emb_shapes[i + 1] * tt_ranks[i + 2];
        m_ = m_ * emb_shapes[i + 1];
    }

    std::vector<Tensor> tr;
    int32_t tr_size = emb_shapes[0] * tt_ranks[1];
    for (int32_t i = 0; i < num_cores - 1; ++i) {
        tr_size = tr_size * emb_shapes[i + 1] * tt_ranks[i + 2] / tt_ranks[i + 1];
        tr.push_back(at::empty({inter_group_nnz, tr_size}, tt_cores[0].options().dtype(at::kFloat)));
    }
    auto a_ptr_tensor = at::empty({(num_cores - 1) * inter_group_nnz}, tt_cores[0].options().dtype(at::kLong));
    auto b_ptr_tensor = at::empty({(num_cores - 1) * inter_group_nnz}, tt_cores[0].options().dtype(at::kLong));
    auto c_ptr_tensor = at::empty({(num_cores - 1) * inter_group_nnz}, tt_cores[0].options().dtype(at::kLong));

    float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
    float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
    float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();
    
    init_grouped_batch_gemm_forward_cuda(
        num_cores,
        inter_group_nnz,
        tt_idx_shapes.data_ptr<int64_t>(),
        inter_group_indices.data_ptr<int64_t>(),
        tt_cores,
        tr,
        intra_group_output.packed_accessor64<float, 2, RestrictPtrTraits>(),
        a_ptr,
        b_ptr,
        c_ptr);

    for (int32_t i = 0; i < num_cores - 1; ++i) {
        cuda_gemm_batched_fp32_fp32(
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m[i],
            n[i],
            k[i],
            &alpha,
            (void**)&(a_ptr[i * inter_group_nnz]),
            m[i],
            (void**)&(b_ptr[i * inter_group_nnz]),
            k[i],
            &beta,
            (void**)&(c_ptr[i * inter_group_nnz]),
            m[i],
            inter_group_nnz);
    }

    /* Inter Group Reduce Operation */

    inter_group_reduce_kernel<<<div_round_up(inter_group_nnz, ty), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        inter_group_nnz,
        emb_dims,
        num_bags,
        inter_group_offsets.data_ptr<int64_t>(),
        tr[num_cores - 2].data_ptr<float>(),
        output.data_ptr<float>());

    return output;
}

Tensor tt_embeddings_forward_cuda(
    int32_t num_bags,
    int32_t emb_dims,
    const std::vector<int>& row_shapes,
    const std::vector<int>& emb_shapes,
    const std::vector<int>& tt_ranks,
    Tensor tt_idx_shapes,
    int32_t nnz,
    Tensor indices,
    Tensor offsets,
    const std::vector<Tensor>& tt_cores,
    Tensor output
) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(indices.get_device());
    device_guard.set_index(offsets.get_device());

    int32_t num_cores = row_shapes.size();

    if (nnz == 0) {
        return output;
    }

    TORCH_CHECK(emb_dims > 0);
    TORCH_CHECK(emb_dims % 4 == 0);
    TORCH_CHECK(num_cores > 0);

    /* Batched GEMM Operation */

    std::vector<int32_t> m(num_cores - 1);
    std::vector<int32_t> n(num_cores - 1);
    std::vector<int32_t> k(num_cores - 1);
    float alpha = 1.0;
    float beta = 0.0;

    int32_t m_ = emb_shapes[0];
    for (int32_t i = 0; i < num_cores - 1; ++i) {
        m[i] = m_;
        k[i] = tt_ranks[i + 1];
        n[i] = emb_shapes[i + 1] * tt_ranks[i + 2];
        m_ = m_ * emb_shapes[i + 1];
    }

    std::vector<Tensor> tr;
    int32_t tr_size = emb_shapes[0] * tt_ranks[1];
    for (int32_t i = 0; i < num_cores - 1; ++i) {
        tr_size = tr_size * emb_shapes[i + 1] * tt_ranks[i + 2] / tt_ranks[i + 1];
        tr.push_back(at::empty({nnz, tr_size}, tt_cores[0].options().dtype(at::kFloat)));
    }
    auto a_ptr_tensor = at::empty({(num_cores - 1) * nnz}, tt_cores[0].options().dtype(at::kLong));
    auto b_ptr_tensor = at::empty({(num_cores - 1) * nnz}, tt_cores[0].options().dtype(at::kLong));
    auto c_ptr_tensor = at::empty({(num_cores - 1) * nnz}, tt_cores[0].options().dtype(at::kLong));

    float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
    float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
    float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();
    
    init_batch_gemm_forward_cuda(
        num_cores,
        nnz,
        tt_idx_shapes.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),
        tt_cores,
        tr,
        a_ptr,
        b_ptr,
        c_ptr);

    for (int32_t i = 0; i < num_cores - 1; ++i) {
        cuda_gemm_batched_fp32_fp32(
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m[i],
            n[i],
            k[i],
            &alpha,
            (void**)&(a_ptr[i * nnz]),
            m[i],
            (void**)&(b_ptr[i * nnz]),
            k[i],
            &beta,
            (void**)&(c_ptr[i * nnz]),
            m[i],
            nnz);
        }

    int32_t tx = kWarpSize;
    int32_t ty = 1024 / tx;
    dim3 threads(tx, ty);
    
    /* Reduce Operation */
    reduce_output_kernel<<<div_round_up(nnz, ty), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        nnz,
        emb_dims,
        num_bags,
        offsets.data_ptr<int64_t>(),
        tr[num_cores - 2].data_ptr<float>(),
        output.data_ptr<float>());

    return output;
}