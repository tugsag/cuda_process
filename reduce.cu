#include <cmath>
#include <stdio.h>
#include <common.h>
#include <vector>




__global__ 
void reduceKernel(float* g_in_data, float2* g_out_data, int lim){
    extern __shared__ float2 sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lim){
        float val = g_in_data[i];
        sdata[tid].x = val;
        sdata[tid].y = val * val;
    } else{
        sdata[tid].x = 0.0f;
        sdata[tid].y = 0.0f;
    }
    __syncthreads();

    // Reduction method -- occurs in shared memory
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
        if (tid < s){
            sdata[tid].x += sdata[tid + s].x;
            sdata[tid].y += sdata[tid + s].y;
        }
        __syncthreads();
    }
    if (tid == 0){
        g_out_data[blockIdx.x].x = sdata[0].x;
        g_out_data[blockIdx.x].y = sdata[0].y;
    }
}

Stats reduce(float* image_data, int width, int height){
    int total_floats = width * height;
    int num_vectors = total_floats;
    const int blockSize = 256;
    const int blocksPerGrid = (num_vectors + blockSize - 1) / blockSize;

    float *d_data;
    float2 *d_out;
    size_t data_size = total_floats * sizeof(float);
    size_t arr_size = blocksPerGrid * sizeof(float) * 2;

    std::vector<float2> sum_arr(blocksPerGrid);

    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_out, arr_size);

    cudaMemcpy(d_data, image_data, data_size, cudaMemcpyHostToDevice);

    int shared_mem_size = blockSize * sizeof(float) * 2;
    reduceKernel<<<blocksPerGrid, blockSize, shared_mem_size>>>(d_data, d_out, total_floats);
    cudaMemcpy(sum_arr.data(), d_out, arr_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_out);

    Stats stats;

    for (int i = 0; i < blocksPerGrid; i++){
        stats.mean += sum_arr[i].x;
        stats.std += sum_arr[i].y;
    }
    stats.mean /= num_vectors;
    stats.var = (stats.std / num_vectors) - (stats.mean * stats.mean);
    stats.std = std::sqrt(stats.var);

    return stats;
}