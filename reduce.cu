#include <climits>
#include <cmath>
#include <stdio.h>
#include <common.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>


float *d_data[2] = {nullptr, nullptr};
float2 *d_out[2] = {nullptr, nullptr};

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

void init_buffers(int width, int height, int num_streams){
    int total_floats = width * height;
    const int blockSize = 256;
    const int blocksPerGrid = (total_floats + blockSize - 1) / blockSize;

    size_t data_size = total_floats * sizeof(float);
    size_t arr_size = blocksPerGrid * sizeof(float2);

    for (int i = 0; i < num_streams; i++){
        cudaMalloc(&d_data[i], data_size);
        cudaMalloc(&d_out[i], arr_size);
    }
}

Stats reduceStream(float* image_data, int width, int height, int stream_idx){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    uintptr_t stream_ptr = reinterpret_cast<uintptr_t>(stream);

    int total_floats = width * height;
    const int blockSize = 256;
    const int blocksPerGrid = (total_floats + blockSize - 1) / blockSize;

    size_t data_size = total_floats * sizeof(float);
    size_t arr_size = blocksPerGrid * sizeof(float2);

    std::vector<float2> sum_arr(blocksPerGrid);

    cudaMemcpyAsync(d_data[stream_idx], image_data, data_size, cudaMemcpyHostToDevice, stream);

    int shared_mem_size = blockSize * sizeof(float2);
    reduceKernel<<<blocksPerGrid, blockSize, shared_mem_size, stream>>>(d_data[stream_idx], d_out[stream_idx], total_floats);
    cudaMemcpyAsync(sum_arr.data(), d_out[stream_idx], arr_size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    Stats stats;

    double running_sum = 0.0;
    double running_sq_sum = 0.0;

    for (int i = 0; i < blocksPerGrid; i++){
        running_sum += (double)sum_arr[i].x;
        running_sq_sum += (double)sum_arr[i].y;
    }

    double final_mean = running_sum / total_floats;
    double final_var = (running_sq_sum / total_floats) - (final_mean * final_mean);

    stats.mean = final_mean;
    stats.var = final_var;
    stats.std = std::sqrt(std::max(0.0, final_var));

    return stats;
}

Stats reduceSimple(float* image_data, int width, int height){
    int total_floats = width * height;
    const int blockSize = 256;
    const int blocksPerGrid = (total_floats + blockSize - 1) / blockSize;

    float *d_data;
    float2 *d_out;
    size_t data_size = total_floats * sizeof(float);
    size_t arr_size = blocksPerGrid * sizeof(float2);

    std::vector<float2> sum_arr(blocksPerGrid);

    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_out, arr_size);

    cudaMemcpy(d_data, image_data, data_size, cudaMemcpyHostToDevice);

    int shared_mem_size = blockSize * sizeof(float2);
    reduceKernel<<<blocksPerGrid, blockSize, shared_mem_size>>>(d_data, d_out, total_floats);
    cudaMemcpy(sum_arr.data(), d_out, arr_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_out);

    Stats stats;

    double running_sum = 0.0;
    double running_sq_sum = 0.0;

    for (int i = 0; i < blocksPerGrid; i++){
        running_sum += (double)sum_arr[i].x;
        running_sq_sum += (double)sum_arr[i].y;
    }

    double final_mean = running_sum / total_floats;
    double final_var = (running_sq_sum / total_floats) - (final_mean * final_mean);

    stats.mean = final_mean;
    stats.var = final_var;
    stats.std = std::sqrt(std::max(0.0, final_var));

    return stats;
}