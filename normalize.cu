#include <cstddef>
#include <stdio.h>



__global__
void normalizeKernel(float4* data, int num_vectors, float mean, float std){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_vectors){
        float4 pixel_vec = data[idx];

        pixel_vec.x = (pixel_vec.x - mean) / std;
        pixel_vec.y = (pixel_vec.y - mean) / std;
        pixel_vec.z = (pixel_vec.z - mean) / std;
        pixel_vec.w = (pixel_vec.w - mean) / std;

        data[idx] = pixel_vec;
    }
}


void normalize(float* image_data, int width, int height, float mean, float std){
    int total_floats = width * height;
    size_t size = total_floats * sizeof(float);
    float* d_data;

    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, image_data, size, cudaMemcpyHostToDevice);

    int num_vectors = total_floats / 4;
    int blockSize = 256;
    int blocksPerGrid = (num_vectors + blockSize - 1) / blockSize;

    normalizeKernel<<<blocksPerGrid, blockSize>>>((float4*)d_data, num_vectors, mean, std);

    cudaMemcpy(image_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}