# CUDA-Process

This is a simple demo of a bridge that uses parallel reduction on GPU via CUDA and C++ to calculate statistics from an image and normalizes them using those stats.

## What this repo currently does
1. Binds Python and C++/CUDA using PyBind11
2. Uses [sequential addressing method](https://medium.com/@rimikadhara/7-step-optimization-of-parallel-reduction-with-cuda-33a3b2feafd8) for parallel reduction using GPU. This minimizes warp divergence and ensures coalesced memory accesses, reducing cache misses and conflicts
    * This reduces O(n) operation to O(log n) time
    * Uses float2 to perform a single pass reduction for sum and sum-of-squares, halving global memory bandwith requirements
3. Relies on PyTorch to build CUDA extension 
4. Can be pip installed and used as a python module
5. NEW: support for massive inputs using async pipeline and CUDA streams leveraging PyTorch's stream API, and uses persistent memory allocation to cut down on transfer latency

## Where to go from the current state (potential improvements)
- [ ] Support multichannel images: use float3 struct to store running sums per channel
    * Adjust addressing method to be interleaved to avoid bank conflicts (avoid slowdowns)
- [x] Support massive inputs: implement an asynchronous pipeline using CUDA Streams to overlap PCIe data transfers with kernel execution to hide transfer latency for out-of-core datasets
    * Use persistent memory allocation for this to cut down on transfer latency
- [ ] Use other allocation methods (cudaHostAlloc) to speed up transfer latency


## Comment on performance
- Using **pure numpy** to calculate stats and normalize an image (size: (5000, 5000, 1)), total runtime was **0.09** seconds
- Using **GPU parallelization** on the same image, total runtime was **0.15** seconds

Because the image isn't very large and there was only 1, we absolutely expect the GPU to be slower. This is because of transfer latency and cold startups. Using CUDA for parallelization excels when there is a lot of data to be processed (very large images/lot of images in sequence). We also include the time for cold starts. See below:

- On a sequence of 10 images sized (10000, 10000, 1), using **pure numpy**, we achieve a total speed of **4.32** seconds
- On the same set of images, using **naive GPU parallelization**, we achieve a total speed of **1.20** seconds to calculate stats and normalize the image
- On the same set of images, using **CUDA streams and persistent memory allocation**, we see a total speed of **1.00** seconds

On very large datasets (e.g. 50TB of imagery), assuming similar gains in speed using GPU, we could expect a total time saved of about **5-6 hours** at the minimum to perform the same tasks as described above.
