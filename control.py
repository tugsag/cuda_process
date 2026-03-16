import numpy as np
import torch
import time
import cuda_process


"""Very simple and imperfect controller to demonstrate how to call CUDA functions from python
Also holds as a skeleton for more advanced profiling/testing implementations in the future
Just uses time.time for now, which isn't ideal, but should be good enough for a quick and dirty look
"""

def use_np(img):
    print('---------numpy--------')
    start = time.time()
    mean = np.mean(img)
    std = np.std(img)
    var = np.var(img)
    print(mean, std, var)
    k = (img - mean) / std
    # print(k[:2, :2, :])
    print(time.time() - start)

def use_cuda_loop(im):
    print('---------cuda loop--------')
    start = time.time()
    means, stds, vars_ = [], [], []
    for subim in im:
        results = cuda_process.calculate_stats(subim)
        means.append(results['mean'])
        stds.append(results['std'])
        vars_.append(results['var'])

    mean, std = np.mean(means), np.mean(stds)
    for subim in im:
        cuda_process.normalize(subim, mean, std)

    print(mean, std, np.mean(vars_))
    print(time.time() - start)

def use_cuda_streaming(im):
    print('---------streaming--------')

    start = time.time()

    cuda_process.init_buffers(im.shape[1], im.shape[2], 2)

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    means, stds, vars_ = [], [], []
    for i in range(0, len(im), 2):
        with torch.cuda.stream(stream1):
            results1 = cuda_process.calculate_stats_stream(im[i], 0)
            means.append(results1['mean'])
            stds.append(results1['std'])
            vars_.append(results1['var'])
        if i + 1 < len(im):
            with torch.cuda.stream(stream2):
                results2 = cuda_process.calculate_stats_stream(im[i+1], 1)
                means.append(results2['mean'])
                stds.append(results2['std'])
                vars_.append(results2['var'])

    torch.cuda.synchronize()

    mean, std = np.mean(means), np.mean(stds)
    for i in range(0, len(im), 2):
        with torch.cuda.stream(stream1):
            cuda_process.normalize(im[i], mean, std)
        if i + 1 < len(im):
            with torch.cuda.stream(stream2):
                cuda_process.normalize(im[i+1], mean, std)

    torch.cuda.synchronize()
    print(mean, std, np.mean(vars_))
    print(time.time() - start)

    
if __name__ == '__main__':
    size = (10, 10000, 10000, 1)
    im = np.random.rand(*size).astype(np.float32)
    # im = np.ones(size).astype(np.float32)

    use_np(im)
    use_cuda_loop(im)
    use_cuda_streaming(im)

