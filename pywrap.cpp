#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <common.h>


namespace py = pybind11;

void init_buffers(int width, int height, int num_streams);
Stats reduceStream(float* data, int width, int height, int stream_idx);
Stats reduceSimple(float* data, int width, int height);
void normalize(float* data, int width, int height, float mean, float std);

py::dict reduceStreamWrapper(py::array_t<float> input_image, int stream_idx){
    py::buffer_info buf = input_image.request();
    float* ptr = static_cast<float*>(buf.ptr);

    int height = buf.shape[0];
    int width = buf.shape[1];

    Stats out = reduceStream(ptr, width, height, stream_idx);

    py::dict result;
    result["mean"] = out.mean;
    result["std"] = out.std;
    result["var"] = out.var;

    return result;
} 

py::dict reduceSimpleWrapper(py::array_t<float> input_image){
    py::buffer_info buf = input_image.request();
    float* ptr = static_cast<float*>(buf.ptr);

    int height = buf.shape[0];
    int width = buf.shape[1];

    Stats out = reduceSimple(ptr, width, height);

    py::dict result;
    result["mean"] = out.mean;
    result["std"] = out.std;
    result["var"] = out.var;

    return result;
} 

void normalizeWrapper(py::array_t<float> input_image, float mean, float std){
    py::buffer_info buf = input_image.request();
    float* ptr = static_cast<float*>(buf.ptr);

    int height = buf.shape[0];
    int width = buf.shape[1];

    normalize(ptr, width, height, mean, std);
}

void initBuffersWrapper(int width, int height, int num_streams){
    init_buffers(width, height, num_streams);
}

PYBIND11_MODULE(cuda_process, m){
    m.def("init_buffers", &initBuffersWrapper, "function to initialize buffers for streams");
    m.def("calculate_stats_stream", &reduceStreamWrapper, "function to find mean and std using CUDA streams");
    m.def("calculate_stats", &reduceSimpleWrapper, "function to find mean and std using CUDA");
    m.def("normalize", &normalizeWrapper, "function to normalize with mean/std using CUDA");

}