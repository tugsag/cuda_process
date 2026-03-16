#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <common.h>


namespace py = pybind11;

Stats reduce(float* data, int width, int height);
void normalize(float* data, int width, int height, float mean, float std);

py::dict reduceWrapper(py::array_t<float> input_image){
    py::buffer_info buf = input_image.request();
    float* ptr = static_cast<float*>(buf.ptr);

    int height = buf.shape[0];
    int width = buf.shape[1];

    Stats out = reduce(ptr, width, height);

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

PYBIND11_MODULE(cuda_process, m){
    m.def("calculate_stats", &reduceWrapper, "function to find mean and std using CUDA");

    m.def("normalize", &normalizeWrapper, "function to normalize with mean/std using CUDA");

}