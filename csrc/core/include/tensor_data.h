#pragma once
#include <vector>
#include <memory>


namespace pyimagematch{

void* fastMalloc(std::size_t size);

class TensorData{
public:
    enum DataDeviceStatus {AT_CPU, AT_GPU, AT_BOTH};
    enum DataDeviceType {DATA_CPU, DATA_GPU};
public:
    TensorData(std::size_t data_size, DataDeviceType data_type=DATA_CPU);
    TensorData(std::size_t data_size, void* user_data, DataDeviceType data_type=DATA_CPU);
    ~TensorData();
    std::size_t getDataSize(){ return _data_size; }
    void* getDataPtr(){  return _data; }
    TensorData copy();

public:
    std::size_t _data_size;
    void* _data = nullptr;
    bool _own_data;
    DataDeviceStatus _data_device_status = AT_CPU;

#ifdef PIM_WITH_CUDA
public:
    void to_cpu();
    void to_cuda();
    void* getGpuDataPtr(){  return _gpu_data; }
private:
    void* _gpu_data = nullptr;
#endif

};

}
