#pragma once
#include <vector>
#include <memory>
#include "core/include/tensor_data.h"

namespace pyimagematch{

template <std::size_t Dim = 0, typename Strides> std::size_t byte_offset_unsafe(const Strides &) { return 0; }
template <std::size_t Dim = 0, typename Strides, typename... Ix>
std::size_t byte_offset_unsafe(const Strides &strides, std::size_t i, Ix... index) {
    return i * strides[Dim] + byte_offset_unsafe<Dim + 1>(strides, index...);
}

template<class T>
int calculateDatasizeWithStrides(const std::vector<int>& shapes, std::vector<int>& strides){
    std::size_t total = sizeof(T);
    for( int i = shapes.size()-1; i >= 0; i-- )
    {
        if (strides[i] > 0){
            total = strides[i];
        } else {
            strides[i] = total;
        }
        total *= shapes[i];
    }
    return total;
}

inline bool isShapeEqual(const std::vector<int>& shapes1, const std::vector<int>& shapes2){
    if(shapes1.size() != shapes2.size()){
        return false;
    }

    for(int i=0; i<shapes1.size(); i++){
        if(shapes1[i] != shapes2[1]){
            return false;
        }
    }

    return true;
}


template<class T>
class TensorBase{
public:
    TensorBase(){}
    TensorBase(const std::vector<int>& shapes){
        create(shapes);
    }

    virtual create(const std::vector<int>& shapes) = 0;

    void createContiguous(const std::vector<int>& shapes){
        if(!isContiguous()){
            release();
        }
        create(shapes);
    }

    bool isContiguous(){
        int num_elem = 1;
        for(int i=_strides.size()-1; i>=0; i--){
            if(i == _strides.size()-1){
                if(sizeof(T) != _strides[i]){
                    return false;
                }
            }else{
                if(_strides[i] != num_elem * _shapes[i+1] * sizeof(T)){
                    return false;
                }
                num_elem *= _shapes[i+1];
            }
        }
        return true;
    }

    T* mutableData(){
        return _data_ptr;
    }

    void release(){
        _data_ptr = nullptr;
        _tensor_data_ptr = nullptr;
        _strides.clear();
        _shapes.clear();
        _ori_data_offset = 0;
    }

    std::vector<int> stride(){ return _strides; }
    std::vector<int> shape(){ return _shapes; }

protected:
    std::vector<int> _strides;
    std::vector<int> _shapes;
    std::shared_ptr<TensorData> _tensor_data_ptr = nullptr;
    T* _data_ptr;
    std::size_t _ori_data_offset = 0;

};


template<class T>
class Tensor: public TensorBase<T>{
public:
    Tensor(){}

    Tensor(const std::vector<int>& shapes, const std::vector<int>& strides){
        create(shapes, strides);
    }
    
    template <typename... Ix> T &operator()(Ix... index){
        return *reinterpret_cast<T*>((unsigned char*)_data_ptr + byte_offset_unsafe(_strides, std::size_t(index)...));
    }

    virtual void create(const std::vector<int>& shapes) override{
        if(isShapeEqual(shapes, _shapes)){
            return;
        }
        _shapes = shapes;
        _strides = std::vector<int>(_shapes.size(), -1);
        int data_size = calculateDatasizeWithStrides<T>(shapes, _strides);
        // TODO: handle cases data_size < 0
        _tensor_data_ptr = std::make_shared<TensorData>((std::size_t) data_size);
        _data_ptr = (T*) _tensor_data_ptr->getDataPtr();
    }

    void create(const std::vector<int>& shapes, const std::vector<int>& strides){
        if(isShapeEqual(shapes, _shapes)){
            return;
        }
        _shapes = shapes;
        _strides = strides;
        // TODO: handle shapes strides different dims
        int data_size = calculateDatasizeWithStrides<T>(shapes, _strides);
        // TODO: handle cases data_size < 0
        _tensor_data_ptr = std::make_shared<TensorData>((std::size_t) data_size);
        _data_ptr = (T*) _tensor_data_ptr->getDataPtr();
    }

    Tensor<T> ascontiguous(){
        if(isContiguous()){

        }else{
            
        }
    }
};

}