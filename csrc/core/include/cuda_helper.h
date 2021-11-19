#pragma once
#include <cuda.h>
namespace pyimagematch{

__host__ __device__ __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

}