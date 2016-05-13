//#include "MonaCu.cuh"

#include <cublas.h>
#include <cublas_api.h>
#include <cublas_v2.h>
#include <nvfunctional>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include "../../include/MonaCuTypeDef.h"

static int threadsPerBlock = 256;

template<typename Real>
__global__ void vecCopy(int numElements, const Real *src, Real *dst){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) dst[i] = src[i];
}

template<typename Real>
__global__ void vecAdd(int numElements, Real *A, Real *B, Real *C){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] + B[i];
}

template<typename Real>
__global__ void vecMinus(int numElements, Real *A, Real *B, Real *C){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] - B[i];
}

template<typename Real>
__global__ void vecMul(int numElements, Real *A, Real *B, Real *C){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] * B[i];
}

template<typename Real>
__global__ void vecDiv(int numElements, Real *A, Real *B, Real *C){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] / B[i];
}


int getCudaBlock(){                                                                              
    return threadsPerBlock;
}
                                                                                             
int setCudaBlock(int num){                                                                       
    threadsPerBlock = num;                                                                   
    return threadsPerBlock;                                                                  
}                                                                                            

void* vectorAlloc(int numElements, int typeSize){
    void *tmp;
    cudaError_t res = cudaMalloc((void**)&tmp, typeSize*numElements);
    if (res != cudaSuccess) return NULL;
    return tmp;
}

MonaCu::error_t vectorFree(void *dst){
    cudaError_t res = cudaFree(dst);
    if (res != cudaSuccess) return MonaCu::Fail;
    return MonaCu::Success;
}

#define FUNCTION_INSTANTIATION(TYPE)                                                            \
void vectorCopy(int numElements, const TYPE *src, TYPE *dst){                                   \
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;                  \
    vecCopy<TYPE> << <blocksPerGrid, threadsPerBlock >> >(numElements, src, dst);               \
}                                                                                               \
void vectorAdd(int numElements, TYPE *A, TYPE *B, TYPE *dst){                                   \
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;                  \
    vecAdd<TYPE> << <blocksPerGrid, threadsPerBlock >> >(numElements, A, B, dst);               \
}                                                                                               \
void vectorMinus(int numElements, TYPE *A, TYPE *B, TYPE *dst){                                 \
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;                  \
    vecMinus<TYPE> << <blocksPerGrid, threadsPerBlock >> >(numElements, A, B, dst);             \
}                                                                                               \
void vectorMul(int numElements, TYPE *A, TYPE *B, TYPE *dst){                                   \
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;                  \
    vecMul<TYPE> << <blocksPerGrid, threadsPerBlock >> >(numElements, A, B, dst);               \
}                                                                                               \
void vectorDiv(int numElements, TYPE *A, TYPE *B, TYPE *dst){                                   \
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;                  \
    vecDiv<TYPE> << <blocksPerGrid, threadsPerBlock >> >(numElements, A, B, dst);               \
}                                                                                               \
MonaCu::error_t getData(int numElements, const TYPE *src, TYPE *dst){                           \
    cudaError_t res = cudaMemcpy(dst, src, sizeof(TYPE)*numElements, cudaMemcpyDeviceToHost);   \
    if (res != cudaSuccess) return MonaCu::Fail;                                                \
    return MonaCu::Success;                                                                     \
}                                                                                               \
MonaCu::error_t setData(int numElements, const TYPE *src, TYPE *dst){                           \
    cudaError_t res = cudaMemcpy(dst, src, sizeof(TYPE)*numElements, cudaMemcpyHostToDevice);   \
    if (res != cudaSuccess) return MonaCu::Fail;                                                \
    return MonaCu::Success;                                                                     \
}

FUNCTION_INSTANTIATION(short);
FUNCTION_INSTANTIATION(int);
FUNCTION_INSTANTIATION(float);
FUNCTION_INSTANTIATION(double);