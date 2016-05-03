#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "operation.cuh"
#include <stdio.h>
#include "MonaCu.h"


int threadsPerBlock = 256;

template<class Real>
__global__ void vecCopy(int numElements, const Real *src, Real *dst){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) dst[i] = src[i];
}

template<class Real>
__global__ void vecAdd(int numElements, Real *A, Real *B, Real *C){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] + B[i]; 
}

template<class Real>
__global__ void vecMinus(int numElements, Real *A, Real *B, Real *C){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] - B[i];
}

template<class Real>
__global__ void vecMul(int numElements, Real *A, Real *B, Real *C){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] * B[i];
}

template<class Real>
__global__ void vecDiv(int numElements, Real *A, Real *B, Real *C){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] / B[i];
}

template<class Real>
void vectorCopy(int numElements, const Real *src, Real *dst){
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vecCopy<Real> << <blocksPerGrid, threadsPerBlock >> >(numElements, src, dst);
}

template<class Real>
void vectorAdd(int numElements, Real *A, Real *B, Real *dst){   
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<Real><< <blocksPerGrid, threadsPerBlock >> >(numElements, A, B, dst);
}

template<class Real>
void vectorMinus(int numElements, Real *A, Real *B, Real *dst){
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vecMinus<Real> << <blocksPerGrid, threadsPerBlock >> >(numElements, A, B, dst);
}

template<class Real>
void vectorMul(int numElements, Real *A, Real *B, Real *dst){   
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vecMul<Real> << <blocksPerGrid, threadsPerBlock >> >(numElements, A, B, dst);
}

template<class Real>
void vectorDiv(int numElements, Real *A, Real *B, Real *dst){
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vecDiv<Real> << <blocksPerGrid, threadsPerBlock >> >(numElements, A, B, dst);
}

int main()
{
    const int arraySize = 5;
    const float a[arraySize] = { 1, 2, 3, 4, 5 };
    const float b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    MonaCu::cuVector<float> lhs(a, arraySize);
    MonaCu::cuVector<float> rhs(b, arraySize);
    MonaCu::cuVector<float> res = lhs;// +rhs;
    cout << "copy lhs" << endl;
    res.print();
    
    cout << "copy rhs" << endl;
    res = rhs;
    res.print();

    cout << "lhs + rhs" << endl;
    res = lhs + rhs;
    res.print();

    cout << "lhs - rhs" << endl;
    res = rhs - lhs;    
    res.print();

    cout << "rhs + lhs - lhs" << endl;
    res = rhs + lhs - lhs;    
    res.print();

    cout << "lhs * rhs" << endl;
    res = lhs * rhs;
    res.print();

    cout << "lhs + lhs * rhs" << endl;
    res = lhs + lhs * rhs;
    res.print();

    cout << "(lhs + lhs) * rhs" << endl;
    res = (lhs + lhs) * rhs;
    res.print();

    cout << "lhs + lhs * rhs - lhs" << endl;
    res = lhs + lhs * rhs - lhs;
    res.print();

    cout << "lhs + lhs * rhs - lhs * rhs" << endl;
    res = lhs + lhs * rhs - lhs * rhs;
    res.print();

    cout << "(lhs - lhs) * rhs :: zero multipulication" << endl;
    res = (lhs - lhs) * rhs;
    res.print();

    cout << "lhs / lhs" << endl;
    res = (lhs / lhs);
    res.print();

    cout << "(lhs - lhs) / rhs :: zero devide" << endl;
    res = (lhs - lhs) / rhs;
    res.print();

    cout << "rhs / (lhs - lhs) :: zero devide" << endl;
    res = (lhs - lhs) / rhs;
    res.print();

    return 0;
}
