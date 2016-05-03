#ifndef __MONACU_H__
#define __MONACU_H__

#include <vector>
#include <iostream>
#include <sstream>
#include <exception>
#include <utility>
#include "operation.cuh"
#include <cuda_runtime.h>
#include <cublas.h>

using namespace std;
namespace MonaCu
{
    class Exception : public std::exception
    {
    public:
        explicit Exception(const char *msg, const char *func, const int line)
            :_msg(msg), _func(func), _line(line){}
        virtual ~Exception() throw() {}

        virtual const char* what() throw()
        {
            stringstream out;
            out << _msg << "[" << _func << "]" << "[" << _line << "]";
            return out.str().c_str();
        }
    private:
        string _msg;
        string _func;
        int _line;
    };
#define WHAT(str) Exception(str, __FUNCTION__, __LINE__)


    template <typename Real>
    class __declspec(dllexport) cuVector
    {
    public:

        cuVector() :_mat(0), _size(0){};
        virtual ~cuVector(){ cudaFree(_mat); _mat = NULL; }

        cuVector(const Real* hostArray, size_t size){
            cudaError_t res = this->alloc(hostArray, size);
            if (res != cudaSuccess) throw WHAT("cuda alloc error"); // cudaSuccess == 0
        }

        cuVector(vector<Real> &hostVector){
            cudaError_t res = this->alloc(hostVector.data(), hostVector.size());
            if (res != cudaSuccess) throw WHAT("cuda alloc error"); //cudaSuccess == 0
        }

        cuVector(cuVector &otherMat){ copy(otherMat); }
        cuVector(cuVector &&otherMat){ // just swap
            Real *tmp = otherMat._mat;
            otherMat._mat = _mat;
            _mat = tmp;
            _size = otherMat.size();
        }
        void operator=(cuVector &otherMat){ copy(otherMat); }
        void operator=(cuVector &&otherMat){ // just swap
            Real *tmp = otherMat._mat;
            otherMat._mat = _mat;
            _mat = tmp;
            _size = otherMat.size();
        }

        cuVector operator+(cuVector &rhs){
            if (_size != rhs.size()) throw WHAT("Vector size not equal");
            cuVector tmp = *this;
            vectorAdd(_size, _mat, rhs._mat, tmp._mat);
            return std::move(tmp);
        }

        cuVector operator-(cuVector &rhs){
            if (_size != rhs.size()) throw WHAT("Vector size not equal");
            cuVector tmp = *this;
            vectorMinus(_size, _mat, rhs._mat, tmp._mat);
            return std::move(tmp);
        }

        cuVector operator*(cuVector &rhs){
            if (_size != rhs.size()) throw WHAT("Vector size not equal");
            cuVector tmp = *this;
            vectorMul(_size, _mat, rhs._mat, tmp._mat);
            return std::move(tmp);
        }

        cuVector operator/(cuVector &rhs){
            if (_size != rhs.size()) throw WHAT("Vector size not equal");
            cuVector tmp = *this;
            vectorDiv(_size, _mat, rhs._mat, tmp._mat);
            return std::move(tmp);
        }

        Real* begin(){ return _mat; }
        Real* end(){ return _mat + _size; }
        size_t size(){ return _size; }

        void print(void){
            stringstream out;
            vector<Real> tmp(_size);
            cudaError_t res = cudaMemcpy(tmp.data(), _mat, sizeof(Real)*_size, cudaMemcpyDeviceToHost);
            for (int i = 0; i < tmp.size(); i++)
            {
                out << tmp[i];
                if (i + 1 % 10) out << ",\t";
                else out << "\n";
            }
            cout << out.str() << endl;
        }

    private:
        Real* _mat;
        size_t _size;

        cudaError_t alloc(const Real *mat, size_t size) throw()
        {
            _size = size;
            if (_mat) cudaFree(_mat);
            cudaError_t res = cudaMalloc((void**)&_mat, sizeof(Real)*size);
            if (res != cudaSuccess) return res;

            res = cudaMemcpy(_mat, mat, sizeof(Real)*size, cudaMemcpyHostToDevice);
            if (res != cudaSuccess) return res;

            return res;
        }

        cudaError_t alloc(size_t size) throw(){
            _size = size;
            if (_mat) cudaFree(_mat);
            cudaError_t res = cudaMalloc((void**)&_mat, sizeof(Real)*size);
            if (res != cudaSuccess) return res;
            return res;
        }

        cudaError_t copy(cuVector &otherMat){
            cudaError_t res = alloc(otherMat.size());
            if (res != cudaSuccess) throw WHAT("cuda alloc error");
            vectorCopy(_size, otherMat._mat, _mat);
            return res;
        }
        void swap(cuVector &&otherMat) throw()
        {
            Real *tmp = otherMat._mat;
            otherMat._mat = _mat;
            _mat = tmp;
            _size = otherMat.size();
        }
    };



}//MonaCu

#endif //__MONACU_H__