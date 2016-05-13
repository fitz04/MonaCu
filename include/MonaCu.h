#ifndef __MONACU_H__
#define __MONACU_H__

#include <vector>
#include <iostream>
#include <sstream>
#include <exception>
#include <utility>
#include <type_traits>
#include "MonaCuTypeDef.h"
#include "../src/CudaBackEnd/CudaBackEnd.h"


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
    class cuVector
    {
    public:

        cuVector() :_mat(0), _size(0){};
        virtual ~cuVector(){ 
            vectorFree(_mat); 
            _mat = NULL; 
        }

        cuVector(const Real* hostArray, size_t size){
            static_assert(is_arithmetic<Real>::value, "Arithmetic type required.");
            error_t res = this->alloc(hostArray, size);
            if (res != Success) throw WHAT("cuda alloc error"); // Success == 0
        }

        cuVector(vector<Real> &hostVector){
            error_t res = this->alloc(hostVector.data(), hostVector.size());
            if (res != Success) throw WHAT("cuda alloc error"); //Success == 0
        }

        cuVector(cuVector<Real> &otherMat){ copy(otherMat); }
        cuVector(cuVector<Real> &&otherMat){
            std::swap(_mat, otherMat._mat);
            _size = otherMat.size();
        }
        void operator=(cuVector<Real> &otherMat){
            copy(otherMat);
        }
        void operator=(cuVector<Real> &&otherMat){
            std::swap(_mat, otherMat._mat);
            _size = otherMat.size();
        }

#define opTemplate(OP, FUNCTOR)                                \
            cuVector<Real> OP (cuVector<Real> &rhs){           \
            if (_size != rhs.size()) throw WHAT("Size error"); \
            cuVector<Real> tmp = *this;                        \
            FUNCTOR(_size, _mat, rhs._mat, tmp._mat);          \
            return tmp; }                                      \

        opTemplate(operator+, vectorAdd);
        opTemplate(operator-, vectorMinus);
        opTemplate(operator*, vectorMul);
        opTemplate(operator/, vectorDiv);

        Real* begin(){ return _mat; }
        Real* end(){ return _mat + _size; }
        size_t size(){ return _size; }
        void print(void){
            stringstream out;
            vector<Real> tmp(_size);
            error_t res = getData(_size, _mat, tmp.data());
            if (res != Success) throw WHAT("getData Error");
            for (size_t i = 0; i < tmp.size(); i++)
            {
                out << tmp[i];
                if ((i + 1) % 10) out << ",\t";
                else out << endl;
            }
            cout << out.str() << endl;
        }

    private:
        Real* _mat;
        size_t _size;

        error_t alloc(const Real *mat, size_t size) throw()
        {
            _size = size;
            if (_mat) vectorFree(_mat);
            _mat = (Real*)vectorAlloc(size, sizeof(Real));
            if (_mat == NULL) return Fail;
        
            error_t res = setData(size, mat, _mat);
            if (res != Success) return res;
        
            return res;
        }

        error_t alloc(size_t size) throw(){
            _size = size;
            if (_mat) vectorFree(_mat);
            _mat = (Real*)vectorAlloc(size, sizeof(Real));
            if (_mat == NULL) return Fail;
            return Success;
        }

        error_t copy(cuVector<Real> &otherMat){
            error_t res = alloc(otherMat.size());
            if (res != Success) throw WHAT("cuda alloc error");
            vectorCopy(_size, otherMat._mat, _mat);
            return res;
        }
    };
}//MonaCu

#endif //__MONACU_H__