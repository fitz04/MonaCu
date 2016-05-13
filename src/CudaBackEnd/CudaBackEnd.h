#ifndef __MONACU_CUDABACKEND_H__
#define __MONACU_CUDABACKEND_H__

#include "MonaCuTypeDef.h"

extern void* vectorAlloc(int numElements, int typeSize);
extern MonaCu::error_t vectorFree(void *dst);
extern int getBlock();
extern int setBlock(int num);

#define CUDAFUNCTION_INSTANTIATION(type)                                     \
extern MonaCu::error_t getData(int numElements, const type *src, type *dst); \
extern MonaCu::error_t setData(int numElements, const type *src, type *dst); \
extern void vectorCopy(int numElements, const type *A, type *B);             \
extern void vectorAdd(int numElements, type *A, type *B, type *dst);         \
extern void vectorMinus(int numElements, type *A, type *B, type *dst);       \
extern void vectorMul(int numElements, type *A, type *B, type *dst);         \
extern void vectorDiv(int numElements, type *A, type *B, type *dst);         \


CUDAFUNCTION_INSTANTIATION(short);
CUDAFUNCTION_INSTANTIATION(int);
CUDAFUNCTION_INSTANTIATION(float);
CUDAFUNCTION_INSTANTIATION(double);

#endif //__MONACU_CUDABACKEND_H__

