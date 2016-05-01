
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "MonaCu.h"

int main()
{
    const int arraySize = 5;
    const float a[arraySize] = { 1, 2, 3, 4, 5 };
    const float b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };


    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);


    MonaCu::cuVector<float> lhs(a, arraySize);
    MonaCu::cuVector<float> rhs(b, arraySize);
    MonaCu::cuVector<float> res = lhs;// +rhs;
    res.print();
    res = rhs;
    res.print();
    res = lhs + rhs;
    cout << "plus" << endl;
    res.print();
    res = rhs - lhs;
    cout << "minus" << endl;
    res.print();
    res = rhs + lhs - lhs;
    cout << "plus, minus" << endl;
    res.print();

    return 0;
}