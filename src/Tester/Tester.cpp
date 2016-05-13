// Tester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "MonaCu.h"

typedef double Real;

int _tmain(int argc, _TCHAR* argv[])
{
    vector<Real> a, b;
    for (Real i=1; i < 5; i++)
    {
        a.push_back(i);
        b.push_back(i * i);
    }
    int arraySize = a.size();

    MonaCu::cuVector<Real> lhs(a.data(), arraySize);
    MonaCu::cuVector<Real> rhs(b.data(), arraySize);
    MonaCu::cuVector<Real> res = lhs;// +rhs;
    cout << "\ncopy lhs" << endl;
    res.print();

    cout << "\ncopy rhs" << endl;
    res = rhs;
    res.print();

    cout << "\nlhs + rhs" << endl;
    res = lhs + rhs;
    res.print();
    
    cout << "\nlhs - rhs" << endl;
    res = lhs - rhs;
    res.print();

    cout << "\nrhs + lhs - lhs" << endl;
    res = rhs + lhs - lhs;
    res.print();

    cout << "\nlhs * rhs" << endl;
    res = lhs * rhs;
    res.print();

    cout << "\nlhs + lhs * rhs" << endl;
    res = lhs + lhs * rhs;
    res.print();

    cout << "\n(lhs + lhs) * rhs" << endl;
    res = (lhs + lhs) * rhs;
    res.print();

    cout << "\nlhs + lhs * rhs - lhs" << endl;
    res = lhs + lhs * rhs - lhs;
    res.print();

    cout << "\nlhs + lhs * rhs - lhs * rhs" << endl;
    res = lhs + lhs * rhs - lhs * rhs;
    res.print();

    cout << "\n(lhs - lhs) * rhs :: zero multipulication" << endl;
    res = (lhs - lhs) * rhs;
    res.print();
    
    cout << "\nlhs / rhs" << endl;
    res = (lhs / rhs);
    res.print();

    cout << "\nlhs / lhs" << endl;
    res = (lhs / lhs);
    res.print();

    cout << "\n(lhs - lhs) / rhs :: zero devide" << endl;
    res = (lhs - lhs) / rhs;
    res.print();

    cout << "\nrhs / (lhs - lhs) :: zero devide" << endl;
    res = rhs / (lhs - lhs);
    res.print();

	return 0;
}

