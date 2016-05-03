template<class Real>
void vectorCopy(int numElements, const Real *A, Real *B);

template<class Real>
void vectorAdd(int numElements, Real *A, Real *B, Real *dst);

template<class Real>
void vectorMinus(int numElements, Real *A, Real *B, Real *dst);

template<class Real>
void vectorMul(int numElements, Real *A, Real *B, Real *dst);

template<class Real>
void vectorDiv(int numElements, Real *A, Real *B, Real *dst);