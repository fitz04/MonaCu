//#include "MonaCu.h"


//namespace MonaCu
//{
//
//    class cublasInst
//    {
//    public:
//        static cublasInst* getInst(){
//            static cublasInst singleTonInst;
//            return &singleTonInst;
//        }
//
//        void cuCopy(int n, const float *src, float *dst){ cublasScopy(n, src, 1, dst, 1); }
//        void cuCopy(int n, const double *src, double *dst){ cublasDcopy(n, src, 1, dst, 1); }
//
//        void vecPlus(int n, float alpha, const float *src, float *dst){ cublasSaxpy(n, alpha, src, 1, dst, 1); }
//        void vecPlus(int n, double alpha, const double *src, double *dst){ cublasDaxpy(n, alpha, src, 1, dst, 1); }
//
//    private:
//        int threadsPerBlock;
//        cublasInst(void) :threadsPerBlock(256){
//            cublasInit();
//        }
//        ~cublasInst(void){
//            cublasShutdown();
//        }
//    };
//#define cuCopy(a, b, c) cublasInst::getInst()->cuCopy(a,b,c)
//#define vecPlus(a, b, c, d) cublasInst::getInst()->vecPlus(a, b, c, d)
// 
//};
