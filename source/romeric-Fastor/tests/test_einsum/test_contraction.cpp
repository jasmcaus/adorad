#include <Fastor/Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2


template<typename T>
void run() {

    using std::abs;
    enum {i,j,k,l,m,n};
    {
        Tensor<T,2,2> II; II.eye();
        auto II_ijkl = contraction<Index<i,j>,Index<k,l>>(II,II);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(0,0,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(1,1,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(1,1,1,1) - 1) < Tol);
        auto II_ijkl2 = outer(II,II);
        FASTOR_EXIT_ASSERT(all_of(II_ijkl==II_ijkl2));
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(0,1,0,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(1,0,1,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(1,1,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(norm(II_ijkl) - norm(II_ikjl)) < Tol);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);
        FASTOR_EXIT_ASSERT(abs(II_iljk(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(0,1,1,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(1,0,0,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(1,1,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(norm(II_ijkl) - norm(II_iljk)) < Tol);


        Tensor<T,2,2> A, B; A.iota(101); B.iota(77);
        Tensor<T,2> D = {T(45.5), T(73.82)};
        auto bb_ijkl = contraction<Index<i,j>,Index<k,l>>(A,B);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ijkl) - 32190.178937) < HugeTol);
        auto bb_ikjl = permutation<Index<i,k,j,l>>(bb_ijkl);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ikjl) - 32190.178937) < HugeTol);
        auto bb_iljk = permutation<Index<i,l,j,k>>(bb_ijkl);
        FASTOR_EXIT_ASSERT(abs(norm(bb_iljk) - 32190.178937) < HugeTol);
        auto bb_ijkl2 = outer(A,B);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ijkl2) - 32190.178937) < HugeTol);


        FASTOR_EXIT_ASSERT(abs(bb_ijkl(0,1,0,1)-bb_ikjl(0,0,1,1)) < BigTol);
        FASTOR_EXIT_ASSERT(abs(bb_ijkl(0,1,0,1)-bb_iljk(0,0,1,1)) < BigTol);

        for (auto ii=0; ii< 2; ii++) {
            for (auto jj=0; jj< 2; jj++) {
                for (auto kk=0; kk< 2; kk++) {
                    for (auto ll=0; ll< 2; ll++) {
                        FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,jj,kk,ll) - bb_ikjl(ii,kk,jj,ll) ) < BigTol);
                        // FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,jj,kk,ll) - bb_iljk(ii,ll,jj,kk) ) < BigTol); // By definition this cannot be
                        FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,ll,jj,kk) - bb_iljk(ii,jj,kk,ll) ) < BigTol);
                    }
                }
            }
        }

        auto bD_ijk = contraction<Index<i,j>,Index<k>>(A,D);
        FASTOR_EXIT_ASSERT(abs(norm(bD_ijk) - 17777.8111) < HugeTol);
        auto bD_ikj = permutation<Index<i,k,j>>(bD_ijk);
        FASTOR_EXIT_ASSERT(abs(norm(bD_ikj) - 17777.8111) < HugeTol);
        auto bD_jki = permutation<Index<j,k,i>>(bD_ijk);
        FASTOR_EXIT_ASSERT(abs(norm(bD_jki) - 17777.8111) < HugeTol);
        auto bD_kji = permutation<Index<k,j,i>>(bD_ijk);
        FASTOR_EXIT_ASSERT(abs(norm(bD_kji) - 17777.8111) < HugeTol);
    }


    {
        Tensor<T,3,3> II; II.eye();
        auto II_ijkl = contraction<Index<i,j>,Index<k,l>>(II,II);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(0,0,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(0,0,2,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(1,1,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(1,1,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(1,1,2,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(2,2,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(2,2,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ijkl(2,2,2,2) - 1) < Tol);
        auto II_ijkl2 = outer(II,II);
        FASTOR_EXIT_ASSERT(all_of(II_ijkl==II_ijkl2));
        auto II_ikjl = permutation<Index<i,k,j,l>>(II_ijkl);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(0,1,0,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(0,2,0,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(1,0,1,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(1,1,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(1,2,1,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(2,0,2,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(2,1,2,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_ikjl(2,2,2,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(norm(II_ijkl) - norm(II_ikjl)) < Tol);
        auto II_iljk = permutation<Index<i,l,j,k>>(II_ijkl);
        FASTOR_EXIT_ASSERT(abs(II_iljk(0,0,0,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(0,1,1,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(0,2,2,0) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(1,0,0,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(1,1,1,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(1,2,2,1) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(2,0,0,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(2,1,1,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(II_iljk(2,2,2,2) - 1) < Tol);
        FASTOR_EXIT_ASSERT(abs(norm(II_ijkl) - norm(II_iljk)) < Tol);


        Tensor<T,3,3> A, B; A.iota(65); B.iota(T(13.2));
        Tensor<T,3> D = {T(124.36), T(-37.29), T(5.61)};
        auto bb_ijkl = contraction<Index<i,j>,Index<k,l>>(A,B);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ijkl) - 10808.437) < HugeTol);
        auto bb_ikjl = permutation<Index<i,k,j,l>>(bb_ijkl);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ikjl) - 10808.437) < HugeTol);
        auto bb_iljk = permutation<Index<i,l,j,k>>(bb_ijkl);
        FASTOR_EXIT_ASSERT(abs(norm(bb_iljk) - 10808.437) < HugeTol);
        auto bb_ijkl2 = outer(A,B);
        FASTOR_EXIT_ASSERT(abs(norm(bb_ijkl2) - 10808.437) < HugeTol);

        FASTOR_EXIT_ASSERT(abs(bb_ijkl(0,1,0,1)-bb_ikjl(0,0,1,1)) < BigTol);
        FASTOR_EXIT_ASSERT(abs(bb_ijkl(0,1,0,1)-bb_iljk(0,0,1,1)) < BigTol);

        for (auto ii=0; ii< 3; ii++) {
            for (auto jj=0; jj< 3; jj++) {
                for (auto kk=0; kk< 3; kk++) {
                    for (auto ll=0; ll< 3; ll++) {
                        FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,jj,kk,ll) - bb_ikjl(ii,kk,jj,ll) ) < BigTol);
                        // FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,jj,kk,ll) - bb_iljk(ii,ll,jj,kk) ) < BigTol); // By definition this cannot be
                        FASTOR_EXIT_ASSERT(abs( bb_ijkl(ii,ll,jj,kk) - bb_iljk(ii,jj,kk,ll) ) < BigTol);
                    }
                }
            }
        }

        auto bD_ijk = contraction<Index<i,j>,Index<k>>(A,D);
        FASTOR_EXIT_ASSERT(abs(norm(bD_ijk) - 26918.8141) < HugeTol);
        auto bD_ikj = permutation<Index<i,k,j>>(bD_ijk);
        FASTOR_EXIT_ASSERT(abs(norm(bD_ikj) - 26918.8141) < HugeTol);
        auto bD_jki = permutation<Index<j,k,i>>(bD_ijk);
        FASTOR_EXIT_ASSERT(abs(norm(bD_jki) - 26918.8141) < HugeTol);
    }

    {
        Tensor<T,5,5,5> A; Tensor<T,5> B;
        A.iota(1); B.iota(2);

        auto C = contraction<Index<i,j,k>,Index<j>>(A,B);
        FASTOR_EXIT_ASSERT(abs(C.sum() - 32750) < Tol);
        auto D = contraction<Index<j>,Index<i,j,k>>(B,A);
        FASTOR_EXIT_ASSERT(abs(D.sum() - 32750) < Tol);
        auto E = contraction<Index<i,j,k>,Index<i,j,l>>(A,A);
        FASTOR_EXIT_ASSERT(abs(E.sum() - 3293125) < Tol);
        auto F = contraction<Index<i>,Index<k>>(B,B);
        FASTOR_EXIT_ASSERT(abs(F.sum() - 400) < Tol);
    }

    {
        Tensor<double,5,5> A; Tensor<double,5,5,5,5> B; Tensor<double,5> C;
        A.iota(); B.iota(); C.iota();
        auto D = contraction<Index<k,j>,Index<k,i,l,j>,Index<l>>(A,B,C);
        FASTOR_EXIT_ASSERT(abs(D.sum() - 6.32e+06) < BigTol);
    }

    {
        Tensor<T,3,2> As; As.iota(1);
        Tensor<T,3> bs; bs.fill(1);
        Tensor<T,2> cs; cs.fill(2);

        FASTOR_EXIT_ASSERT((contraction<Index<i,j>,Index<j>>(As,cs)).sum() - 42. < Tol);
        FASTOR_EXIT_ASSERT((contraction<Index<i,j>,Index<i>>(As,bs)).sum() - 21. < Tol);
        FASTOR_EXIT_ASSERT((contraction<Index<i>,Index<i,j>>(bs,As)).sum() - 21. < Tol);
        FASTOR_EXIT_ASSERT((contraction<Index<j>,Index<i,j>>(cs,As)).sum() - 42. < Tol);
    }

    {
        // Test strided_contraction when second tensor disappears
        Tensor<T,4,4,4> a; a.iota(1);
        Tensor<T,4,4> b; b.iota(1);

        Tensor<T,4> c1 = contraction<Index<i,j,k>,Index<j,k> >(a,b);
        Tensor<T,4> c2 = contraction<Index<i,j,k>,Index<i,k> >(a,b);
        Tensor<T,4> c3 = contraction<Index<i,j,k>,Index<i,j> >(a,b);

        FASTOR_EXIT_ASSERT(abs(c1(0) - 1496.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c1(1) - 3672.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c1(2) - 5848.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c1(3) - 8024.) < Tol);

        FASTOR_EXIT_ASSERT(abs(c2(0) - 4904.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c2(1) - 5448.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c2(2) - 5992.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c2(3) - 6536.) < Tol);

        FASTOR_EXIT_ASSERT(abs(c3(0) - 5576.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c3(1) - 5712.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c3(2) - 5848.) < Tol);
        FASTOR_EXIT_ASSERT(abs(c3(3) - 5984.) < Tol);
    }

    // Catches the bug in 3 network contraction
    {
        enum {a,b,c,d,e,f,g,h,i,j,k,l};

        Tensor<double,3,3> A = 0;
        A(0,0) = 1;
        A(1,1) = 1;
        A(2,2) = -1;

        Tensor<double,3,3,3> B = 0;
        B(0,0,0) = 1;
        B(0,2,2) = 1;

        B(1,0,1) = 1;
        B(1,1,0) = -1;

        B(2,0,2) = -1;
        B(2,2,0) = 1;

        auto C = contraction<Index<a,c>,Index<f,g>,Index<c,f,b>,Index<g,d,a>>(A, A, B, B);
        FASTOR_EXIT_ASSERT(abs(C(0,0) + 1.) < Tol);
        FASTOR_EXIT_ASSERT(abs(C(1,1)) < BigTol);
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

int main() {

    print(FBLU(BOLD("Testing tensor contraction: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing tensor contraction: double precision")));
    run<double>();

    return 0;
}
