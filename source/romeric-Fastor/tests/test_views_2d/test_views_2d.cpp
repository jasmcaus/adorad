#include <Fastor/Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5


template<typename T>
void run() {

    using std::abs;
    {
        // Check construction from views
        Tensor<T,15,18> a1; a1.iota(11);
        decltype(a1) a2 = a1(all,all);
        FASTOR_EXIT_ASSERT(abs(norm(a1)-norm(a2)) < Tol);
        decltype(a1) a3;
        a3 = a1(seq(0,-1),seq(first,last,1));
        FASTOR_EXIT_ASSERT(abs(norm(a1)-norm(a3)) < Tol);
        Tensor<T,3,5> a4 = a1(seq(3,last,4),seq(4,18,3));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 2205) < Tol);
        a4 += 5*a1(seq(3,last,4),seq(4,18,3)) / 4 ; // This covers eval
        FASTOR_EXIT_ASSERT(abs(norm(a4) - 1380.12329) < BigTol);

        // Assigning to a view from numbers/tensors/views
        a4(all,seq(first,2)) = 2;
        FASTOR_EXIT_ASSERT(abs(norm(a4) - 1087.620165) < 1e-3);
        a4(all,seq(first,2)) += 11;
        FASTOR_EXIT_ASSERT(abs(norm(a4) - 1088.0751) < 1e-3);
        a4(all,seq(first,2)) -= 11;
        FASTOR_EXIT_ASSERT(abs(norm(a4) - 1087.620165) < 1e-3);
        a4(seq(0,2,1),seq(first,2)) *= 3;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 3065.5) < Tol);
        a4(seq(0,2,1),seq(first,2)) /= 3;
        FASTOR_EXIT_ASSERT(abs(norm(a4) - 1087.620165) < 1e-3);

        // Check for both when right operand requires/does not require evaluation
        a4(all,all) = a4;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 3049.5) < Tol);
        // this is one of those quirkey IEEE cases where SP and DP differ
        // just check [a4 % trans(a4) % a4] with current a4 values
        // a4(all,all) = a4 + (a4 % trans(a4) % a4 - a4 % trans(a4) % a4);
        // FASTOR_EXIT_ASSERT(abs(a4.sum() - 3049.5) < Tol);
        a4(all,all) = a4 + trans(trans(a4)) - trans(trans(a4));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 3049.5) < Tol);
        a4(all,all) += 2*a4;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 9148.5) < Tol);
        a4(all,all) += trans(trans(a4)) - trans(trans(a4));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 9148.5) < Tol);
        a4(seq(first,last),seq(0,-1)) -= 2*a4;
        FASTOR_EXIT_ASSERT(abs(a4.sum() + 9148.5) < Tol);
        a4(all,all) -= trans(trans(a4)) - trans(trans(a4));
        FASTOR_EXIT_ASSERT(abs(a4.sum() + 9148.5) < Tol);
        a4(seq(0,2),seq(1,5,2)) *=10;
        FASTOR_EXIT_ASSERT(abs(a4.sum() + 23107.5) < Tol);
        a4(all,all) *= 1 + trans(trans(a4)) - trans(trans(a4));
        FASTOR_EXIT_ASSERT(abs(a4.sum() + 23107.5) < Tol);
        a4(seq(first,last),seq(first,last)) *=a4;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 139586878.125) < 20); // SP gives a large difference
        a4(all,all) /=a4;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < Tol);
        a4(all,all) /= 1 + trans(trans(a4)) - trans(trans(a4));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < BigTol);


        a4(all,all) = a1(seq(3,last,4),seq(4,18,3));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 2205) < Tol);
        decltype(a4) a5; a5.iota(12);
        a4.ones();
        a4(all,all) += a5(all,all);
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 300.) < Tol);
        a4(all,all) -= a5(all,all);
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < Tol);
        a4(seq(0,2),seq(0,2)) *= a5(seq(1,3),seq(2,4));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 99.) < Tol);
        a4(seq(0,2),seq(0,2)) /= a5(seq(1,3),seq(2,4));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < Tol);

        // Check overlap
        a4(all,all) = a4(all,all);
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < Tol);
        a4(all,all).noalias() = a4(all,all);
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < Tol);
        a4.iota(1);
        // a4(all,seq(0,4)) = a4(all,seq(1,5)); // this does not alias
        a4(all,seq(2,5)).noalias() = a4(all,seq(0,3));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 102.) < Tol);
        a4(all,seq(2,5)).noalias() += a4(all,seq(0,3));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 159.) < Tol);
        a4(all,seq(2,5)).noalias() += 2 + a4(all,seq(0,3));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 252.) < Tol);
        a4(all,seq(2,5)).noalias() -= a4(all,seq(0,3));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 153.) < Tol);
        a4(all,seq(2,5)).noalias() -= a4(all,seq(0,3))*3;
        FASTOR_EXIT_ASSERT(abs(a4.sum() + 90.) < Tol);
        a4(all,seq(2,5)).noalias() *= a4(all,seq(0,3));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 420.) < Tol);
        a4(all,seq(2,5)).noalias() *= a4(all,seq(0,3)) - 1;
        FASTOR_EXIT_ASSERT(abs(a4.sum() + 59101.) < Tol);
        a4(all,seq(2,5)).noalias() /= a4(all,seq(0,3)) + 13;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 14.26) < 1e-2);
        a4.arange(13);
        a4(all,seq(2,5)).noalias() /= a4(all,seq(0,3));
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 120.9967) < 1e-2);

        // Check scanning
        a5.iota();
        a5(all,last) = a4(all,first);
        FASTOR_EXIT_ASSERT(abs(a5.sum() - 132.) < Tol);
        a5(first,all) = a4(2,seq(first,last,1));
        FASTOR_EXIT_ASSERT(abs(a5.sum() - 163.25) < 1e-1);
        // scalar indexing
        a5(2,last) = 10;
        FASTOR_EXIT_ASSERT(abs(a5(2,last) - 10.) < Tol);
        a5.iota();
        FASTOR_EXIT_ASSERT(abs(a5(last,last) - 14.) < Tol);
        FASTOR_EXIT_ASSERT(abs(a5(last-1,last-1) - 8.) < Tol);
        FASTOR_EXIT_ASSERT(abs(a5(last-2,last-3) - 1.) < Tol);
        int counter = 0;
        for (int i=0; i<3; ++i) {
            for (int j=0; j<5; ++j) {
                FASTOR_EXIT_ASSERT(abs(a5(i,j) - counter) < Tol);
                FASTOR_EXIT_ASSERT(abs(a5(last-i,last-j) - (14-counter) ) < Tol);
                counter++;
            }
        }

        print(FGRN(BOLD("All tests passed successfully")));
    }
}

int main() {

    print(FBLU(BOLD("Testing 2-dimensional tensor views: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing 2-dimensional tensor views: double precision")));
    run<double>();


    return 0;
}
