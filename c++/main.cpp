#include "src/Tensor.h"
#include <iostream>

int main() {
    std::cout << "Hello Tensors!" << std::endl;

    std::vector<std::vector<double>> a_init { {1, 2, 3}, {4, 5, 6} };
    std::vector<std::vector<double>> b_init { {1, 1}, {1, 1}, {1, 1} };
    std::vector<std::vector<double>> c_init { {6, 6}, {15, 15} };

    ad::Tensor m1(a_init);
    ad::Tensor m2(b_init);
    ad::Tensor expected(c_init);

    ad::Tensor actual = m1*m2; 

    actual.print();


    return 1;

}