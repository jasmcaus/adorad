#include "src/Tensor.h"
#include <iostream>

int main() {
    std::cout << "Hello Tensors!" << std::endl;
    ad::Tensor m1(3, 2, false, 3);
    ad::Tensor m2(3, 2, false, 3);

    m1 + m2;
    m1.print();
    m2.print();
    return 1;

}