#include "src/Tensor.h"
#include <iostream>

int main() {
    std::cout << "Hello Tensors!" << std::endl;
    ad::Tensor m1(3, 6, false, 3);
    ad::Tensor m2(3, 2, false, 3);

    ad::Tensor c = m1 * m2;
    c.print();

    return 1;

}