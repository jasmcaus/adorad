#include "src/Tensor.h"
#include "src/TensorFuncs.h"
#include <iostream>

int main() {
    std::cout << "Hello Tensors!" << std::endl;

    ad::Tensor c = ad::eye(4);
    ad::Tensor d = ad::reveye(4);

    c.print();
    std::cout << "New" << std::endl;
    d.print();

    return 1;

}