#include "adorad/Tensor.h"
#include "adorad/TensorFuncs.h"
#include <iostream>


int main() {
    std::cout << "Hello Tensors!" << std::endl;

    ad::Tensor c = ad::eye(4);
    ad::Tensor d = ad::reveye(4);

    ad::Tensor sum = c*d;

    c.print();
    std::cout << "New" << std::endl;
    d.print();
    std::cout << "New" << std::endl;
    sum.print();

    return 1;

}