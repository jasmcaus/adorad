#include "src/Tensor.h"
#include <iostream>

int main() {
    std::cout << "Hello Tensors!" << std::endl;
    ad::Tensor m(3, 2, false, 0);

    m.print();
    return 1;
}