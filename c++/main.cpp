#include "../srcTensor.h"
#include <iostream>

int main() {
    std::cout << "Hello Tensors!" << std::endl;
    Tensor *m = new Tensor(3, 2, true);

    m->print();
    return 1;
}