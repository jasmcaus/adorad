#pragma once 

#include "Tensor.h"

namespace ad {

// Transposes the original Tensor 
Tensor transpose(Tensor &tens){
    int rows = tens.shape()[0];
    int columns = tens.shape()[1]; 
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            tens._setVal(j, i, tens._getVal(i, j));
        }
    }
    return tens;
}


// Clone a Tensor 
Tensor clone(Tensor &tens) {
    int rows = tens.shape()[0];
    int columns = tens.shape()[1]; 

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            tens._setVal(i, j, tens._getVal(i, j));
        }
    }
    return tens;
}


// Find the sum of two Tensors 
Tensor sum(Tensor &tens1, Tensor &tens2) {
    if(tens1.shape() != tens2.shape()) {
        throw std::invalid_argument("Tensors must have the same dimensions.");
    }

    if(tens1.ndim != 2 || tens1.ndim != 2 ) {
        throw std::invalid_argument("Can sum only 2D Tensors currently.");
    }

    //TODO: Use the operator '+' to 
    return tens1 + tens2;

}


// Find the product of two Tensors 
Tensor matmul(Tensor &tens1, Tensor &tens2) {
    if(tens1.ndim != 2 || tens1.ndim != 2 ) {
        throw std::invalid_argument("Can multiply only 2D Tensors currently.");
    }

    //TODO: Use the operator '+' to 
    return tens1 * tens2;

}


// Identity Tensor 
Tensor eye(unsigned int dim) {
    Tensor c(dim, dim, false);

    for(unsigned int i=0; i<dim; i++) {
        for(unsigned int j=0; j<dim; j++) {
            if(i==j) {
                c._setVal(i, j, 1);
            }
        }
    }
    return c;
}

// Reverse Identity Tensor 
Tensor reveye(unsigned int dim) {
    Tensor c(dim, dim, false);

    for(unsigned int i=0; i<dim; i++) {
        for(unsigned int j=0; j<dim; j++) {
            if((i+j)==dim-1) {
                c._setVal(i, j, 1);
            }
        }
    }
    return c;
}

} //namespace ad 

#endif // _FUNCS_H