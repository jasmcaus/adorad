#ifndef _FUNCS_H
#define _FUNCS_H

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

// Find the sum of two Tensors 
Tensor matmul(Tensor &tens1, Tensor &tens2) {
    if(tens1.ndim != 2 || tens1.ndim != 2 ) {
        throw std::invalid_argument("Can sum only 2D Tensors currently.");
    }

    //TODO: Use the operator '+' to 
    return tens1 * tens2;

}

} //namespace ad 

#endif // _FUNCS_H