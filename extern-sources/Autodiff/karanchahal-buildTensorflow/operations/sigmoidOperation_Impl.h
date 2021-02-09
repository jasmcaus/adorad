/*
    This file contains the implementation of the forward and backward pass of
    the sigmoid operation.
*/

#include "operations/sigmoidOperation.h"

#ifndef __OP_IMPL_SIG_INCLUDED__
#define __OP_IMPL_SIG_INCLUDED__

/* 
    Backpropogation of the sigmoid operation. Example of a operation is as follows:
    
    F = sigmoid(x) is forward propogation
    The gradients would be as follows:
    1. dF/dx = (1 - F)*F

    This formula can be derived by hand.
*/
template <typename T>
void SigmoidOperation<T>::backward(Matrix<T> grad) {
    auto g = ((T)1 - this->t3->val) * this->t3->val; // (1 - F)*F
    this->t1->backward(grad * g); // follow chain rule by multiplying by incoming gradient
}

/* 
    Forward Propogation of the operation. Returns a tensor.

    TODO: Remove: See addition operation impl for more details
*/
template <typename T>
Tensor<T> SigmoidOperation<T>::forwardDeprecated() {
   // Not implemented as yet
   return NULL;
}

/* 
    Forward Propogation of the operation. Return pointer to the tensor.
*/
template <typename T>
Tensor<T>* SigmoidOperation<T>::forward() {
    auto val = matrixOps::sigmoid(this->t1->val);
    this->t3 = new Tensor<T>(val,this);
    return this->t3;
}

#endif

