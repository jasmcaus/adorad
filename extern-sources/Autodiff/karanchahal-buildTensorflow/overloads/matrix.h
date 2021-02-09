/*
    This file defines the various mathematical overloads of the Matrix Class.
*/

#include <types/matrix.h>

#ifndef __MATRIX_OPS_INCLUDED__   
#define __MATRIX_OPS_INCLUDED__  


namespace matrixOps {

    // Sigmoid Operation
    template<typename T>
    Matrix<T> sigmoid(const Matrix<T> &a) {
        return (T)1/((T)1 + (((T)-1)*a).exp());
    }

    // Power Operation
    template<typename T>
    Matrix<T> power(Matrix<T> &a, T pow) {
        return a^pow;
    }
};

// Overloaded function for printing matrix: cout<<matrix<<endl;
template<typename T>
ostream & operator << (ostream &out, Matrix<T> &m) {
    vector<int> stack;
    return m.print(out,stack,0);
}

// Divison with a scalar as divident
template<typename T>
Matrix<T> operator / (const T t, const Matrix<T> &rhs) {
    auto res =  t/rhs.val;
    auto resShape = rhs.shape;
    return Matrix<T>(res, resShape);
}

// Multiplication with a scalar
template<typename T>
Matrix<T> operator * (const T t, const Matrix<T> &rhs) {
    auto res =  t*rhs.val;
    auto resShape = rhs.shape;
    return Matrix<T>(res, resShape);
}

// Addition with a scalar
template<typename T>
Matrix<T> operator + (const T t, const Matrix<T> &rhs) {
    auto res =  t+rhs.val;
    auto resShape = rhs.shape;
    return Matrix<T>(res, resShape);
}

// Subtraction with a scalar
template<typename T>
Matrix<T> operator - (const T t, const Matrix<T> &rhs) {
    auto res =  t-rhs.val;
    auto resShape = rhs.shape;
    return Matrix<T>(res, resShape);
}

#endif