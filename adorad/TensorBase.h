#pragma once 

#include <vector>
#include <random>
#include <iostream>
#include <random>
#include <iomanip>

namespace ad {

class Tensor {
public:
    const int ndim = 2; // 2D Tensor 
    const long numelem = rows*columns;
    const int sci_mode = 1; //sci-mode for now

    const long numel() const { return this->numelem; }
    const int ndimen() const { return this->ndim; }
    const int mode() const { return this->sci_mode; }


    Tensor(int rows, int columns, bool isRandom, double init=0.0){
        if(rows == 0 || columns == 0) 
            throw std::invalid_argument("Tensor dimensions cannot be zero.");

        this->rows = rows;
        this->columns = columns;

        if(isRandom)
            __set_random_tensor();
        else
            __set_tensor(init);
    }


    // Create a new tensor copying the values of the original Tensor.
    Tensor(const Tensor &tens) {
        rows = tens.rows;
        columns = tens.columns;

        for(int r = 0; r < tens.shape()[0]; ++r) {
            std::vector<double> row;
            for(int c = 0; c < tens.shape()[1]; ++c) {
                row.push_back(tens(r, c));
            }
            values.push_back(row);
        }
    }


    // Create a new Tensor given all the values. Shape of the Tensor is infered by the argument.
    Tensor(std::vector<std::vector<double>> &x) {
        rows = x.size();
        columns = x[0].size();

        for(int r = 0; r < rows; ++r) {
            if(x.at(r).size() != columns)
                throw std::invalid_argument("Rows must have the same number of columns.");
            
            this->values.push_back(x[r]);
        }
    }

    // Pretty print is implemented in Formatting.h 
    void print() {
        for(int i = 0; i < this->rows; i++) {
            for(int j = 0; j < this->columns; j++) {
                std::cout << this->values.at(i).at(j) << "\t";
            }
            std::cout << std::endl;
        }
    }


    //
    // Properties
    // 

    int64_t dim() const;
    int64_t size() const; 
    
    // Returns true if the Tensor was originally defined on the GPU
    bool on_cuda() const;
    // Returns true if the Tensor was originally defined on the CPU
    bool on_cpu() const;

    // Is a boolean tensor?
    bool is_bool() const;


    // Return shape of the matrix in a vector of ints.
    std::vector<int> shape() const { 
        return std::vector<int>{rows, columns}; 
    }

    std::vector<std::vector<double>> getTensorValues() { 
        return this->values; 
    }

    int getNumRows() { return this->rows; }
    int getNumCols() { return this->columns; }


    // Find the sum of two Tensors 
    Tensor sum(Tensor &tens) {
        if( rows != tens.shape()[0] || columns != tens.shape()[1])
            throw std::invalid_argument("Tensors must have the same dimensions.");

        Tensor sum(tens);
        for(int r = 0; r < tens.shape()[0]; ++r) {
            for(int c = 0; c < tens.shape()[1]; ++c) {
                sum(r, c) = sum(r, c) + (*this)(r, c);
            }
        }
        return sum;
    }

    // Find the sum of two Tensors 
    // No. of Columns of the 1st Tensor must be = no. of rows of the 2nd Tensor
    Tensor matmul(Tensor &tens) {
        if(this->rows != tens.shape()[0])
            throw std::invalid_argument("Tensor dimensions not compatible for Tensor multiplication");

        Tensor m(this->rows, tens.shape()[1], false, 0);
        for(int r = 0; r < this->rows; ++r) {  // loop over rows of Tensor 1
            for(int c = 0; c < tens.shape()[1]; ++c) { // loop over cols of Tensor 2
                double prod = 0;
                for(int i = 0; i < this->columns; ++i) {
                    prod += ((*this)(r, i) * tens(i, c));
                }
                m(r, c) = prod;
            }
        }
        return m;
    }


    // Operator stuff
    double& operator()(int r, int c) { 
        return this->values.at(r).at(c); 
    }
    const double& operator()(int r, int c) const { 
        return values.at(r).at(c); 
    }


    // Tensor sum 
    Tensor operator+(const Tensor &tens) {
        if( rows != tens.shape()[0] || columns != tens.shape()[1])
            throw std::invalid_argument("Tensors must have the same dimensions.");

        Tensor sum(tens);
        for(int r = 0; r < tens.shape()[0]; ++r) {
            for(int c = 0; c < tens.shape()[1]; ++c) {
                sum(r, c) = sum(r, c) + (*this)(r, c);
            }
        }

        return sum;
    }

    // No. of Columns of the 1st Tensor must be = no. of rows of the 2nd Tensor
    Tensor operator*(const Tensor &tens) {
        if(this->columns != tens.shape()[0])
            throw std::invalid_argument("Tensor dimensions not compatible for Tensor multiplication");

        Tensor m(this->rows, tens.shape()[1], false, 0);

        for(int r = 0; r < this->rows; ++r) {  // loop over rows of Tensor 1
            for(int c = 0; c < tens.shape()[1]; ++c) { // loop over cols of Tensor 2
                double prod = 0;
                for(int i = 0; i < this->columns; ++i) {
                    prod += ((*this)(r, i) * tens(i, c));
                }
                m(r, c) = prod;
            }
        }
        return m;
    }


    // Similar to __repr__ in Python
    // TODO: Work on this:

    // operator std:string() const {
    //     return "Tensor String"
    // }


    // std::string toString() const {
    //     std::string base_str;

    //     if (scalar_type() == ScalarType::Undefined) { 
    //         base_str = "UndefinedType"; 
    //     } else { 
    //         base_str = std::string(ad::toString(options().backend())) + ad::toString(scalar_type()) + "Type";  
    //     }
        
    //     return base_str;
    // }
    
    // Only to be used internally
    void _setVal(int rows, int columns, int val) { 
        this->values.at(rows).at(columns) = val; 
    }

    int _getVal(int rows, int columns) {
        return this->values.at(rows).at(columns); 
    }

private:
    int rows;
    int columns;

    std::vector<std::vector<double>> values;

    double genRandom() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-.0001, .0001);
        return dis(gen);
    }

    void __set_tensor(double init) {
        for(int i=0; i<rows; i++) {
            std::vector<double> colValues;

            for(int j=0; j<columns; j++) {
                colValues.push_back(init);
            }

            this->values.push_back(colValues);
        }
    }

    void __set_random_tensor() {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0, 1);

        for(int r = 0; r < rows; ++r) {
            std::vector<double> row;
            for(int c = 0; c < columns; ++c) {
                row.push_back(distribution(generator));
            }
            values.push_back(row);
        }
    }

}; // class Tensor

} //namespace ad