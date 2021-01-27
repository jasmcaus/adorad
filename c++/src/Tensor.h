#ifndef _TENSOR_H
#define _TENSOR_H 

#include <vector>
#include <random>
#include <iostream>
#include <random>
#include <iomanip>

namespace ad {

class Tensor {
public:
    /**
     * Constructor for a 2D Tensor. If random is true, the 
     * matrix is initialized with random values between 1 and 0, otherwise it is 
     * initialized with init value.
     * 
     * If one dimension is zero, 'invalid argument' exception is raised.
     */
    Tensor(int rows, int columns, bool isRandom, double init){
        if(rows == 0 || columns == 0) 
            throw std::invalid_argument("Tensor dimensions can't be zero");

        this->rows = rows;
        this->columns = columns;


        if(isRandom)
            _set_random_tensor();
        else
            _set_tensor(init);
    }

    // /**
    //  * Copy constructor. Create a new tensor copying the values of the Tensor.
    //  */

    // Tensor(const Tensor &tens)
    // {
    //     rows = tens.rows;
    //     columns = tens.columns;

    //     for(int r = 0; r < tens.shape()[0]; ++r) {
    //         std::vector<double> row;
    //         for(int c = 0; c < tens.shape()[1]; ++c) {
    //             row.push_back(tens(r, c));
    //         }
    //         values.push_back(row);
    //     }
    // }

    // Return shape of the matrix in a vector of ints.
    std::vector<int> shape() {
        return std::vector<int>{rows, columns};
    }

    // Doesn't transpose the original Tensor
    Tensor transpose(){
        Tensor m(this->columns, this->rows, false, 0);

        for(int i = 0; i < this->rows; i++) {
            for(int j = 0; j < this->columns; j++) {
                m.setVal(j, i, this->getVal(i, j));
            }
        }
        return m;
    }

    Tensor clone() {
        Tensor m(this->rows, this->columns, false, 0);

        for(int i = 0; i < this->rows; i++) {
            for(int j = 0; j < this->columns; j++) {
                m.setVal(i, j, this->getVal(i, j));
            }
        }
        return m;
    }

    void print() {
        for(int i = 0; i < this->rows; i++) {
            for(int j = 0; j < this->columns; j++) {
                std::cout << this->values.at(i).at(j) << "\t\t";
            }
            std::cout << std::endl;
        }
    }

    std::vector<std::vector<double>> getValues() { 
        return this->values; 
    }

    void setVal(int rows, int columns, int val) {
        this->values.at(rows).at(columns) = val;
    }
    
    int getVal(int rows, int columns) {
        return this->values.at(rows).at(columns);
    }

    int getNumRows() { 
        return this->rows; 
    }
    int getNumCols() { 
        return this->columns; 
    }



    // Operator stuff
    const double& operator()(int r, int c) const {
        return this->values.at(r).at(c);
    }

private:
    double genRandom() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-.0001, .0001);
        return dis(gen);
    }

    void _set_tensor(double init) {
        for(int i=0; i<rows; i++) {
            std::vector<double> colValues;

            for(int j=0; j<columns; j++) {
                colValues.push_back(init);
            }

            this->values.push_back(colValues);
        }
    }

    void _set_random_tensor() {
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

    int rows;
    int columns;

    std::vector<std::vector<double>> values;

};

} //namespace ad


#endif // _TENSOR_H