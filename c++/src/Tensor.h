#ifndef _TENSOR_H
#define _TENSOR_H 

#include <vector>
#include <random>
#include <iostream>
#include <random>
#include <iomanip>

    class Tensor {
    public:
        Tensor(int rows, int columns, bool isRandom);

        Tensor *transpose();
        Tensor *clone();

        void setVal(int rows, int columns, int val);
        int getVal(int rows, int columns);

        std::vector<std::vector<double>> getValues();

        void print();
        int getNumRows();
        int getNumCols();

    private:
        double genRandom();

        int rows;
        int columns;

        std::vector<std::vector<double>> values;

    };

// } //namespace ad


#endif // _TENSOR_H