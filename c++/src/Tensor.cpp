// #include "Tensor.h"


// Tensor::Tensor(int rows, int columns, bool isRandom) {
//     this->rows = rows;
//     this->columns = columns;

//     for(int i=0; i<rows; i++) {
//         std::vector<double> colValues;

//         for(int j=0; j<columns; j++) {
//             double r = isRandom == true ? this->genRandom() : 0.00;
//             colValues.push_back(r);
//         }

//         this->values.push_back(colValues);
//     }
// }

// Tensor *Tensor::clone() {
//     Tensor *m = new Tensor(this->rows, this->columns, false);

//     for(int i = 0; i < this->rows; i++) {
//         for(int j = 0; j < this->columns; j++) {
//             m->setVal(i, j, this->getVal(i, j));
//         }
//     }
//     return m;
// }

// Tensor *Tensor::transpose() {
//     Tensor *m = new Tensor(this->columns, this->rows, false);

//     for(int i = 0; i < this->rows; i++) {
//         for(int j = 0; j < this->columns; j++) {
//             m->setVal(j, i, this->getVal(i, j));
//         }
//     }
//     return m;
// }

// double Tensor::genRandom() {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<> dis(-.0001, .0001);
//     return dis(gen);
// }

// void Tensor::print() {
//     for(int i = 0; i < this->rows; i++) {
//         for(int j = 0; j < this->columns; j++) {
//             std::cout << this->values.at(i).at(j) << "\t\t";
//         }
//         std::cout << std::endl;
//     }
// }


// void Tensor::setVal(int rows, int columns, int val) {
//     this->values.at(rows).at(columns) = val;
// }

// int Tensor::getVal(int rows, int columns) {
//     return this->values.at(rows).at(columns);
// }

// std::vector<std::vector<double>> Tensor::getValues() { 
//     return this->values; 
// }

// int Tensor::getNumRows() { 
//     return this->rows; 
// }

// int Tensor::getNumCols() { 
//     return this->columns; 
// }
