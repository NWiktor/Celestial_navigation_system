#ifndef MATRIX
#define MATRIX

// Standard headers
#include <stdio.h>
#include <fstream> // for file access
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>

// Own headers
using std::vector;
using std::tuple;

class Matrix {
  private:
    unsigned m_rowSize;
    unsigned m_colSize;
    vector<vector<double> > m_matrix;

  public:
    // Three costructors:
    Matrix(unsigned, unsigned, double);
    // Matrix(const char *); // read data from file - not needed
    Matrix(const Matrix &); // copy data
    ~Matrix(); // Destructor

    // Matrix Operations
    Matrix operator+(Matrix &);
    Matrix operator-(Matrix &);
    Matrix operator*(Matrix &);
    Matrix transpose();

    // Scalar Operations
    Matrix operator+(double);
    Matrix operator-(double);
    Matrix operator*(double);
    Matrix operator/(double);

    // Aesthetic Methods
    double& operator()(const unsigned &, const unsigned &); // Get value by index
    void print() const; // for Matlab
    unsigned getRows() const;
    unsigned getCols() const;

    // Power Iteration
    tuple<Matrix, double, int> powerIter(unsigned, double);

    // Deflation
    Matrix deflation(Matrix &, double&);

};

#endif
