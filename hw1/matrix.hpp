#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <ostream>
#include <stdexcept>
#include <vector>

/*
This is the pure abstract base class specifying general set of functions for a
square matrix.

Concrete classes for specific types of matrices, like MatrixSymmetric, should
implement these functions.
*/
template <typename T>
class Matrix {
  // Returns reference to matrix element (i, j).
  virtual T& operator()(int i, int j) = 0;
  // Number of non-zero elements in matrix.
  virtual unsigned NormL0() const = 0;
  // Enables printing all matrix elements using the overloaded << operator
  virtual void Print(std::ostream& ostream) = 0;

  template <typename U>
  friend std::ostream& operator<<(std::ostream& stream, Matrix<U>& m);
};

/* TODO: Overload the insertion operator by modifying the ostream object */
template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix<T>& m) {
  m.Print(stream);
  return stream;
}

/* MatrixDiagonal Class is a subclass of the Matrix class */
template <typename T>
class MatrixDiagonal : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;
  T def;

  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

 public:
  // TODO: Default constructor
  MatrixDiagonal() : n_(0) {}

  // TODO: Constructor that takes matrix dimension as argument
  MatrixDiagonal(const int n) : n_(n) { 
    if (n < 0) throw std::invalid_argument("Received negative size");
    else data_.resize(n);
  }

  // TODO: Function that returns the matrix dimension
  unsigned int size() const { return n_; }

  // TODO: Function that returns reference to matrix element (i, j).
  T& operator()(int i, int j) override {
    if (i >= (int)n_ || j >= (int)n_ || i < 0 || j < 0) throw std::out_of_range("Indices out of range");
    if (i == j) return data_[i];
    else{
      def = T();
      return def;
    }
  }

  // TODO: Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override {
    unsigned m = 0;
    for (unsigned i = 0; i < n_; i++){
      if (data_[i] != 0) m++;
    }
    return m;
  }

  // TODO: Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& ostream) override {
    for (unsigned i = 0; i < n_; i++){
      for (unsigned j = 0; j < n_; j++){
        if (i != j) ostream << 0 << " ";
        else ostream << data_[i] << " ";
      }
      ostream << "\n";
    }
  }
};

/* MatrixSymmetric Class is a subclass of the Matrix class */
template <typename T>
class MatrixSymmetric : public Matrix<T> {
 private:
  // Matrix Dimension. Equals the number of columns and the number of rows.
  unsigned int n_;
  // Elements of the matrix. You get to choose how to organize the matrix
  // elements into this vector.
  std::vector<T> data_;

 public:
  // TODO: Default constructor
  MatrixSymmetric() : n_(0) {}

  // TODO: Constructor that takes matrix dimension as argument
  MatrixSymmetric(const int n) : n_(n) {
    if (n < 0) throw std::invalid_argument("Received negative size");
    else data_.resize((n+1)*n/2);
  }

  // TODO: Function that returns the matrix dimension
  unsigned int size() const { return n_; }

  // TODO: Function that returns reference to matrix element (i, j).
  T& operator()(int i, int j) override { 
    if (i >= (int)n_ || j >= (int)n_ || i < 0 || j < 0) throw std::out_of_range("Indices out of range");
    if (i > j){
      unsigned h = i;
      i = j;
      j = h;
    }
    return data_[i*n_ - i*(i-1)/2 + j - i];
  }

  // TODO: Function that returns number of non-zero elements in matrix.
  unsigned NormL0() const override {
    unsigned m = 0;
    for (unsigned i = 0; i < (n_+1)*n_/2; i++){
      if (data_[i] != 0) m = m + 2;
    }
    for (unsigned i = 0; i < n_; i++){
      if (data_[i*n_ - i*(i-1)/2] != 0) m--;
    }
    return m;
  }

  // TODO: Function that modifies the ostream object so that
  // the "<<" operator can print the matrix (one row on each line).
  void Print(std::ostream& ostream) override {
    for (unsigned i = 0; i < n_; i++){
      for (unsigned j = 0; j < n_; j++){
        ostream << (*this)(i,j) << " ";
      }
      ostream << "\n";
    }
  }
};

#endif /* MATRIX_HPP */