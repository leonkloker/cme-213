#ifndef MATRIX_RECT
#define MATRIX_RECT

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

template <typename T>
class Matrix2D;

template <typename T>
bool Broadcastable(Matrix2D<T>& A, Matrix2D<T>& B) {
  if ((A.size_rows() == B.size_rows() || A.size_rows() == 1 || B.size_rows() == 1)
    && (A.size_cols() == B.size_cols() || A.size_cols() == 1 || B.size_cols() == 1)) 
    return true;
  else return false;
}

template <typename T>
class Matrix2D {
 private:
  // The size of the matrix is (n_rows, n_cols)
  unsigned int n_rows;
  unsigned int n_cols;

  // Vector storing the data in row major order. Element (i,j) for 0 <= i <
  // n_rows and 0 <= j < n_cols is stored at data[i * n_cols + j].
  std::vector<T> data_;

 public:
  // Empty matrix
  Matrix2D() { 
    n_rows = 0;
    n_cols = 0;
  }

  // Constructor takes argument (m,n) = matrix dimension.
  Matrix2D(const int m, const int n) {
    n_rows = m;
    n_cols = n;
    data_.resize(n*m);
  }

  unsigned int size_rows() const { return n_rows; }
  unsigned int size_cols() const { return n_cols; }

  // Returns reference to matrix element (i, j).
  T& operator()(int i, int j) {
    return data_[i * n_cols + j];
  }
    
  void Print(std::ostream& ostream) {
    for (unsigned i = 0; i < n_rows; i++){
      for (unsigned j = 0; j < n_cols; j++){
        ostream << data_[i * n_cols + j] << " ";
      }
      ostream << "\n";
    }
  }

  Matrix2D<T> dot(Matrix2D<T>& mat) {
    if (Broadcastable<T>(*this, mat)) {
      unsigned n_rows_b = mat.size_rows();
      unsigned n_cols_b = mat.size_cols();
      unsigned rows = 0;
      unsigned cols = 0;

      if (n_rows > n_rows_b) rows = n_rows;
      else rows = n_rows_b;
      if (n_cols > n_cols_b) cols = n_cols;
      else cols = n_cols_b;
      
      Matrix2D<T> ret(rows, cols);

      for (unsigned i = 0; i < rows; i++) {
        for (unsigned j = 0; j < cols; j++) {
          ret(i,j) = (*this)(i%n_rows,j%n_cols) * mat(i%n_rows_b,j%n_cols_b);
        }
      }
      return ret;

    } else {
      throw std::invalid_argument("Incompatible shapes of the two matrices.");
    }
  }

  template <typename U>
  friend std::ostream& operator<<(std::ostream& stream, Matrix2D<U>& m);
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, Matrix2D<T>& m) {
  m.Print(stream);
  return stream;
}

#endif /* MATRIX_RECT */
