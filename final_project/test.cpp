#include <iostream>
#include <armadillo>

using namespace std;

int main(){
    arma::mat A(5,6);


    cout << "size of A: " << A.n_cols*A.n_rows << endl;
}