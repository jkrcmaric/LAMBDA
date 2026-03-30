#include "lambda.hpp"


void ldlt(const Eigen::MatrixXd& Q, Eigen::MatrixXd& L, Eigen::VectorXd& D) {
    int n = Q.rows();

    // Initialize L and D
    L = Eigen::MatrixXd::Identity(n, n);
    D = Eigen::VectorXd::Zero(n);

    // iterate through columns
    for (size_t j = 0; j < n; ++j) {
        
        // Calculate D(j)
        D(j) = Q(j, j);
        for (size_t k = 0; k < j; ++k) {
            D(j) -= L(j, k) * L(j, k) * D(k);
        }

        // Calculate L(i,j)
        for (size_t i = j+1; i < n; ++i) {
            if (D(j)  == 0.0) {
                L(i, j) = 0.0;
                continue;
            }
            L(i, j) = Q(i, j);
            for (size_t k = 0; k < j; ++k) {
                L(i, j) -= L(i, k) * L(j, k) * D(k);
            }
            L(i, j) /= D(j);
        }
    }
}