#include <utility>

#include "../include/lambda.hpp"


bool permute(Eigen::MatrixXd& L, Eigen::VectorXd& D, Eigen::MatrixXd& Z) {
    int n = L.rows();

    // sweep from bottom-top
    for (int i = n - 1; i > 0; --i) {

        double d1 = D(i - 1);
        double d2 = D(i);
        double l_old = L(i, i -1);

        // Lovasz Condition - calculate new upper variance
        double delta = d2 + (l_old * l_old) * d1;

        // check if new upper variance is smaller than old upper variance
        if (d1 > delta) {

            // Swap in Z matrix
            Z.col(i).swap(Z.col(i - 1));

            // Update diagonal variances
            D(i - 1) = delta;
            D(i) = (d1 * d2) / delta;

            // Update L matrix
            L(i, i - 1) = (d1 * l_old) / delta;
            double l_new = L(i, i - 1);

            for (size_t j = 0; j < i - 1; ++j) {
                double l_row_i = L(i, j);
                L(i, j) = L(i - 1, j) - l_old * l_row_i;
                L(i - 1, j) =  l_row_i + l_new * L(i, j);
            }

            for (size_t j = i + 1; j < n; ++j) {
                std::swap(L(j, i), L(j, i - 1));
            }

            // return imediately if there was a swap
            return true;
        }
    }

    return false;
}