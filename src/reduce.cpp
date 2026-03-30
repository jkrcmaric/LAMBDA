#include <cmath>

#include "lambda.hpp"


void reduce(Eigen::MatrixXd& L, Eigen::MatrixXd& Z) {
    int n = L.rows();

    // sweep through matrix bottom-top right-left
    for (int i = n - 1; i > 0; --i) {
        for (int j = i - 1; j >= 0; --j) {

            // nearest integer to element
            double mu = std::round(L(i, j));

            // only perform if element is outside [-0.5, 0.5] (rounds to 0.0)
            if (mu != 0.0) {

                // update L
                L.row(i).head(j + 1) -= mu * L.row(j).head(j + 1);

                // update Z
                Z.col(i) -= mu * Z.col(j);
            }
        }
    }
}