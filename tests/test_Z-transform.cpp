#include <iostream>

#include "../include/lambda.hpp"


int main() {
    // A highly correlated 3x3 float ambiguity covariance matrix (Q)
    Eigen::MatrixXd Q(3, 3);
    Q <<   5.0, -10.0,  15.0,
         -10.0,  21.0, -32.0,
          15.0, -32.0,  49.5;


    int n = Q.rows();

    // Initialize LAMBDA variables
    Eigen::MatrixXd L;
    Eigen::VectorXd D;
    Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(n, n); // Z starts as the Identity

    std::cout << "--- Initial Float Covariance (Q) ---\n" << Q << "\n\n";

    // Initial Factorization
    ldlt(Q, L, D);

    // The Core LAMBDA Decorrelation Loop
    bool is_swapped = true;
    while (is_swapped) {
        reduce(L, Z);
        is_swapped = permute(L, D, Z);
    }

    // Validate the Results
    std::cout << "--- Final Z-Transformation Matrix ---\n" << Z << "\n\n";
    std::cout << "Determinant of Z: " << Z.determinant() << "\n\n";

    // Apply Z to the original Q to see the decorrelated matrix
    Eigen::MatrixXd Q_decorrelated = Z.transpose() * Q * Z;
    
    std::cout << "--- Decorrelated Covariance (Z^T * Q * Z) ---\n" << Q_decorrelated << "\n";

    return 0;
}