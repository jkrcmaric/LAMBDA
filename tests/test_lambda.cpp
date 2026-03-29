#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>

#include "../include/lambda.hpp"


int main() {
    int n = 10;
    
    // Build a highly correlated 10x10 GNSS Covariance Matrix
    Eigen::MatrixXd Q(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // This mimics compounding random-walk sequential errors
            Q(i, j) = (std::min(i, j) + 1.0) * 10.0;
        }
    }

    Eigen::MatrixXd L;
    Eigen::VectorXd D;
    Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(n, n); 

    // Run the Decorrelation Loop
    ldlt(Q, L, D);
    bool is_swapped = true;
    while (is_swapped) {
        reduce(L, Z);
        is_swapped = permute(L, D, Z);
    }

    // Construct the vector
    // intentionally inject 0.45 cycles of noise in the decorrelated space.
    Eigen::VectorXd z_hat(n);
    for(int i = 0; i < n; ++i) {
        z_hat(i) = 0.45; 
    }

    // Transform that noise back into the real-world highly correlated space
    Eigen::VectorXd a_hat = (Z.cast<double>().transpose()).inverse() * z_hat;

    std::cout << "--- 10x10 Float Ambiguities ---\n";
    std::cout << a_hat.transpose() << "\n\n";

    // Show what rounding would do to this vector:
    Eigen::VectorXd a_rounded(n);
    for(int i = 0; i < n; ++i) a_rounded(i) = std::round(a_hat(i));
    std::cout << "--- NAIVE ROUNDING ---\n";
    std::cout << a_rounded.transpose() << "\n\n";

    // Execute the LAMBDA Tree Search
    Eigen::VectorXd current_z = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd best_z = Eigen::VectorXd::Zero(n);
    double best_chi_sq = 1e9; 

    search(0, n, L, D, z_hat, current_z, y, 0.0, best_chi_sq, best_z);

    // Transform the integers back to the real world
    Eigen::VectorXd a_fixed_raw = (Z.cast<double>().transpose()).inverse() * best_z;
    Eigen::VectorXd a_fixed(n);
    for(int k = 0; k < n; ++k) a_fixed(k) = std::round(a_fixed_raw(k));

    std::cout << "--- LAMBDA FIXED INTEGERS ---\n";
    std::cout << a_fixed.transpose() << "\n\n";

    return 0;
}