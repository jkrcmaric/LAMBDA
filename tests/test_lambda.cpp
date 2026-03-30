#include <iostream>
#include <cmath>

#include "lambda.hpp"


int main() {
    int n = 10;
    double noise = 0.45;
    
    // Build a highly correlated GNSS Covariance Matrix
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
    // intentionally inject cycle noise in the decorrelated space.
    Eigen::VectorXd z_hat(n);
    for(int i = 0; i < n; ++i) {
        z_hat(i) = noise; 
    }

    // Transform that noise back into the real-world highly correlated space
    Eigen::VectorXd a_hat = (Z.cast<double>().transpose()).inverse() * z_hat;

    std::cout << "--- Float Ambiguities ---\n";
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
    Eigen::VectorXd second_best_z = Eigen::VectorXd::Zero(n);
    
    double best_chi_sq = 1e9; 
    double second_best_chi_sq = 1e9; 

    int max_iter = 10000000;
    int iter_count = 0;
    // Execute the search
    search(0, n, L, D, z_hat, current_z, y, iter_count, max_iter,
           0.0, best_chi_sq, best_z, second_best_chi_sq, second_best_z);

    if (iter_count >= max_iter) {
        std::cout << "Max Iterations Reached! Could not fix the ambiguities.\n";
        return 0;
    }

    // Calculate the Critical Ratio
    double ratio = second_best_chi_sq / best_chi_sq;

    std::cout << "--- LAMBDA RATIO TEST ---\n";
    std::cout << "Best Chi-Sq: " << best_chi_sq << "\n";
    std::cout << "Second Best Chi-Sq: " << second_best_chi_sq << "\n";
    std::cout << "Ratio: " << ratio << "\n\n";

    if (ratio >= 3.0) {
        std::cout << "SUCCESS: Ambiguities fixed with high confidence! (Ratio >= 3.0)\n\n";
    } else {
        std::cout << "REJECTED: Ratio too low. Keep using float solution.\n\n";
    }

    // Transform the integers back to the real world
    Eigen::VectorXd a_fixed_raw = Z.cast<double>().transpose().colPivHouseholderQr().solve(best_z);
    Eigen::VectorXd a_fixed(n);
    for(int k = 0; k < n; ++k) a_fixed(k) = std::round(a_fixed_raw(k));

    std::cout << "--- LAMBDA FIXED INTEGERS ---\n";
    std::cout << a_fixed.transpose() << "\n\n";

    return 0;
}