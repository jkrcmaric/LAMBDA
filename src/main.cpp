#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

#include "lambda.hpp"


// LAMBDA
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <data.csv>\n";
        return 1;
    }

    std::string filename = argv[1];
    int max_iter = 10000000;

    // Parse the CSV File
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return 1;
    }

    std::string line;
    std::vector<double> a_vec;
    
    // Read first line (Float Ambiguities)
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ',')) {
            a_vec.push_back(std::stod(val));
        }
    }

    int n = a_vec.size();
    Eigen::VectorXd a_hat = Eigen::Map<Eigen::VectorXd>(a_vec.data(), n);
    Eigen::MatrixXd Q(n, n);

    // Read remaining lines (Covariance Matrix)
    int row = 0;
    while (std::getline(file, line) && row < n) {
        std::stringstream ss(line);
        std::string val;
        int col = 0;
        while (std::getline(ss, val, ',') && col < n) {
            Q(row, col) = std::stod(val);
            col++;
        }
        row++;
    }
    file.close();

    std::cout << "Loaded " << n << " ambiguities from " << filename << ".\n";

    // Run LAMBDA Decorrelation
    Eigen::MatrixXd L;
    Eigen::VectorXd D;
    Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(n, n); 

    ldlt(Q, L, D);
    bool is_swapped = true;
    while (is_swapped) {
        reduce(L, Z);
        is_swapped = permute(L, D, Z);
    }

    // Run Sequential Tree Search
    Eigen::VectorXd z_hat = Z.cast<double>().transpose() * a_hat;
    Eigen::VectorXd current_z = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd best_z = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd second_best_z = Eigen::VectorXd::Zero(n);
    
    double best_chi_sq = 1e9; 
    double second_best_chi_sq = 1e9; 
    int iter_count = 0;

    search(0, n, L, D, z_hat, current_z, y, iter_count, max_iter, 
           0.0, best_chi_sq, best_z, second_best_chi_sq, second_best_z);

    // Output and Ratio Test
    std::cout << "\n--- SEARCH RESULTS ---\n";
    std::cout << "Iterations Consumed: " << iter_count << " / " << max_iter << "\n";
    
    if (iter_count > max_iter) {
        std::cout << "\nWARNING: Search aborted! Max iterations exceeded. Matrix is too noisy to fix.\n";
        return 1;
    }

    double ratio = second_best_chi_sq / best_chi_sq;
    std::cout << "Best Chi-Sq: " << best_chi_sq << "\n";
    std::cout << "Second Best Chi-Sq: " << second_best_chi_sq << "\n";
    std::cout << "Ratio Test Result: " << ratio << "\n\n";

    if (ratio >= 3.0) {
        Eigen::VectorXd a_fixed_raw = Z.cast<double>().transpose().colPivHouseholderQr().solve(best_z);
        Eigen::VectorXd a_fixed(n);
        for(int k = 0; k < n; ++k) a_fixed(k) = std::round(a_fixed_raw(k));

        std::cout << "SUCCESS! Fixed Integers:\n" << a_fixed.transpose() << "\n";
    } else {
        std::cout << "REJECTED: Ratio too low (< 3.0). Keep using Float solution.\n";
    }

    return 0;
}