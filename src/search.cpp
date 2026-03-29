#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "../include/lambda.hpp"


void search(int i, int n, const Eigen::MatrixXd& L, const Eigen::VectorXd& D, const Eigen::VectorXd& z_hat,
            Eigen::VectorXd& current_z, Eigen::VectorXd& y, double current_chi_sq, double& best_chi_sq, 
            Eigen::VectorXd& best_z) {
    
    // We reached the bottom of the tree! (A full vector)
    if (i == n) {
        // If this full vector consumed less budget than our previous best...
        if (current_chi_sq < best_chi_sq) {
            best_chi_sq = current_chi_sq; // SHRINK THE SPHERE!
            best_z = current_z;           // Lock in the new best vector
        }
        return; // Backtrack up one level
    }

    // Calculate the Conditional Float Estimate (z_cond)
    double z_cond = z_hat(i);
    for (int j = 0; j < i; ++j) {
        z_cond += L(i, j) * y(j); 
    }

    // Calculate the Search Bounds using Remaining Budget
    double remaining_budget = best_chi_sq - current_chi_sq;
    
    // If negative budget, kill the branch
    if (remaining_budget <= 0.0) return; 

    double bound = std::sqrt(remaining_budget * D(i));
    int left_bound = std::ceil(z_cond - bound);
    int right_bound = std::floor(z_cond + bound);

    // Alternating Search (Sorting the Candidates)
    std::vector<int> candidates;
    for (int c = left_bound; c <= right_bound; ++c) {
        candidates.push_back(c);
    }
    
    // Sort candidates based on distance to the conditional float estimate
    std::sort(candidates.begin(), candidates.end(), [z_cond](int a, int b) {
        return std::abs(a - z_cond) < std::abs(b - z_cond);
    });

    // Traverse the Branches
    for (int candidate : candidates) {
        current_z(i) = candidate;
        y(i) = candidate - z_cond;

        double step_chi_sq = (y(i) * y(i)) / D(i);
        double new_chi_sq = current_chi_sq + step_chi_sq;

        // Only dive deeper if this branch is strictly within the shrinking sphere limit
        if (new_chi_sq < best_chi_sq) {
            search(i + 1, n, L, D, z_hat, current_z, y, new_chi_sq, best_chi_sq, best_z);
        }
    }
}