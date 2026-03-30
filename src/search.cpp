#include <cmath>

#include "lambda.hpp"


void search(int i, int n, const Eigen::MatrixXd& L, const Eigen::VectorXd& D, const Eigen::VectorXd& z_hat,
            Eigen::VectorXd& current_z, Eigen::VectorXd& y, int& iter_count, int max_iter, 
            double current_chi_sq, double& best_chi_sq, Eigen::VectorXd& best_z, 
            double& second_best_chi_sq, Eigen::VectorXd& second_best_z) {

    if (iter_count >= max_iter)  return;
    iter_count++;
    
    // We reached the bottom of the tree
    if (i == n) {
        if (current_chi_sq < best_chi_sq) {

            second_best_chi_sq = best_chi_sq;
            second_best_z = best_z;

            best_chi_sq = current_chi_sq; 
            best_z = current_z;           
        } 
        else if (current_chi_sq < second_best_chi_sq) {
            second_best_chi_sq = current_chi_sq;
            second_best_z = current_z;
        }
        return; 
    }

    // Calculate the Conditional Float Estimate
    double z_cond = z_hat(i) + L.row(i).head(i).dot(y.head(i));

    // Calculate the Search Bounds using Remaining Budget
    double remaining_budget = second_best_chi_sq - current_chi_sq;
    
    // If negative budget, kill the branch
    if (remaining_budget <= 0.0) return; 

    // Find the absolute closest integer
    int center = std::round(z_cond);
    int direction = (z_cond >= center) ? -1 : 1;
    int max_steps = 100000;

    for (int step = 0; step < max_steps; ++step) {
        // Generate the alternating offset sequence
        int offset = (step % 2 == 0) ? (step / 2) * direction : -((step + 1) / 2) * direction;
        int candidate = center + offset;
            
        current_z(i) = candidate;
        y(i) = candidate - z_cond;

        double step_chi_sq = (y(i) * y(i)) / D(i);
        double new_chi_sq = current_chi_sq + step_chi_sq;

        // Kill branch if it exceeds max allowed chi squared
        if (new_chi_sq >= second_best_chi_sq ) {
            break; 
        }

        search(i + 1, n, L, D, z_hat, current_z, y, iter_count, max_iter, 
                new_chi_sq, best_chi_sq, best_z, second_best_chi_sq, second_best_z);
    }
}