#include <eigen3/Eigen/Dense>


// Non-pivoting LDLT Decomposition
// inputs:
//      Q: float ambiguity covariance matrix (n x n)
// outputs:
//      L: lower triangular matrix (n x n) of ldlt decomposition
//      D: Vector containing the diagonal variances (n x 1) of ldlt decomposition
void ldlt(const Eigen::MatrixXd& Q, Eigen::MatrixXd& L, Eigen::VectorXd& D);

// Integer Guass Reduction
// inputs/outputs:
//      L: lower triangular matrix (n x n) from ldlt decomposition
//      Z: transformation matrix (n x n)
void reduce(Eigen::MatrixXd& L, Eigen::MatrixXd& Z);

// Permutation if Lovasz Condition not met
// inputs/outputs:
//      L: lower triangular matrix (n x n) from reduce step
//      D: diagonal variances vector (n x 1) from ldlt decomposition
//      Z: transformation matrix (n x n) from reduce step
// returns:
//      true/false: true if swap occured, false if matrix if sorted
bool permute(Eigen::MatrixXd& L, Eigen::VectorXd& D, Eigen::MatrixXd& Z);

// Recursive Sequential Conditional Tree Search
// inputs:
//   i: Current level in the tree (starts at 0)
//   n: Total number of ambiguities
//   L, D: The decorrelated covariance factors
//   z_hat: The float ambiguities transformed into Z-space
//   current_z: The integer vector we are currently building
//   y: The accumulated errors (z - z_cond) from previous levels
//   current_chi_sq: The budget consumed so far
// outputs (Passed by reference):
//   best_chi_sq: The smallest budget achieved (The "Shrinking Sphere" radius)
//   best_z: The absolute best integer vector found
void search(int i, int n, const Eigen::MatrixXd& L, const Eigen::VectorXd& D, const Eigen::VectorXd& z_hat,
            Eigen::VectorXd& current_z, Eigen::VectorXd& y, double current_chi_sq, double& best_chi_sq, 
            Eigen::VectorXd& best_z);