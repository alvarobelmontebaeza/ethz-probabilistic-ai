import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


domain = np.array([[0, 5]])


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        # Initialize randomly the parameters in the domain
        self.x = np.atleast_2d([])
        self.y_f = np.atleast_2d([])
        self.y_v = np.atleast_2d([])
        ker_f = 0.5 * Matern(length_scale=0.5, nu=2.5)
        ker_v = np.sqrt(2) * Matern(length_scale=0.5, nu=2.5)
        self.gpr_f = GaussianProcessRegressor(kernel=ker_f, alpha=0.15**2)
        self.gpr_v = GaussianProcessRegressor(kernel=ker_v, alpha=1e-6)
        self.xi = 0.01
        pass


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.        
        x_opt = np.atleast_2d(self.optimize_acquisition_function())

        return x_opt

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """
        
        # TODO: enter your code here
        x = np.atleast_2d(x)
        # Use GP models to predict value at query point
        mu_f, sigma_f = self.gpr_f.predict(x, return_std=True)
        mu_v, sigma_v = self.gpr_v.predict(x, return_std=True)
        mu_v += 1.5

        # Set to zero when stddevs are zero
        if sigma_f == 0.0 or sigma_v == 0.0:
            ucb = 0.0
            return ucb

        ucb = (mu_f + 2.0 * sigma_f) * norm.cdf((mu_v - 1.2)/ sigma_v)**5
        return ucb.item()
        
        


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        f = np.atleast_2d(f)
        v = np.atleast_2d(v)
        # Add parameter and observations
        self.x = np.append(self.x, x, axis=1)
        self.y_f = np.append(self.y_f, f, axis=1)
        self.y_v = np.append(self.y_v, v-1.5, axis=1)
        # Update GP
        self.gpr_f.fit(np.transpose(self.x), np.transpose(self.y_f))
        self.gpr_v.fit(np.transpose(self.x), np.transpose(self.y_v))


        

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        y_f = self.y_f
        # Ignore points where the constraint is not satisfied
        y_f[self.y_v < (1.2 - 1.5)] = -100.0
        # Get the maximum accuracy option which respects the constraint
        idx = np.argmax(y_f)

        x_opt = self.x[0,idx]
        return x_opt


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"
            
        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
