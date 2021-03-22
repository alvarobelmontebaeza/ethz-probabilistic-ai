import numpy as np
import scipy as sp

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, ConstantKernel, Matern, RationalQuadratic, DotProduct, WhiteKernel



## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model():

    def __init__(self):
        # Define the kernel
        # self.kernel = RBF(length_scale=0.1) + DotProduct() + WhiteKernel()
        # self.kernel = RationalQuadratic() 
        # self.kernel = DotProduct() + WhiteKernel() 
        # self.kernel = ExpSineSquared(1.0,5.0) + WhiteKernel(0.1)
        self.kernel = Matern(nu=0.5)

        # Create the GPRegressor object 
        self.gp = GaussianProcessRegressor(kernel=self.kernel, random_state=6,n_restarts_optimizer=0)


    def predict(self, test_x):

        # Use the trained model in order to make predictions
        predictions, std_dev = self.gp.predict(test_x, return_std=True)
        
        # Add the std_dev to the mean prediction. This is done so that the worst
        # pollution value is always considered in order to prevent false negatives
        return predictions + std_dev

    def fit_model(self, train_x, train_y):

        # Since the data is triplicated, eliminate the duplicates in data points
        # dimensionality reduction
        reduced_x = train_x[0:5750][:]
        reduced_y = train_y[0:5750]
        # Keep the highest measure in order to be more restrictive and avoid predicting
        # lower values than the real
        for i in range(5750):
            reduced_y[i] = np.max([train_y[i],train_y[5750+i],train_y[11500+i]])

        # Train the GP with the opimized data
        self.gp.fit(reduced_x,reduced_y)

def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)


if __name__ == "__main__":
    main()
