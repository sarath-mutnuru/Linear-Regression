# Linear-Regression

The basic concept of Linear Regression is demonstrated here.

**Basis Theory of Linear Regression**:

Linear Regression involves predicting a continuous real variable using a Linear function of other variables.

y_i_est = beta* x_i + beta_0 , x_i is the ith data point, y_i_est is the estimated value for x_i , beta is the slope and beta_0 is the interecept.

If we represent the above in Matrix format,

Y_est = X* beta

X is the Nx(p+1) Matrix of i/p variables, each row being a data point and first column being all ones to include the intercept, beta is the coefficient vector and Y_est the N-vector of estimated output variables.

This is generally referred to as Multiple Linear Regression as we have more than 1 feature.

After we formulate a the equation, we need to find the best beta, to do this we define the notion of best as the one which minimizes Mean-Squared Error.

MSE = sum {(y_i-y_i_est)^2}/ N , N is number of training samples.

in matrix format , 

MSE = (Y_est-Y).T * (Y_est-Y) = (Y-Xb).T* (Y-Xb) with beta as the variable

If we minimize this on beta by equating gradient to zero and proceeding , we obtain

beta_est = (X.T* X)^-1 * X.T * Y

which can be used to predict y for a new test data point 

x_t as y* = beta.T* x_t.

**Techiniques Employed here**

However we can include squared feature terms and feature interaction terms to bring in some quadratic degree in the formula which might improve the accuracy.We are basically expanding the feature space of the input.This is sometimes done using Kernel functions for example in SVM.

Here we perform Linear Regression using the direct variables and also using combination of given features and their squares and compare the performance on the data.

The functions are all hand written in python and use no external library.

**Analysis of the performance**

If we run the code , we will observe that the MSE on the validation set using Linear Regression is 2.745 which is generally high
but after we have included the second degree terms the MSE on the validation set has gone down to 0.25 which is a quite good improvement.

The reason for this will become clear if we examine the data.

The below is the scatter plot of the data and the varaible we have estimated using linear and using squared terms.
The data has a dent in it at 4(we have only one feature) so the best line the linear regression predicts has considerable error but after we used squared terms also it tried to fit a second degree curve which minimized tge error.

![alt text](https://github.com/sarath-mutnuru/Linear-Regression/blob/master/figure.png)

This analysis proves the imortance of data examination which is the crucial part of Machine Learning task.

