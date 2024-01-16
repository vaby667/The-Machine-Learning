1.Fitting and predicting using a linear regression function:
from sklearn.linear_model import LinearRegression

# Fit linear classifier
def fit_lc(y, x):
    x = np.column_stack((np.ones(len(x)), x))
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_

# Make predictions from linear classifier
def predict_lc(x, beta):
    x = np.column_stack((np.ones(len(x)), x))
    return np.dot(x, beta)

2.Add the squared terms of x1 and x2 to the linear model:
# Fit linear classifier with squared terms
def fit_lc_squared(y, x):
    x_squared = np.column_stack((x, x**2))
    x_squared = np.column_stack((np.ones(len(x)), x_squared))
    model = LinearRegression()
    model.fit(x_squared, y)
    return model.coef_

# Make predictions from linear classifier with squared terms
def predict_lc_squared(x, beta):
    x_squared = np.column_stack((x, x**2))
    x_squared = np.column_stack((np.ones(len(x)), x_squared))
    return np.dot(x_squared, beta)


3.How a more flexible model affects the bias-variance trade-off: A more flexible model (with the square terms of x1 and x2 added) will generally reduce bias because it can better fit the nonlinear relationships in the data. However, it may also increase variance because the model is more likely to overfit the training data. As a result, a more flexible model that trades off bias and variance may perform better on training data, but may perform worse on unseen data