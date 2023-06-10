

# start to train model by linear regression, xgboost, random forest, and neural network
# split the data into train and test
import pandas as pd
from sklearn.model_selection import train_test_split

dat = pd.read_pickle("../../data/external/external.pkl")

# split the data into train and test
X = dat.drop('fare_amount', axis=1)
y = dat['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# use the linear regression model to train the data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Linear Regression RMSE: ", mean_squared_error(y_test, y_pred, squared=False))

# print the coefficients
print("Linear Regression Coefficients: ", lr.coef_)

# print the intercept
print("Linear Regression Intercept: ", lr.intercept_)

# print the r squared
print("Linear Regression R Squared: ", lr.score(X_test, y_test))

# use the random forest model to train the data
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=123)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest RMSE: ", mean_squared_error(y_test, y_pred, squared=False))

# print the importance of each feature
print("Random Forest Feature Importance: ", rf.feature_importances_)

# predict accuracy
print("Random Forest Accuracy: ", rf.score(X_test, y_test))


# use the xgboost model to train the data
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Training RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Training R2 Score: {train_r2}")
    print(f"Test R2 Score: {test_r2}")
    
    return y_pred_test, test_rmse, test_r2, model.feature_importances_

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=123)

y_pred, rmse, r2, feature_importances = train_and_evaluate(xgb_model, X_train, y_train, X_test, y_test)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances(importances, feature_names):
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

# Assuming your feature names are stored in a list named "features"
plot_feature_importances(feature_importances, X_train.columns)


