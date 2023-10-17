import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from utils import  data_preprocessing
 
def data_preparation(num_feature, rho, num_row):
    c = 0.3
    u1 = np.random.randn(num_feature)
    u1 = (u1 - np.mean(u1)) / (np.std(u1) * np.sqrt(num_feature))
    u2 = np.random.randn(num_feature)
    u2 -= u2.dot(u1) * u1
    u2 /= np.linalg.norm(u2)

    # k = np.random.randn(num_feature)
    # u1 = np.random.randn(num_feature) 
    # u1 -= u1.dot(k) * k / np.linalg.norm(k)**2
    # u1 /= np.linalg.norm(u1)
    # k /= np.linalg.norm(k)
    # u2 = k
    w1 = c * u1
    w2 = c * (rho * u1 + np.sqrt((1 - rho**2))*u2)
    X = np.random.normal(0, 1, (num_row, num_feature))
    eps1 = np.random.normal(0, 0.01)
    eps2 = np.random.normal(0, 0.01)
    Y1 = np.matmul(X, w1) + np.sin(np.matmul(X, w1))+eps1
    Y2 = np.matmul(X, w2) + np.sin(np.matmul(X, w2))+eps2
    X, target_df1, target_df2 = data_preprocessing(input_data=X, target_label1=Y1, target_label2=Y2)
    Y = np.concatenate((target_df1, target_df2), axis=1)
    return X, Y


X, Y = data_preparation(num_feature=5, rho=0.3, num_row=1000)
f = plt.figure(figsize=(15, 5))
f.add_subplot(1,2,1)
plt.title("Xs input data")
plt.plot(X)
plt.xlabel("Samples")
f.add_subplot(1,2,2)
plt.title("Ys output data")
plt.plot(Y)
plt.xlabel("Samples")
plt.show()
 
xtrain, xtest, ytrain, ytest=train_test_split(X, Y, test_size=0.20)
print("xtrain:", xtrain.shape, "ytrian:", ytrain.shape)
print("xtest:", xtest.shape, "ytest:", ytest.shape)



gbr = xgb.XGBRegressor()
model = MultiOutputRegressor(estimator=gbr)
print(model)

model.fit(xtrain, ytrain)
score = model.score(xtrain, ytrain)
print("Training score:", score)

ypred = model.predict(xtest)

print("y1 MSE:%.4f" % mean_squared_error(ytest[:,0], ypred[:,0]))
print("y2 MSE:%.4f" % mean_squared_error(ytest[:,1], ypred[:,1]))
print("y1 RMSE:%.4f" % np.sqrt(mean_squared_error(ytest[:,0], ypred[:,0])))
print("y2 RMSE:%.4f" % np.sqrt(mean_squared_error(ytest[:,1], ypred[:,1])))


x_ax = range(len(xtest))
fig = plt.figure(figsize=(15, 5))
plt.plot(x_ax, ytest[:,0], label="y1-test", color='c')
plt.plot(x_ax, ypred[:,0], label="y1-pred", color='b')
plt.plot(x_ax, ytest[:,1], label="y2-test", color='m')
plt.plot(x_ax, ypred[:,1], label="y2-pred", color='r')
plt.legend()
plt.show() 