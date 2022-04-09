import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# หาแพทเทินของตัวเลขสองชุด x, y
from sklearn.linear_model import LinearRegression
from sklearn import metrics
weather = pd.read_csv("file/Weather.csv")

x = weather['MinTemp'].values.reshape(-1, 1)
y = weather['MaxTemp'].values.reshape(-1, 1)

x_train, x_test = train_test_split(x, train_size=0.7, random_state=0)
y_train, y_test = train_test_split(y, train_size=0.7, random_state=0)

# Learnning model
model = LinearRegression()
model.fit(x_train, y_train)

# test model
y_pred = model.predict(x_test)

# compare true
df = pd.DataFrame({'Actually':y_test.flatten(), 'Predicted':y_pred.flatten()})

print('MAE = ', metrics.mean_absolute_error(y_test, y_pred))
print('MSE = ', metrics.mean_squared_error(y_test, y_pred))
print("RMSE = ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Score = ", metrics.r2_score(y_test, y_pred))
