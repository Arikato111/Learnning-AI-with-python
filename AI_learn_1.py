from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_data = load_iris()

x_train, x_test = train_test_split(iris_data['data'], train_size=.75, random_state=0)
y_train, y_test = train_test_split(iris_data['target'], train_size=.75, random_state=0)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)