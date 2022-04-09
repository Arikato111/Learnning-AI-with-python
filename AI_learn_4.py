from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# อ้างอิงจากขอบเขตใกล้เคียง โดยต้องกำหนดขอบเขต n_neighbors = _ ขอบเขต โดยกำหนดเป็นตัวเลข
from sklearn.neighbors import KNeighborsClassifier  # โมเดล
from sklearn.metrics import classification_report, accuracy_score
iris_dataset = load_iris()

x_train, x_test = train_test_split(iris_dataset['data'], test_size=0.4, random_state=0)
y_train, y_test = train_test_split(iris_dataset['target'], test_size=0.4, random_state=0)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(classification_report(y_test, y_pred, target_names=iris_dataset['target_names']))
print("ความแม่นยำ", accuracy_score(y_test, y_pred)*100)