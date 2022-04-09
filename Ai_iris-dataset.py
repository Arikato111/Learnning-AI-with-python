from sklearn.datasets import load_iris # load datasets for train & test model AI
from sklearn.model_selection import train_test_split # แบ่ง data สำหรับ train & test
# อ่านคำอธิบายเพิ่มเติมที่ไฟล์ AI_adult_GaussianNB.py
from sklearn.naive_bayes import GaussianNB # model of AI
from sklearn.metrics import accuracy_score  # ทดสอบความแม่นยำ AI

# load datasets
iris_dataset = load_iris()

# assign attribute , target
x = iris_dataset['data']
y = iris_dataset['target']

# train , test
x_train, x_test, y_train, y_test = train_test_split(x, y)

#model
model = GaussianNB()

# train
model.fit(x_train, y_train)

# prediction
y_pred = model.predict(x_test)

# accuracy score
print("accuracy score = ", accuracy_score(y_test, y_pred))
