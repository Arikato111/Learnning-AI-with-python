from sklearn.model_selection import train_test_split
# อ้างอิงจากขอบเขตใกล้เคียง โดยต้องกำหนดขอบเขต n_neighbors = _ ขอบเขต โดยกำหนดเป็นตัวเลข
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("file/diabetes.csv")

x = df.drop("Outcome", axis=1).values   # values คือการเอาแค่ค่าข้อมูลเพียงอย่างเดียว
y = df['Outcome'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)

# prediction
y_pred = knn.predict(x_test)

print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=["Prediction"], margins=True))
