import random

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
# แบ่งกลุ่ม ทายผลว่ารูปนี้ใช่ ... หรีอไม่
# เช่น ในโปรแกรมนี้
# ทดสอบว่า ภาพที่นำเข้าไปใช่เลข 0 หรีอไม่
# โดยที่ ต้องนำ ภาพเลข 0 และภาพอื่นๆ ใส่เข้าไปใน model พร้อมผลเฉลยเป็นค่า true & false
# นำภาพเข้าไปทำนายผล แล้ว model จะทายผลออกมาเป็นค่า true & fales เพื่อบอกว่าภาพที่นำเข้าไปใช่เลข 0 หรือไม่
from sklearn.linear_model import SGDClassifier  # โมเดล
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
def displayConfusionMatrix(cm,cmap=plt.cm.GnBu):
    classes=["Other Number","Number 5"]
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    trick_marks=np.arange(len(classes))
    plt.xticks(trick_marks,classes)
    plt.yticks(trick_marks,classes)
    thresh=cm.max()/2
    for i , j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],'d'),
        horizontalalignment='center',
        color='white' if cm[i,j]>thresh else 'black')

    plt.tight_layout()
    plt.ylabel('Actually')
    plt.xlabel('Prediction')
    plt.show()

def displayImage(x):
    plt.imshow(x.reshape(28, 28),
               cmap='binary',
               interpolation='nearest'
               )
    plt.show()

def displayPredict(clf, actually_y, x):
    print("Actually = ", actually_y)
    print("Prediction = ", clf.predict([x])[0])

mnist_raw = loadmat("file/mnist-original.mat")
mnist = {
    "data":mnist_raw['data'].T,
    "target":mnist_raw["label"][0]
}

# training & test
x_train, x_test = mnist['data'][:60000], mnist['data'][60000:]
y_train, y_test = mnist['target'][:60000], mnist['target'][60000:]

predict = 5000
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)

# model for AI
sgd_clf = SGDClassifier()
sgd_clf.fit(x_train, y_train_0)

y_train_pred = cross_val_predict(sgd_clf, x_train,y_train_0, cv=3)
cm = confusion_matrix(y_train_0, y_train_pred)

classes = ['Other Number', 'Number 0']
y_test_pred = sgd_clf.predict(x_test)
print(classification_report(y_test_0, y_test_pred, target_names=classes))

print("Accuracy Score = ", accuracy_score(y_test_0, y_test_pred) * 100)
