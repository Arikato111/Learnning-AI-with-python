import pandas as pd
from sklearn.preprocessing import LabelEncoder # ใช้เปลี่ยนข้อมูลตัวอักษรเป็นตัวเลข
from sklearn.model_selection import train_test_split    # แบ่งข้อมูลสำหรับ train & test
from sklearn.naive_bayes import GaussianNB  # model AI
from sklearn.metrics import accuracy_score  # ตรวจสอบความแม่นยำของผลทำนายจาก AI

# read file data
dataset = pd.read_csv("file/adult.csv")

# เปลี่ยนช้อมูลตัวอักษรเป็นตัวเลขรหัสประเภท
def cleandata(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset

# แยกข้อมูลจากไฟล์ออกมา และแบ่งประเภทเพื่อนำไป train ให้ AI
# features คือ คุณสมบัตื    label คือ ประเภทใหญ่
# ตัวอย่าง label = แมว , features = มีขน สี่ขา มีหาง ร้องเหมียว
# และจะนำข้อมูลในส้วน features ไปทำนายผล เช่น หากมีสี่ขา ร้องเหมียว ก็มีโอกาสที่จะเป็นแมว
def split_feature_class(dataset, feature):
    features = dataset.drop(feature, axis=1)
    labels = dataset[feature].copy()
    return features, labels

dataset = cleandata(dataset)    # ใช้ฟังค์ชั่น เปลี่ยนข้อมูลอักษรเป็นรหัสตัวเลข
train_features, train_labels = split_feature_class(dataset, "income")   # แบ่งข้อมูล features & label | label = "income"

# แบ่งข้อมูลสำหรับใช้ train model & test model
x_train, x_test = train_test_split(train_features, test_size=0.2)
y_train, y_test = train_test_split(train_labels, test_size=0.2)

# train model
model = GaussianNB()
model.fit(x_train, y_train)

# predict
y_predit = model.predict(x_test)    # นำข้อมูลคุณสมบัติสำหรับทดสอบลงใน model เพื่อทำนายผล
print("Accuracy = ", accuracy_score(y_predit, y_test))  # ตรวจสอบความแม่นยำ | เทียบค่าผลทำนายกับเฉลย