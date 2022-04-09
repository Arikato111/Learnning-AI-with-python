from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
# นำค่าเข้าไปเพื่อแบ่งกลุ่ม ตามแต่ละต่ำแหน่งที่ใกล้เคียง โดยนำค่าเข้าไปแค่ พิกัด x , y
from sklearn.cluster import KMeans  # โมเดล
from sklearn.metrics import accuracy_score
x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)

# new point
x_test, y_test = make_blobs(n_samples=10, centers=4, cluster_std=0.5, random_state=0)


model = KMeans(n_clusters=4)
model.fit(x)
y_predit = model.predict(x)
y_predit_new = model.predict(x_test)
center = model.cluster_centers_

plt.scatter(x[:, 0], x[:, 1], c=y_predit)
plt.scatter(x_test[:,0], x_test[:, 1], c=y_predit_new,s=120)
plt.scatter(center[0,0], center[0, 1], c='green', label="Centroid 1")
plt.scatter(center[1,0], center[1, 1], c='red', label="Centroid 2")
plt.scatter(center[2,0], center[2, 1], c='yellow', label="Centroid 3")
plt.scatter(center[3,0], center[3, 1], c='pink', label="Centroid 4")

plt.legend(frameon=True)
plt.show()
print(accuracy_score(y_predit, y))
print(y)
print(y_predit)
