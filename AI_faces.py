from sklearn.datasets import fetch_lfw_people   # datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC #model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sb

# # download & display image
face = fetch_lfw_people(min_faces_per_person=60)

# print(face.target_names)
# print(face.images.shape)

# fig, ax = plt.subplots(3, 5)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(face.images[i], cmap='bone')
#     axi.set(xticks=[], yticks=[])
#     axi.set_xlabel(face.target_names[face.target[i]].split()[-1], color='black')
# plt.show()

# reduce & create model
pca = PCA(n_components=150, svd_solver='randomized', whiten=True)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# train , test data
x_train, x_test, y_train, y_test = train_test_split(face.data, face.target, random_state=40)

param = {"svc__C":[1, 5, 10, 50], "svc__gamma":[0.0001, 0.0005, 0.001, 0.005]}
# train data to model
grid = GridSearchCV(model, param)
grid.fit(x_train, y_train)

print(grid.best_params_)

model = grid.best_estimator_

# predict
y_predict = model.predict(x_test)

# show real image & predict name
# fig, ax = plt.subplots(4, 6)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(x_test[i].reshape(62, 47), cmap='bone')
#     axi.set(xticks=[], yticks=[])
#     axi.set_xlabel(face.target_names[y_predict[i]].split()[-1],
#                    color='green' if y_predict[i] == y_test[i] else 'red')
# plt.show()

# print("accuracy = ", accuracy_score(y_test, y_predict))
mat = confusion_matrix(y_test, y_predict)
sb.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
           xticklabels=face.target_names,
           yticklabels = face.target_names
           )
plt.xlabel("True Data")
plt.ylabel("Predict Data")
plt.show()