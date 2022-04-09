from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sb

from sklearn.decomposition import PCA
# PCA คือ ทำให้ข้อมูลมีขนาดเล็กลง แต่สามารถนำไปเทรนได้ผลลัพธ์เหมือนเดิม
# load datasets
iris = sb.load_dataset('iris')

x = iris.drop('species', axis=1)
y = iris['species']

pca = PCA(n_components=3)
x_pca = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_pca, y)

model = GaussianNB()
model.fit(x_train, y_train)
y_predit = model.predict(x_test)

print(accuracy_score(y_predit, y_test))