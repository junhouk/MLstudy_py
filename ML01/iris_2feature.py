#SVM, DecisionTree, KNN을 iris dataset에 적용
#iris dataset에서 2개의 feature만 추출하여 적용

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# 시각화툴
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.01):
    # 마커와 컬러맵 설정하기
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정경계를 그리기
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # 테스트 샘플을 강조
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')


csv = pd.read_csv('./data/iris.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
csv_np = csv.to_numpy(copy=True)

# 데이터 분류
X = csv_np[:, [2, 3]]#extract two feature
y = csv_np[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7601, stratify=y)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  # preprocessing 모듈에서 StandardScaler 클래스를 로드
sc.fit(X_train)  # 각 특성마다 샘플평균과 표준편차를 계산
X_train_std = sc.transform(X_train)  # X_train 값을 표준화
X_test_std = sc.transform(X_test)  # X_test 값을 표준화

X_train_std = X_train
X_test_std = X_test

X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# supprot vector machine(SVM)
from sklearn.svm import SVC
from sklearn import metrics
svm = SVC(kernel='linear', C=1.0, random_state=1)
#svm = SVC(kernel='poly')
svm.fit(X_train_std, y_train)
result = svm.predict(X_test_std)
score = metrics.accuracy_score(result, y_test)
print("SVM  :  ", score*100)
# plot
plot_decision_regions(X_combined, y_combined, classifier=svm, test_idx = range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#DecisionTree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=7601)
    #criterion : 불순도 결정 방식 entropy/gini
tree.fit(X_train, y_train)
result = tree.predict(X_test)
score = metrics.accuracy_score(result, y_test)
print("Tree  :  ", score*100)
# plot
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx = range(105,150))
plt.xlabel('sepal length [cm]')
plt.ylabel('sepal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="C:/dataset/iris_tree.dot",
                feature_names=['sepal length', 'sepal width'],
                class_names=['1', '2', '3'],
                rounded=True, filled=True)
import pydot
(graph, ) = pydot.graph_from_dot_file('C:/dataset/iris_tree.dot', encoding='utf8')
graph.write_png('C:/dataset/iris_tree.png')

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
result = knn.predict(X_test)
print(result)
score = metrics.accuracy_score(result, y_test)
print("Knn  :  ", score*100)
# plot
plot_decision_regions(X_combined, y_combined, classifier=knn, test_idx = range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
