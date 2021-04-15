#SVM, DecisionTree, KNN을 iris dataset에 적용
#랜덤한 방식 trainset과 testset을 나눠 반복하고 그 평균을 통해 성능을 비교해보고자함

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

"""
import seaborn as sns
sns.pairplot(data=csv, hue='species')
plt.show()
"""

# 데이터 분류
data = csv_np[:, 0:4]
label = csv_np[:, 4]

iterate = 1000
scoreA = np.zeros((iterate,))
scoreB = np.zeros((iterate,))
scoreC = np.zeros((iterate,))
scoreD = np.zeros((iterate,))

from sklearn.preprocessing import StandardScaler
for i in range(iterate):
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3,
                                                                      )#random_state=7601)
    sc = StandardScaler()  # preprocessing 모듈에서 StandardScaler 클래스를 로드
    sc.fit(data_train)  # 각 특성마다 샘플평균과 표준편차를 계산
    data_train_std = sc.transform(data_train)  # data_train 값을 표준화
    data_test_std = sc.transform(data_test)  # data_test 값을 표준화
    data_train_std = data_train
    data_test_std = data_test

    # supprot vector machine(SVM)
    from sklearn.svm import SVC
    from sklearn import metrics
    svmlin = SVC(kernel='linear', C=1.0, random_state=1)
    svmlin.fit(data_train_std, label_train)
    result = svmlin.predict(data_test_std)
    score = metrics.accuracy_score(result, label_test)
    scoreA[i, ] = score
    print("SVMlin  :  ", score*100)
    svmpoly = SVC(kernel='rbf')
    svmpoly.fit(data_train_std, label_train)
    result = svmpoly.predict(data_test_std)
    score = metrics.accuracy_score(result, label_test)
    scoreB[i,] = score
    print("SVMrbf  :  ", score*100)


    #DecisionTree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=7601)
        #criterion : 불순도 결정 방식 entropy/gini
    tree.fit(data_train_std, label_train)
    result = tree.predict(data_test_std)
    score = metrics.accuracy_score(result, label_test)
    scoreC[i,] = score
    print("Tree  :  ", score*100)

    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(data_train_std, label_train)
    result = knn.predict(data_test_std)
    score = metrics.accuracy_score(result, label_test)
    print("Knn  :  ", score*100)
    scoreD[i,] = score

print(scoreA.sum()*100/iterate)
print(scoreB.sum()*100/iterate)
print(scoreC.sum()*100/iterate)
print(scoreD.sum()*100/iterate)
