import numpy as np
import matplotlib.image as mpimg
import glob
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def get_label_from_path(path):
    return int(path.split('\\')[-2])
def read_image(path):
    image = mpimg.imread(path)
    return image.reshape(784)
def onehot_encode_label(path):
    onehot_label = get_label_from_path(path)
    return onehot_label

path = './data/mnist_png/training\\*\\*.png'
data_list = glob.glob(path)
label_name_list = []
for path in data_list:
    label_name_list.append(get_label_from_path(path))
unique_label_name = np.unique(label_name_list)
print(unique_label_name)


#Hyper parameter
vec_size = 784
num_classes = 1
data = np.zeros((len(data_list), vec_size))
label = np.zeros((len(data_list), ))
n = 0
for path in data_list:
    image = read_image(path)
    onehot_label = onehot_encode_label(path)
    data[n, :] = image
    label[n, ] = onehot_label
    n = n + 1
    print(n)
	
#Classifier
classifier = 'svm'

if classifier == 'svm':
	svm = SVC(kernel='poly')
	svm.fit(data, label)
elif classifier == 'knn':
	knn = KNeighborsClassifier(n_neighbors=3)
	knn.fit(data, label)

#print(data.shape)
#print(label.shape)
#print("end")indent
path2 = './data/mnist_png/testing\\3\\*.png'
test_list = glob.glob(path2)
#print(len(test_list))
test_size = 10
vec_size = 784
test_image = np.zeros((test_size, vec_size))
test_label = np.zeros((test_size, ))
#print(test_image.shape)
for test_n in range(10):
    test_data = test_list[test_n * test_size: (test_n + 1)*test_size]
    m = 0
    for path in test_data:
        image = read_image(path)
        onehot_label = onehot_encode_label(path)
        test_image[m, :] = image
        test_label[m, ] = onehot_label
        m = m + 1
    if classifier == 'svm':
        result = svm.predict(test_image)
    elif classifier =='knn':
        result = knn.predict(test_image)
    print(result)
    score = metrics.accuracy_score(result, test_label)
    print("Accuracy score  :  ", score*100)