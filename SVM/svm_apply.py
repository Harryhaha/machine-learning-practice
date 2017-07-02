import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt


def iris_type(s):
    s = s.decode('utf-8')
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(tip + 'accuracy：', np.mean(acc))



if __name__ == "__main__":
    path = '../data/iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    # here only choose sepal length and width as the features for visualization
    x = x[:, :2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    # accuracy
    print(clf.score(x_train, y_train))
    y_predict = clf.predict(x_train)
    show_accuracy(y_predict, y_train, 'training set')
    print(clf.score(x_test, y_test))
    y_predict = clf.predict(x_test)
    show_accuracy(y_predict, y_test, 'testing set')

    # draw
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # first column range
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # second column range
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j] # grid sampling point
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # testing points

    Z = clf.decision_function(grid_test)    # distance from sample to decision boundary
    print(Z)
    grid_hat = clf.predict(grid_test)       # predict
    print(grid_hat)
    grid_hat = grid_hat.reshape(x1.shape)
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)

    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)
    plt.xlabel('sepal length', fontsize=13)
    plt.ylabel('sepal width', fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('Iris SVM classification', fontsize=15)
    plt.grid()
    plt.show()
