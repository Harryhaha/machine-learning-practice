import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier


def iris_type(s):
    s = s.decode('utf-8')
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'


if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    path = '../data/iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x_prime, y = np.split(data, (4,), axis=1)

    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    plt.figure(figsize=(10, 9), facecolor='#FFFFFF')
    for i, pair in enumerate(feature_pairs):
        x = x_prime[:, pair]

        # Random forest
        clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=4)
        rf_clf = clf.fit(x, y.ravel())

        N, M = 500, 500
        x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
        x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)
        x_test = np.stack((x1.flat, x2.flat), axis=1)

        y_hat = rf_clf.predict(x)
        y = y.reshape(-1)
        c = np.count_nonzero(y_hat == y)
        print('feature：  ', iris_feature[pair[0]], ' + ', iris_feature[pair[1]],)
        print('\tthe num of correct predictions：', c,)
        print('\taccuracy rate: %.2f%%' % (100 * float(c) / float(len(y))))

        cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
        y_hat = rf_clf.predict(x_test)
        y_hat = y_hat.reshape(x1.shape)
        plt.subplot(2, 3, i+1)
        plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)
        plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=cm_dark)
        plt.xlabel(iris_feature[pair[0]], fontsize=14)
        plt.ylabel(iris_feature[pair[1]], fontsize=14)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid()
    plt.tight_layout(2.5)
    plt.subplots_adjust(top=0.92)
    plt.suptitle(u'result', fontsize=18)
    plt.show()
