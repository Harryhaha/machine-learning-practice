import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def iris_type(s):
    s = s.decode('utf-8')
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    path = '../data/iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    x = x[:, :2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    model = Pipeline([
        ('ss', StandardScaler()),
        ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=3))])
    model = model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)

    # Draw
    N, M = 100, 100  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.flat, x2.flat), axis=1)

    cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
    y_show_hat = model.predict(x_show)
    y_show_hat = y_show_hat.reshape(x1.shape)
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', s=100, cmap=cm_dark, marker='o')  # 测试数据
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
    plt.xlabel('sepal length', fontsize=15)
    plt.ylabel('sepal width', fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title('Iris Decision Tree classification', fontsize=17)
    plt.show()

    y_test = y_test.reshape(-1)
    print(y_test_hat)
    print(y_test)
    result = (y_test_hat == y_test)
    acc = np.mean(result)
    print('accuracy: %.2f%%' % (100 * acc))

    # Overfitting: Error rate
    depth = np.arange(1, 15)
    err_list = []
    for d in depth:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        clf = clf.fit(x_train, y_train)
        y_test_hat = clf.predict(x_test)
        result = (y_test_hat == y_test)
        err = 1 - np.mean(result)
        err_list.append(err)
        print(d, ' accuracy: %.2f%%' % (100 * err))
    plt.figure(facecolor='w')
    plt.plot(depth, err_list, 'ro-', lw=2)
    plt.xlabel('Depth of the decision tree', fontsize=15)
    plt.ylabel('Error rate', fontsize=15)
    plt.title('decision tree depth and overfitting', fontsize=17)
    plt.grid(True)
    plt.show()