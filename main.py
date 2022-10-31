import math
from itertools import combinations


# 欧式距离
def L(x, y, p=2):
    # 保证两个向量的维数一样，并且，维数大于等于1位，
    if len(x) == len(y) and len(x) >= 1:
        # 用于累加
        sum = 0
        # 循环维数次
        for i in range(len(x)):
            # abs(x[i] - y[i]) 计算绝对值后，在通过math.pow()进行p次方运算，再痛sum+=累加起来
            sum += math.pow(abs(x[i] - y[i]), p)
        # 最后的累加结果进行1/p次方
        return math.pow(sum, 1 / p)
    else:
        return 0


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# 载入花花数据集
iris = load_iris()
# 将数据转换为pandas数据，利用pandas方便操作
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#  150行，4列

# 定义标签
df['label'] = iris.target
# 前50个标签为0，中间50个标签为1，后面50个标签为2
# 重新命名属性名。 以前的名字是：'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# 下面画图
# scatter 用于画散点图 前50个label为0，后50个label为1。
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
# x,y轴的名
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
# plt.show()


# df.iloc[:100, [0, 1, -1] 取df数据的前100行，第0,1,和最后一列，最后一列也就是标签页
data = np.array(df.iloc[:100, [0, 1, -1]])
# 分离数据和标签
# data[:, :-1] 取所有行，除了最后一列，其中的-1表示最后一列，0省略了，获取除了标签的属性
# data[:, -1]  取所有行，最后一列，获取标签
X, y = data[:, :-1], data[:, -1]
# 划分训练集和测试集，测试集大小占20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class KNN:
    #  X_train, y_train训练集
    #  n_neighbors 临近点个数
    #  P距离量度
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        knn_list = []
        # 选算前n个点的距离
        for i in range(self.n):
            # dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            # 计算距离
            dist = L(X, self.X_train[i], self.p)
            # 将前n点的距离算出来，同标签，放进 knn_list
            knn_list.append((dist, self.y_train[i]))

        # 再算后面n个的
        for i in range(self.n, len(self.X_train)):
            # 找到前面n个最大值的索引 max（knn_list, key=lambda x: x[0]） key=lambda x: x[0] 表示按照距离找到最大值，
            # knn_list.index(）找到最大值对应的索引
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            # 算距离
            # dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            dist = L(X, self.X_train[i], self.p)
            # 前面最大值的大于这次算的,
            if knn_list[max_index][0] > dist:
                # 将前面最大值替换，换为这次算的，相当于把最大值顶出去
                knn_list[max_index] = (dist, self.y_train[i])
        # 上面循环完了之后，knn_list中存着距离最短的n个节点的距离与标签

        # 统计
        # 拿到所有标签
        knn = [k[-1] for k in knn_list]
        # Counter()统计标签  例如 {0：10;1:2} 表示0标签10个，1标签2个
        count_pairs = Counter(knn)
        #         max_count = sorted(count_pairs, key=lambda x: x)[-1]
        # count_pairs.items() 获取{}，key=lambda x: x[1] 表示 按照个数排序，[-1][0]中的[-1]表示最后一个，也就是个数最多的一个，[0]表示标签
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
        # 返回标签
        return max_count

    # 计算得分
    def score(self, X_test, y_test):
        # 初始化分正确的个数
        right_count = 0
        # zip()让属性和标签一起取
        for X, y in zip(X_test, y_test):
            # 预测的标签
            y_pre = self.predict(X)
            # 预测标签和真实标签一致，表示预测成功，预测个数+1；
            if y_pre == y:
                right_count += 1
        # 计算预测真确率
        return right_count / len(X_test)


# 训练
clf = KNN(X_train, y_train,4)
print(clf.score(X_test, y_test))

from sklearn.neighbors import KNeighborsClassifier
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
print('='*100)
print(clf_sk.score(X_test, y_test))
