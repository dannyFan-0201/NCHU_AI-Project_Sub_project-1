from sklearn.metrics import confusion_matrix  # 生成混淆矩阵的函数
from matplotlib import pyplot as plt
import numpy as np
import os
'''
首先是从结果文件中读取预测标签与真实标签，然后将读取的标签信息传入python内置的混淆矩阵矩阵函数confusion_matrix(真实标签,
预测标签)中计算得到混淆矩阵，之后调用自己实现的混淆矩阵可视化函数plot_confusion_matrix()即可实现可视化。
三个参数分别是混淆矩阵归一化值，总的类别标签集合，可是化图的标题
'''


def plot_confusion_matrix(cm, labels_name, title):
    np.set_printoptions(precision=2)
    # print(cm)
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title('cow_resnet50')  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # show confusion matrix
    plt.savefig('./fig/' + title + '.png', format='png')


gt = []
pre = []
with open("result.txt", "r") as f:
    for line in f:
        line = line.rstrip()  # rstrip() 删除 string 字符串末尾的指定字符（默认为空格）
        words = line.split()
        pre.append(int(words[0]))
        gt.append(int(words[1]))

cm = confusion_matrix(gt, pre)  # 计算混淆矩阵
print('type=', type(cm))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]# 类别集合
os.makedirs('./fig/', exist_ok=True)
plot_confusion_matrix(cm, labels, 'confusion_matrix')  # 绘制混淆矩阵图，可视化
