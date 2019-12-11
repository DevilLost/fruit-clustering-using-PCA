import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import random


def get_data(num):  # 创建数据集(结合data_to_csv()函数可以起到刷新数据的作用)
    # 总数据集
    dataset = []
    # 填充数据集
    for i in range(num):
        # 临时数据，存放一条数据
        data_temp = []
        # 甜度
        sweet = random.randint(0, 100)
        data_temp.append(sweet)
        # 酸度
        sour = random.randint(0, 100)
        data_temp.append(sour)
        # 水分
        water = random.randint(0, 100)
        data_temp.append(water)
        # 脆度
        crisp = random.randint(0, 100)
        data_temp.append(crisp)
        # 是否喜爱(0:不喜欢;1:一般般;2:喜欢)
        if sour >= 60:
            data_temp.append(0)
        elif (sweet >= 50 and water >= 50) or (sweet >= 50 and crisp >= 50):
            data_temp.append(2)
        else:
            data_temp.append(1)
        # 填充至总数据集
        dataset.append(data_temp)
    # 将数据集转换成numpy.array
    dataset = np.array(dataset)
    return dataset


def data_to_csv(dataset):  # 将数据保存到csv文件
    df = pd.DataFrame(dataset)  # 创建DataFrame
    df.columns = ['甜度', '酸度', '水分', '脆度', '喜好']  # 修改列名
    df.to_csv('Data.csv', encoding='utf-8')  # 保存至csv
    return df


def read_data_from_csv():  # 读取数据保存为DataFrame
    df = pd.read_csv('Data.csv', encoding='utf-8', index_col=0)
    return df


def data_group(df):  # 对数据根据喜好进行分组
    grouped = df.groupby('喜好')
    dislike = grouped.get_group(0)  # 不喜欢
    just_soso = grouped.get_group(1)  # 一般般
    like = grouped.get_group(2)  # 喜欢
    return like, just_soso, dislike  # 返回分组


def data_pca(data):  # 使用PCA对特征值进行降维，维度为2
    pca = PCA(n_components=2)  # 创建对象实例，设置维度为2
    new_data = pca.fit_transform(data)  # PCA之后的新数据
    return new_data


def get_feature(data):  # 提取数据集中的特征向量
    new_data = data[:, 0: 4]
    return new_data


def get_label(data):  # 提取数据集中的标签
    labels = np.transpose(data[:, 4])
    return labels


def get_plot(df):  # 绘制气泡图
    # 设置中文字体
    myfont = matplotlib.font_manager.FontProperties(fname="simsun.ttc")
    # 对原始数据进行分类
    like, just_soso, dislike = data_group(df)
    # 作图
    fig, ax = plt.subplots(2, 2)
    # 气泡颜色字典
    color = {0: 'red', 1: 'orange', 2: 'blue'}
    # 以PCA之后的数据作x、y轴，根据喜好分类填充颜色
    # 第一个图:三分类汇总图
    data_0 = data_pca(get_feature(df.values))
    labels_0 = get_label(df.values)
    x = [data_0[i][0] for i in range(len(data_0))]
    y = [data_0[j][1] for j in range(len(data_0))]
    colors = [color[k] for k in labels_0]
    ax[0][0].scatter(x, y, color=colors, alpha=0.6)
    ax[0][0].set_title('三分类汇总气泡图', fontproperties=myfont)
    # 第二个图:喜欢和一般般
    data_tmp_1 = like.append(just_soso)
    data_1 = data_pca(get_feature(data_tmp_1.values))
    labels_1 = get_label(data_tmp_1.values)
    x = [data_1[i][0] for i in range(len(data_1))]
    y = [data_1[j][1] for j in range(len(data_1))]
    colors = [color[k] for k in labels_1]
    ax[0][1].scatter(x, y, color=colors, alpha=0.6)
    ax[0][1].set_title('喜欢和一般般分布气泡图', fontproperties=myfont)
    # 第三个图:一般般和不喜欢
    data_tmp_2 = just_soso.append(dislike)
    data_2 = data_pca(get_feature(data_tmp_2.values))
    labels_2 = get_label(data_tmp_2.values)
    x = [data_2[i][0] for i in range(len(data_2))]
    y = [data_2[j][1] for j in range(len(data_2))]
    colors = [color[k] for k in labels_2]
    ax[1][0].scatter(x, y, color=colors, alpha=0.6)
    ax[1][0].set_title('一般般和不喜欢分布气泡图', fontproperties=myfont)
    # 第四个图:不喜欢和喜欢
    data_tmp_3 = dislike.append(like)
    data_3 = data_pca(get_feature(data_tmp_3.values))
    labels_3 = get_label(data_tmp_3.values)
    x = [data_3[i][0] for i in range(len(data_3))]
    y = [data_3[j][1] for j in range(len(data_3))]
    colors = [color[k] for k in labels_3]
    ax[1][1].scatter(x, y, color=colors, alpha=0.6)
    ax[1][1].set_title('喜欢和不喜欢分布气泡图', fontproperties=myfont)
    # 展示图像
    plt.show()


if __name__ == '__main__':
    # 数据生成和导出
    data_to_csv(get_data(100))
    # 数据读入
    df = read_data_from_csv()
    # 数据分析及绘图
    get_plot(df)
    pass
