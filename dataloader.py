import csv
import torch
from torch.utils.data import Dataset, DataLoader
import conf
import numpy as np
# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, filepath, skiprows = 1, usecols = range(1,7)):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32, skiprows = skiprows, usecols = usecols)
        self.len = xy.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def getDataLoader(csv_file):
    # 创建数据集对象
    # csv_file = "data.csv"
    # batch_size = 32
    dataset = MyDataset(csv_file)
    # 创建 DataLoader 对象
    # 划分训练集合和测试集合
    train_size = int(conf.train_size_propotion * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

    # 创建训练集合 DataLoader 对象
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size)
    # 创建测试集合 DataLoader 对象
    test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size)
    return train_dataloader, test_dataloader
