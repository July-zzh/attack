import torch
import torch.utils.data as data
import os


class MyDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root  # 数据集的根目录
        self.transform = transform  # 可选的数据变换
        self.classes = os.listdir(root)  # 获取所有的文件夹名，作为类别名
        self.data = []  # 空列表，用于存放数据和标签
        for i, cls in enumerate(self.classes):  # 遍历每个文件夹
            cls_path = os.path.join(root, cls)  # 获取文件夹的路径
            for file_name in os.listdir(cls_path):  # 遍历每个文件
                file_path = os.path.join(cls_path, file_name)  # 获取文件的路径
                tensor = torch.load(file_path)  # 加载文件中的张量
                self.data.append((tensor, i))  # 将张量和对应的标签索引添加到列表中

    def __len__(self):
        return len(self.data)  # 返回数据集的长度

    def __getitem__(self, index):
        tensor, label = self.data[index]  # 根据索引获取张量和标签
        if self.transform is not None:
            tensor = self.transform(tensor)  # 应用数据变换
        return tensor, label  # 返回张量和标签


