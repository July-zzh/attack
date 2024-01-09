import datetime
import torch
import torch.nn as nn
from Mydataset import *
from CNN import *
from torch.utils.data import DataLoader


# 模型训练
def main():
    device = "cuda:0"
    model1 = Generator()
    model2 = CNN()
    # model1 = model1.to(device)
    # model2 = model2.to(device)
    model = nn.Sequential(model1, model2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # lr:学习率
    dataset = MyDataset(root="./mnist/mnist")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    path = "./model/model.pkl"
    for epoch in range(0, 20):
        start = datetime.datetime.now()
        model.train()
        for batchsz, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            y_predict = model(data)  # 模型预测
            loss = criterion(y_predict, label)
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # loss值反向传播
            optimizer.step()  # 更新参数
            # index += 1
            if batchsz % 100 == 0:  # 每一百次保存一次模型，打印损失
                end = datetime.datetime.now()
                if not os.path.exists(path):
                    os.makedirs(path, 0o0777)
                torch.save(model.state_dict(), path)  # 保存模型
                print("epoch: {}, 训练次数为：{}，损失值为：{}, 当前时间为: {}".format(epoch, batchsz, loss.item(), end))


main()
