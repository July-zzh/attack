import copy
from PIL import Image
import os
import torch
import torch.nn as nn
from torchvision import models, transforms


def Resnet50(pool: bool):
    model = models.resnet50(pretrained=True)
    if not pool:
        model.avgpool = nn.Sequential()
        model.fc = nn.Linear(2048 * 7 * 7, 1000)
    return model


def get_grad_dl(model: nn.Module, x, y, device):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()]
    )
    model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    x, y = x.to(device), y.to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    grad = torch.autograd.grad(loss, model.parameters())
    grad = [g.detach() for g in grad]
    return grad


# 读取函数，用来读取文件夹中的所有函数，输入参数是文件名
def read_directory(directory_name, to_path, origin_model):
    temp = to_path
    num = 1
    y = torch.tensor([0.])
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()]
    )
    for filename in os.listdir(directory_name):
        print(filename)
        print(num)
        num = num + 1
        img = Image.open(directory_name + "/" + filename).convert('RGB')
        img = transform(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        y = y.type(torch.LongTensor)
        grad = get_grad_dl(origin_model, img, y, device)
        g_w = grad[-2]
        g_b = grad[-1]
        # offset_w = torch.stack([g for idx, g in enumerate(g_w) if idx not in y], dim=0).mean(dim=0) * (1 - 1) / 1
        # offset_b = torch.stack([g for idx, g in enumerate(g_b) if idx not in y], dim=0).mean() * (1 - 1) / 1
        conv_out = (g_w[y]) / (g_b[y]).unsqueeze(1)
        conv_out[torch.isnan(conv_out)] = 0.
        conv_out[torch.isinf(conv_out)] = 0.
        to_path = to_path + "/" + filename + ".pt"
        torch.save(conv_out, to_path)
        to_path = temp


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet50 = Resnet50(pool=False)
origin_model = resnet50.to(device)

read_directory("D:\study\\newfgla\MNIST - JPG - training\\0_1-400", "D:\study\\attack\mnist\\1", origin_model)
