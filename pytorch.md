# pytorch用于数据分析

## 数据
必须变成DataLoader才能用于神经网络的训练，下面是一个标准的方法
```python 
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
def get_data(x, y, batch_size, shuffle):
    tx = torch.from_numpy(x).type(torch.FloatTensor)
    ty = torch.from_numpy(y).type(torch.FloatTensor)
    dataset = TensorDataset(tx, ty) #快速建立pytorch格式的DataLoader
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=0)
train_data = get_data(X_train, y_train, 128, True)
valid_data = get_data(X_test, y_test, 256, False)
```
其中数据类型需要变为float，[官方数据类型参考](https://pytorch.org/docs/stable/tensors.html)
## 训练
首先准备网络
```python
from torch import nn
from torch.optim.lr_scheduler import StepLR
# 构建网络与优化器
hidden = 5
model = nn.Sequential(
    nn.Linear(features, hidden),
    nn.ReLU(True),
    nn.Linear(hidden, 1),
    )
opm = torch.optim.Adam(model.parameters(), lr = 1e-1, weight_decay=1e-5)
Loss = nn.MSELoss(size_average=True) #除以样本数
lr_sche = StepLR(opm, 20, gamma = 0.6)
```
然后训练，下面是回归的训练代码
```python
import os
n_iter = 100
use_pretrained = True
if use_pretrained and os.path.exists("fnn.pth"):
    model.load_state_dict(torch.load("fnn.pth"))
    print("model has been loaded")

n_batch = len(train_data)
for i in range(n_iter):
    lr_sche.step()
    train_loss = 0
    for x, y in train_data:
        fx = model(x)
        loss = Loss(fx, y)
        opm.zero_grad()
        loss.backward()
        opm.step()
        train_loss += loss.data.numpy()
    print("loss is %f" %(train_loss/n_batch))
```
## 测试与保存
```python
from sklearn.metrics import mean_squared_error
sum_mse = 0
model.eval()
for x, y in valid_data:
    fx = model(x)
    sum_mse += mean_squared_error(fx.detach().numpy(), y)
print("valid mse is %f" %(sum_mse/len(valid_data)))
#保存
torch.save(model.state_dict(), "fnn.pth")
#载入
model.load_state_dict(torch.load("fnn.pth"))
```
