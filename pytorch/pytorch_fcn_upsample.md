# fcn的双线性插值上采样
## 显示tensor图像
对于处理好的单个tensor图像，可以直接显示出来
```python
import torchvision.transforms as tfs

fo = VocFolder( './VOCdevkit' )
img, label = fo[5]

unloader = tfs.ToPILImage()
image = unloader(img.data) #用data更合适
plt.imshow(image)
plt.show()
```
## 双线性插值
```python
import numpy as np
import torch
import torch.nn as nn

# 定义 bilinear kernel
def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

# 转置卷积
conv_trans = nn.ConvTranspose2d(3, 3, 4, 2, 1)
conv_trans.weight.data = bilinear_kernel(3, 3, 4)

```
## 测试效果
### 直接读取图片
```python
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    
    x = Image.open('./VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg')
    x = np.array(x)
    plt.imshow(x)
    plt.show()

    x = torch.from_numpy(x.astype('float32')).permute(2, 0, 1).unsqueeze(0)

    y = conv_trans(x).data.squeeze().permute(1, 2, 0).numpy()
    plt.imshow(y.astype('uint8'))
    plt.show()
    print(y.shape)   
```
### 从定义的dataset里载入
这种方法显示的处理后图像有些错误
```python
import matplotlib.pyplot as plt
from data import VocFolder

if __name__ == '__main__':
    fo = VocFolder( './VOCdevkit' )
    for i in range(10, 20):
        img, label = fo[i]

        # 把处理好的tensor展示出来
        image = unloader(img.data)
        plt.imshow(image)
        plt.show()

		#处理图像数据
        img = img.unsqueeze(0) #变成一个minibatch的数据
        y = conv_trans(img)
        y = y.data.squeeze()  #变成单个图片的数据

        #查看一下处理后的结果
        y = unloader(y)
        plt.imshow(y)
        plt.show()

```