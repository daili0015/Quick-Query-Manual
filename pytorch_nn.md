# pytorch数据增强

##　用类封装
### 定义transform组合
```python
import torchvision.transforms as tfs
train_tfs = tfs.Compose([
	#中心crop
	tfs.CenterCrop(size=(h, w)) #或者填int
	#随机crop后resize
	tfs.RandomResizedCrop(size=(32, 32), scale = (0.8, 1.0)),
	#随机水平翻转
	tfs.RandomHorizontalFlip(),
	#resize
	tfs.Resize(size=(32, 32)),
	#亮度，对比度，饱和度增强；在原来的基础上上下浮动0.2
	tfs.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0)
	#变为tensor
    tfs.ToTensor(),
	#归一化，按照imageNet的均值
    tfs.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
    ])
```
[更多数据增强方法](https://blog.csdn.net/u011995719/article/details/85107009)

### 保证图像与分割图同等变换
在分割任务中，要求原图与分割图进行相同的变换
```python
import random
TrainImg_tfs = tfs.Compose([
    tfs.RandomHorizontalFlip(),
    tfs.Resize(size=(256, 384)), #分别是高和宽
    tfs.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    tfs.ToTensor(),
    # tfs.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
    ])
TrainSeg_tfs = tfs.Compose([
    tfs.RandomHorizontalFlip(),
    tfs.Resize(size=(256, 384)),
    tfs.ToTensor(),
    ])

# 在dataset的__getitem__()里面这么写
if self.transform:
    # 让img和seg共享同一个随机种子
    seed = np.random.randint(2147483647)
    random.seed(seed)
    img = self.transform[0](img)
    random.seed(seed)
    seg = self.transform[1](seg)
```

##　用函数灵活地封装
### 定义transform函数

