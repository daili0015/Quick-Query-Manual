# pytorch神经网络常用指令
##　图形变换

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
