# pytorch神经网络常用指令
##　图形变换

```python
import torchvision.transforms as tf
train_tf = tf.Compose([
	#中心crop
	tf.CenterCrop(size=(h, w)) #或者填int
	#随机crop后resize
    tf.RandomResizedCrop(size=(32, 32), scale = (0.8, 1.0)),
	#随机水平翻转
	tf.RandomHorizontalFlip(),
	#resize
	tf.Resize(size=(32, 32)),
	#亮度，对比度，饱和度增强；在原来的基础上上下浮动0.2
	tf.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0)
	#变为tensor
    tf.ToTensor(),
	#归一化，按照imageNet的均值
    tf.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
    ])
```

[更多数据增强方法](https://blog.csdn.net/u011995719/article/details/85107009)
