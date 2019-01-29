# pytorch数据增强

## 常见transform
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

## img与seg同变换
### 用随机种子
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

### 用函数
```python
import torchvision.transforms.functional as tf
def transform(self, image, seg, size = None):

    # Resize
    if size:
        image = tf.resize(image, size)
        seg = tf.resize(seg, size)

    # 随机裁剪 指定裁剪后大小
    # 由于函数tf.cro需要制定参数，这些参数太麻烦易错，我不想自己写，所以可以这么做
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=size)
    image = tf.crop(image, i, j, h, w)
    seg = tf.crop(seg, i, j, h, w)

    #中心裁剪
    image = tf.center_crop(image, size)

    # 随机水平翻转，默认的概率就是0.5
    if random.random() > 0.5:
        image = tf.hflip(image)
        seg = tf.hflip(seg)

    # 随机竖直翻转
    if random.random() > 0.5:
        image = tf.vflip(image)
        seg = tf.vflip(seg)

    # 亮度，对比度，饱和度增强；在原来的基础上上下浮动0.2
    # 由于只需要对图像做处理，所以不需要考虑同变换的问题
    CJ = tfs.ColorJitter.get_params(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)
    image = CJ(image)

    # 变成tensor
    image = tf.to_tensor(image)
    seg = tf.to_tensor(seg)

    return image, seg
```
