# 图像分割预处理
## 重要操作
### 读取图片列表
voc2012的数据集是通过读取txt得到的
```python
import os
def read_imgs(root, train = True):
    # you can see a folder named 'VOC2012' under root
    txt_fname = os.path.join(root, 'VOC2012\ImageSets\Segmentation', 'train.txt' 
        if train else val.txt)
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    imgs = [os.path.join(root, 'VOC2012\JPEGImages',  i+'.jpg') for i in images]
    segs = [os.path.join(root, 'VOC2012\SegmentationClass',  i+'.png') for i in images]
    return imgs, segs
```
### 处理标注图
把原来的彩色分割图片变成训练需要的tensor

首先获得一个数组cmap，它在指定的下标处存储了类别i

cmap[ 颜色对应的哈希值 ] = 类别i

只有21个地方的值有i，只有它们是有意义的

```python
import numpy as np
# RGB color for each class
COLORMAP = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

#<<操作符优先级太低了！！！必须加括号
color_hash = lambda color: (((color[0]<<8) + color[1])<<8) + color[2]
seg_hash = lambda color: (((color[:,:,0]<<8) + color[:,:,1])<<8) + color[:,:,2]

# 本想用字典的，但是字典无法实现花哨的索引
def get_cmap(colormap):
    cmap = np.zeros(256 ** 3, dtype=np.int32)
    for i, cm in enumerate(COLORMAP):
        cmap[ color_hash(cm) ] = i
    return cmap

cmap = get_cmap(COLORMAP)
print(cmap.shape)

```
利用numpy花哨索引，转换标注图
```python
# https://pytorch.org/docs/stable/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss
# 可见需要的label的格式是 (batch_size, h, w)
# 给一张彩色的seg图，转换为值在0-21之间的numpy数组，方法是numpy的花式索引
def seg2label(seg, cmap):
    data = np.array(seg, dtype=np.int32)
    ind = seg_hash(data)
    label = cmap[ind]
    return label
```
## 数据集类
```python
class VocFolder(Dataset): #Dataset是ImageFolder的爷爷！！！

    def __init__(self, root, transform=False):
        self.root = root
        self.img_list, self.seg_list = read_imgs(self.root)
        self.img_list, self.seg_list = self._filter_(self.img_list), self._filter_(self.seg_list)
        self.transform = transform

    def __getitem__(self, index):

        img = Image.open(self.img_list[index]).convert('RGB')
        seg = Image.open(self.seg_list[index]).convert('RGB') # 1ms
        
        if self.transform:
            img, seg = self.augment(img, seg, size = (256, 384))

        label = seg2label(seg, cmap) # 3ms

        return img, label

    def __len__(self):
        return len(self.img_list)

    def _filter_(self, img_list, ratio = 1.2): #1.2这个比例1400剩下1100，可以了
        res = []
        for im in img_list:
            size = Image.open(im).size
            if size[0]/size[1] > ratio:
                res += im,
        return res

    def augment(self, image, seg, size = None):
		#数据增强的实现连接在下面

    def all_size(self): #查看数据集的图像尺寸
        sizes = set()
        ratios = set()
        for i in self.img_list:
            img_size = Image.open(i).convert('RGB').size
            sizes.add(img_size)
            ratios.add(img_size[0]/img_size[1])
        return sizes, ratios

```
数据增强的[实现](https://github.com/daili0015/Quick-Query-Manual/blob/master/pytorch_augment.md#img与seg同变换)

查看一下，记得把to_tensor注释掉
```python
fo = VocFolder( './VOCdevkit', True )
for i in range(10, 20):
    img, label = fo[i]
    plt.imshow(img)
    plt.show()
    plt.imshow(label)
    plt.show()

```