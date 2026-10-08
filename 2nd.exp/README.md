# 实验二：图像增强  
### 202410315012_张连艺

## 一、实验目的
学会OpenCV的基本使用方法，利用OpenCV等计算机库对图像进行平滑、滤波等操作，实现图像增强。

## 二、实验内容
### 1.导入图像滤波相关的依赖包
OpenCV库：计算机视觉与图像处理
scikit-image库：图像增强与分析，random_noise 用于给图像添加噪声
NumPy库：科学计算的基础库
Matplotlib库：绘图与数据可视化
```python
import cv2
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt
```

### 2.读取原始图像并进行色彩空间转换
读取计算机本地图像⽂件，获取并输⼊【100，100】处像素点的RBG参数并输
出,通过下⾯的输出结果可以看到，【100,100】像
```python
#读取并显示原始图像
img = cv2.imread('p1.jpg')
(b, g, r) = img[100, 100]
print(f"像素(100,100) BGR：{b}, {g}, {r}")
plt.imshow(img)
plt.title('Original Image')
plt.savefig('output_images/original_bgr.jpg', dpi=300)
plt.show()
```
```python
输出：
像素(100,100) BGR：242, 220, 255
```
OpenCV `cv2.imread()` 读取图片默认是 **BGR 通道顺序**，但 matplotlib 的`plt.imshow()`默认按照 **RGB** 解析：

<img width="847" height="657" alt="image" src="https://github.com/user-attachments/assets/d78fe035-cee5-414e-9c35-ddb9c9584738" />

将原始图像BGR格式转换成RGB格式，蓝色和红色互换
```python
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.title('RGB Image')
plt.savefig('output_images/rgb_image.jpg', dpi=300)
plt.show()
```
<img width="848" height="647" alt="image" src="https://github.com/user-attachments/assets/162c0671-d294-4165-a3da-cd3deea093d3" />

转灰度图
```python
# BGR转灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title('Gray Image')
plt.savefig('output_images/gray_image.jpg', dpi=300)
plt.show()
```
<img width="842" height="646" alt="image" src="https://github.com/user-attachments/assets/10434bc7-7db1-408d-bf57-77260625442c" />

### 3.添加噪声
这里在原始图像的基础上添加噪声，椒盐噪声和高斯噪声：
我们调用 random_noise函数，椒盐噪声的amount设定为0.3；高斯噪声均值与方差值分别设定为0.2、0.03）
```python
sp_noise_img = random_noise(rgb_img, mode='s&p', amount=0.3)
gus_noise_img = random_noise(rgb_img, mode='gaussian', mean=0.2, var=0.03)
```
对比图如下：
<img width="714" height="277" alt="image" src="https://github.com/user-attachments/assets/cc8778aa-64aa-4442-b801-7fe854c0d126" />

### 4.图像滤波
```python
# 均值滤波
mean_sp = cv2.blur(sp_noise_img, (5, 5))
mean_gus = cv2.blur(gus_noise_img, (5, 5))
```
```python
#中值滤波
mid_sp = cv2.medianBlur((sp_noise_img * 255).astype(np.uint8), 5)
mid_gus = cv2.medianBlur((gus_noise_img * 255).astype(np.uint8), 5)
```
```python
#高斯滤波
gauss_sp = cv2.GaussianBlur((sp_noise_img * 255).astype(np.uint8), (5, 5), 0)
gauss_gus = cv2.GaussianBlur((gus_noise_img * 255).astype(np.uint8), (5, 5), 0)
```
椒盐噪声的三种滤波对比图：





高斯噪声的三种滤波对比图：

