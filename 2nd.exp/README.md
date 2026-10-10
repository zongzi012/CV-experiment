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
<img width="1638" height="424" alt="image" src="https://github.com/user-attachments/assets/153e93a4-7ac3-420c-8863-c3e4bf7e1b4f" />

高斯噪声的三种滤波对比图：
<img width="1642" height="417" alt="屏幕截图 2026-10-08 142527" src="https://github.com/user-attachments/assets/75e9438f-463e-4fad-a21b-6419cfb789a4" />

**三种滤波区别分析**
1. 均值滤波 cv2.blur
对滤波窗口内所有像素取算术平均值替换中心像素。属于线性滤波。
对高斯噪声有平滑抑制效果；但是会模糊图像边缘细节。面对椒盐噪声，黑白极值点会参与平均计算，噪点很难消除，只能轻微淡化噪点。

2. 中值滤波 cv2.medianBlur
将窗口内像素灰度排序，取中间值作为输出，属于非线性滤波。对椒盐噪声效果最好，孤立黑白极值在排序后会被直接丢弃，去噪同时可以较好保留物体边缘轮廓。但对于高斯噪声，降噪能力有限。

3. 高斯滤波 cv2.GaussianBlur
窗口内像素按照高斯分布做加权平均，中心像素权重最高，越远离中心权重越低，线性滤波。
擅长处理高斯噪声，平滑效果柔和，相比均值滤波边缘保留效果更好。但本质仍是加权平均，无法有效去除椒盐噪声的孤立黑白噪点。

> 
> 适配规律：椒盐噪声优先使用中值滤波；高斯噪声适合高斯滤波或者均值滤波。
### 5.手动实现一个滤波方式
```python
def manual_median_filter_color(image, kernel_size=5):
    pad = kernel_size // 2
    filtered_img = np.zeros_like(image)
    # 逐个通道处理
    for c in range(3):
        channel = image[:, :, c]
        padded_channel = np.pad(channel, pad_width=pad, mode='edge')
        h, w = channel.shape
        for i in range(h):
            for j in range(w):
                region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                filtered_img[i, j, c] = np.median(region)
    return filtered_img #中值滤波后的彩色图像

# 椒盐噪声图转uint8输入手动滤波
manual_mid = manual_median_filter_color((sp_noise_img * 255).astype(np.uint8), kernel_size=5)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(sp_noise_img)
plt.title("s&p noise img")

plt.subplot(1, 2, 2)
plt.imshow(manual_mid)
plt.title("manual median filter")
```
⼿动构建中值滤波效果图：
<img width="1040" height="451" alt="image" src="https://github.com/user-attachments/assets/3bdc8786-9260-444d-ad58-97c4b1cdaea4" />

### 6.总体分析
1. OpenCV 图像存储为 BGR 通道，Matplotlib 按照 RGB 解析，直接显示会发生色彩错乱，必须做通道转换。灰度图像抛弃色彩，仅保留亮度，简化后续图像处理。

2. 椒盐噪声是孤立突变噪点，高斯噪声是全局均匀随机扰动，两类噪声特性完全不同，不能用同一套滤波处理。

3. 线性滤波（均值、高斯滤波）适合处理服从正态分布的高斯噪声；非线性的中值滤波更适合脉冲式椒盐噪声。均值滤波容易造成图像模糊；高斯滤波加权平滑，视觉效果更自然。

4. 手动实现彩色中值滤波证明，彩色图像需要分开各个通道进行滤波处理，手动算法与库函数效果基本一致，加深对滑动窗口滤波原理的理解。
   
## 三、实验小结 
本次实验完成图像读取、BGR‑RGB 色彩空间转换、灰度图像生成，完成添加椒盐噪声、高斯噪声，对比均值滤波、中值滤波、高斯滤波的去噪效果，并且手动实现彩色版本的中值滤波。

实验证明：中值滤波对椒盐噪声去除效果最优，能够保护图像边缘；高斯滤波适合抑制高斯噪声；均值滤波会带来明显的图像模糊。不同噪声类型应当选择适配的滤波算法。手动实现滤波帮助理解滑动窗口、通道分离、边缘填充的底层逻辑，为后续图像增强算法学习打下基础。
