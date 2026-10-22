# 实验三：图像关键点检测及图像拼接
### 202410315012_张连艺

## 一、实验目的
1.理解 SIFT 尺度不变特征变换算法原理，掌握图像关键点、特征描述子的提取方法。
2.掌握 FLANN 匹配器、K 近邻匹配以及 Lowe 比率测试筛选匹配点的流程
3.掌握透视变换、图像平移拼接方法，实现两张有重叠区域图像的全景拼接。

## 二、实验内容
### 1.导入必要库
- cv2:OpenCV库 <br>
- numpy:Numpy库 <br>
```python
import cv2
import numpy as np
```

### 2.读取图像并预处理
这里先读取两张图像作为拼接的原素材， 再将图像转换为灰度图
```python
# 读取图像
img_a = cv2.imread("a.jpg")
img_b = cv2.imread("b.jpg")
# 将图像转换为灰度图
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray_a.jpg", gray_a)
cv2.imwrite("gray_b.jpg", gray_b)
```
结果图：

<img width="403" height="533" alt="image" src="https://github.com/user-attachments/assets/f6d363e4-3086-497a-acf1-1dd8ad9385c9" />
<img width="438" height="530" alt="image" src="https://github.com/user-attachments/assets/9109be35-0d71-4100-b2f7-3dd2c33a1352" />

### 3.关键点检测
SIFT 可以提取图像上具有尺度不变、旋转不变特性的关键点，并生成 128 维特征描述子，描述关键点邻域梯度信息，用于不同图像之间特征比对。
即使图像存在缩放、旋转、亮度变化，依然可以稳定找到同名特征点。
初始化SIFT检测器并提取特征：
```python
sift = cv2.SIFT_create()
kp_a,des_a = sift.detectAndCompute(gray_a,None)
kp_b,des_b = sift.detectAndCompute(gray_b,None)
```
使用FLANN算法快速找到两幅图像特征点之间的对应关系，然后再过滤掉不可靠的匹配点，只保留高质量的匹配，最后得到高质量的关键点
```python
# 使用FLANN匹配器进行特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 检查次数越多，匹配越准确但速度越慢
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_a, des_b, k=2)  # k=2表示每个特征点返回2个最佳匹配

# 应用Lowe's比率测试筛选优质匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 比率阈值通常取0.7-0.8
        good_matches.append(m)

# 输出匹配统计信息
print("=== 特征匹配统计 ===")
print(f"初始匹配点对数: {len(matches)}")
print(f"筛选后优质匹配点对数: {len(good_matches)}")
print(f"过滤掉的匹配点对数: {len(matches) - len(good_matches)}")
print(f"优质匹配点比例: {len(good_matches)/len(matches):.2%}")
```
输出结果：
```python
=== 特征匹配统计 ===
初始匹配点对数: 2728
筛选后优质匹配点对数: 863
过滤掉的匹配点对数: 1865
优质匹配点比例: 31.63%
```
绘制匹配的SIFT关键点：
```python
matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a, img_b, kp_b, good_matches,
    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```
结果如下：

<img width="2559" height="1588" alt="matched_keypoints" src="https://github.com/user-attachments/assets/712a6114-938c-4373-8c21-ab9fa97e7e3c" />



### 4.图像拼接
#### 图像拼接的几何变换核心流程
- 图像拼接的几何变换过程始于特征点坐标的精确提取，其中`src_pts`从待变换的源图像（img_b）中获取匹配关键点坐标，而`dst_pts`则从基准图像（img_a）中提取对应点坐标，这些点对建立了两幅图像之间的空间对应关系。接着，系统采用RANSAC算法鲁棒地估计3×3单应性矩阵H，该矩阵定义了从img_b到img_a的透视变换关系，其中的重投影误差阈值（5.0）有效控制了内点筛选的严格程度，确保在存在部分错误匹配的情况下仍能获得准确的变换参数。
- 在确定几何关系后，算法计算img_b四个角点经过透视变换后的新位置，通过将变换后的img_b角点与原始img_a角点合并，计算出所有角点的最小和最大坐标值，从而确定最终拼接图像的完整尺寸范围。为了处理变换后可能出现的负坐标问题，系统创建了一个平移矩阵，将整个变换结果平移到正坐标区域，确保所有像素都位于可视范围内。
- 最后，通过对img_b应用组合变换（透视变换与平移变换的结合），并将img_a精确复制到变换后图像的对应位置，实现了两幅图像的无缝空间对齐和内容融合。这一完整的"特征匹配→几何估计→图像变换"技术流水线，体现了计算机视觉中图像配准的核心原理，为生成高质量的全景拼接图像奠定了坚实基础。整个过程中，单应性矩阵发挥了关键的坐标映射作用，而RANSAC算法则保证了在复杂场景下的算法鲁棒性，使得即使存在一定比例的误匹配，仍能获得令人满意的拼接效果

```python
# 提取匹配点的坐标
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像的关键点
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像的关键点

# 使用RANSAC算法估计单应矩阵（透视变换矩阵）
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 获取输入图像尺寸
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

# 计算图像变换后的四个角坐标
pts = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
dst_corners = cv2.perspectiveTransform(pts, H)

# 确定拼接后图像的最终尺寸（包含所有像素）
all_corners = np.concatenate([dst_corners, np.float32([[0,0], [w_a,0], [w_a,h_a], [0,h_a]]).reshape(-1,1,2)], axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() + 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# 创建平移矩阵，确保所有像素都在可视区域内
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

# 对图像进行透视变换和平移
stitched_img = cv2.warpPerspective(
    img_b,
    translation_matrix @ H,  # 组合平移矩阵和单应矩阵
    (x_max - x_min, y_max - y_min)  # 输出图像尺寸
)

# 将图像复制到拼接结果的对应位置
stitched_img[-y_min : h_a - y_min, -x_min : w_a - x_min] = img_a
```
显示拼接结果并保存
```python
# 显示匹配关键点和拼接结果
cv2.imshow('Matched SIFT Keypoints', matched_keypoints_img)  # 显示匹配的关键点
cv2.imshow('Stitched Result', stitched_img)                   # 显示拼接后的图像
cv2.waitKey(0)  # 等待用户按键（按任意键关闭窗口）
cv2.destroyAllWindows()  # 关闭所有显示窗口

# 保存拼接结果
cv2.imwrite("matched_keypoints.jpg", matched_keypoints_img)
cv2.imwrite("stitched_result.jpg", stitched_img)
```
结果图：
<img width="2115" height="1588" alt="stitched_result" src="https://github.com/user-attachments/assets/5db77df7-2405-47a1-9000-fe859eb1642f" />

## 三、实验结果与分析
###1.实验结果
本次实验输入两张存在重叠区域的图像，输出统计：
- 初始匹配点对数：2728
- 筛选后优质匹配点对数：863
- 过滤掉的匹配点对数：1865
- 优质匹配点比例：31.63%

结果输出两张图片：
1. `matched_keypoints.jpg`：绘制两张图之间成功匹配的特征点连线，可以直观看到重叠区域的匹配关系。
2. `stitched_result.jpg`：全景拼接图像，两张图像完成配准，重叠区域对齐，生成广角全景图。

### 2.实验分析
1. 经过 Lowe 比率测试后，匹配点数量大幅下降，大量模糊、错误匹配被剔除，保证后续单应矩阵求解稳定。
2. RANSAC 算法有效抵抗误匹配点的干扰，即使存在部分外点，依然可以估算出合理的透视变换矩阵。
3. 本实验仅做简单直接拷贝拼接，没有重叠区域融合，拼接接缝处会存在明显亮度分界线；后续可以引入加权融合、泊松融合优化接缝效果。
4. 影响拼接效果的因素：两张图像重叠区域大小、拍摄是否绕光心旋转、图像纹理丰富度；纹理过少的区域 SIFT 提取特征点少，容易拼接失败。
5. SIFT 算法计算开销偏大，特征点数量多的时候运行速度慢，在实时场景下可以替换为 ORB 等轻量特征算法。

## 四、实验总结与心得
本次实验完成了基于 SIFT 特征、RANSAC 单应矩阵估计的图像全景拼接。完整实现了特征提取、特征匹配、误匹配剔除、透视变换、画布平移拼接整套流程。实验验证了 SIFT 对光照、旋转变化的鲁棒性，理解了 RANSAC 在含有噪声数据下的参数估计思路。同时也看到直接拼接方案的缺陷，为后续图像融合、多图连续全景拼接打下基础。
