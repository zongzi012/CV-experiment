# 实验一：计算机视觉库的安装

## 实验目的：
掌握 Anaconda 的安装与基本操作，熟悉 GPU 使用环境的配置及对应版本 PyTorch 的安装，并完成 OpenCV 的安装与配置

## 实验内容：
### 1、Anaconda的安装及配置
已安装如图展示：

<img width="681" height="196" alt="1" src="https://github.com/user-attachments/assets/c9bb7c7e-21c4-4727-855e-100a625c0927" />

### 2、conda的基本操作与OpenCV的安装
1.  conda create -n [env_name] python==[version] 创建虚拟环境并制定python版本。
2.  activate cv 进入创建的虚拟环境， pip install opencv-python 安装OpenCV。
已安装如图展示：

<img width="1143" height="234" alt="2" src="https://github.com/user-attachments/assets/647207b3-5dbe-469f-92ce-051b97e41df4" />

### 3、GPU加速环境配置
nvidia-smi 显示显卡状态信息，如下：
<img width="1526" height="443" alt="3" src="https://github.com/user-attachments/assets/8a849476-c150-4b43-8f11-f456b8519451" />

### 4、PyTorch安装
<img width="1154" height="266" alt="4" src="https://github.com/user-attachments/assets/77054461-31d2-4331-ba0c-4374ad0c9839" />

### 5、PyTorch GPU加速环境验证
<img width="1832" height="453" alt="5" src="https://github.com/user-attachments/assets/014992aa-97e5-45fb-9281-22b82d6c4a66" />

## 实验总结
本次实验完成了 Anaconda、OpenCV 以及 PyTorch GPU 环境的部署与验证。通过 Anaconda 创建独立的`cv`虚拟环境，实现项目依赖隔离，避免不同库版本之间产生冲突；使用 pip 完成 OpenCV 安装，成功配置计算机视觉基础库。
在 GPU 环境配置中，通过`nvidia-smi`查询显卡驱动信息，本机 RTX 5060 显卡驱动支持最高 CUDA 12.8 版本，安装了带 CUDA 126 编译版本的 PyTorch。虽然没有单独安装 CUDA Toolkit，无法使用`nvcc`命令，但 PyTorch 内置 CUDA 运行时。运行验证代码，`torch.cuda.is_available()`和`torch.backends.cudnn.is_available()`均输出`True`，证明 PyTorch 可以正常调用 GPU，GPU 加速环境可用。
熟悉了 conda 虚拟环境管理、Python 视觉库安装以及 PyTorch GPU 环境排查的基本流程，理解了显卡驱动、CUDA 版本、PyTorch 版本三者之间的匹配关系，掌握了 GPU 可用性的验证方法，为后续计算机视觉模型训练与加速打下环境基础。
