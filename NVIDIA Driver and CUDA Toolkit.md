### 1. NVIDIA Driver和CUDA Toolkit概述
配置GPU，我们需要依次安装以下三大主要部分：
1. NVIDIA Driver (英伟达显卡驱动)
2. CUDA Toolkit ( parallel computing platform and application programming interface(API) )
3. NVIDIA cuDNN ( NVIDIA CUDA® Deep Neural Network library )

其中，NVIDIA Driver负责与底层的GPU交互，CUDA Toolkit通过包装并提供了一系列常用并行计算接口，NVIDIA cuDNN则是对深度学习的常见运算做进一步包装提供对应接口。

三者之间有很深的相互依赖关系，在安装之前就需确定好三者的版本。一般来说一旦安装完NVIDIA Driver，后两者的版本也就确定了。

一般我们会根据服务器的GPU型号，以及相应的软件需求，如TensorFlow等来选择三者的版本。CUDA toolkit version、NVIDIA Driver version和GPU architecture三者版本依赖关系如下：

|CUDA toolkit version|	Driver version|	GPU architecture|
|:--|:--|:--|  
|6.5|	>= 340.29|	>= 2.0 (Fermi)|
|7.0|	>= 346.46|	>= 2.0 (Fermi)|
|7.5|	>= 352.39|	>= 2.0 (Fermi)|
|8.0|	== 361.93 or >= 375.51|	== 6.0 (P100)|
|8.0|	>= 367.48|	>= 2.0 (Fermi)|
|9.0|	>= 384.81|	>= 3.0 (Kepler)|
|9.1|	>= 387.26|	>= 3.0 (Kepler)|
|9.2|	>= 396.26|	>= 3.0 (Kepler)|
|10.0|	>= 384.111, < 385.00|	Tesla GPUs|
|10.0|	>= 410.48|	>= 3.0 (Kepler)|
|10.1|	>= 384.111, < 385.00|	Tesla GPUs|
|10.1|	>=410.72, < 411.00|	Tesla GPUs|
|10.1|	>= 418.39	>= 3.0| (Kepler)|
### 2. 分别下载文件


### 3. 安装NVIDIA Driver

### 4. 安装CUDA Toolkit

### 5. 安装NVIDIA Driver和CUDA Toolkit过程中踩过的坑（欢迎后续安装环境的小伙伴补充）

### 6. 参考
* [Ubuntu16.04 安装 Nvidia Drivers+Cuda+Cudnn](https://zhuanlan.zhihu.com/p/68069328)
* [CUDA wiki](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA)
* [在Docker中使用Tensorflow Serving](http://fancyerii.github.io/books/tfserving-docker/)
* [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)
* [ubuntu 16.04系统下GTX970显卡不支持导致无法开机或开机黑屏解决方法](https://blog.csdn.net/Good_Day_Day/article/details/74352534)
* [Ubuntu16.04 + 1080Ti深度学习环境配置教程](https://www.jianshu.com/p/5b708817f5d8?from=groupmessage)
* [Nvidia驱动下载](https://www.geforce.cn/drivers)
