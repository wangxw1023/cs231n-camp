### 1. NVIDIA Driver和CUDA Toolkit概述
配置GPU，我们需要依次安装以下三大主要部分：
1. NVIDIA Driver (英伟达显卡驱动)
2. CUDA Toolkit ( parallel computing platform and application programming interface(API) )
3. NVIDIA cuDNN ( NVIDIA CUDA® Deep Neural Network library )

其中，NVIDIA Driver负责与底层的GPU交互，CUDA Toolkit通过包装并提供了一系列常用并行计算接口，NVIDIA cuDNN则是对深度学习的常见运算做进一步包装提供对应接口。NVIDIA Driver的安装必须有root权限，而后两者可以用一般用户权限安装。

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
### 2. 分别下载安装文件
安装NVIDIA Driver和CUDA Toolkit有多种方式，但是经过各种前人总结以及自己踩坑后推荐分别下载NVIDIA Driver和CUDA Toolkit的run文件进行安装。

1. NVIDIA Driver run文件下载

在[NVIDIA Driver Downloads](https://www.nvidia.com/Download/Find.aspx?lang=en-us)页面输入Product Type, Product Series, Product, Operating System等查询条件，即可下载对应的NVIDIA Driver安装文件。

本文下载的NVIDIA Driver run文件名为：NVIDIA-Linux-x86_64-418.43.run，存储目录为192.168.0.174的/work/xiaowan。

2. CUDA Toolkit run文件下载

在[CUDA Toolkit Archive Downloads](https://developer.nvidia.com/cuda-toolkit-archive)页面选择所需安装的CUDA版本，然后在点开的页面中选择Operating System, Architecture, Distribution, Version, Installer Type即可下载对应的CUDA Toolkit 安装文件。

本文下载的CUDA Toolkit run文件名为：cuda_9.2.148_396.37_linux.run，存储目录为192.168.0.174的/work/xiaowan。

### 3. 安装NVIDIA Driver

step1. 添加nomodeset  
```shell
 sudo vim /etc/default/grub
``` 
将 
```shell
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
```
改为 
```shell
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nomodeset"
```
更新grub
```shell
sudo update-grub
```
重启系统，之后仍然进入文本模式。

step2. 禁止集成的noubeau驱动
Ubuntu系统集成的显卡驱动程序是nouveau，它是第三方为NVIDIA开发的开源驱动，我们需要先将其屏蔽才能安装NVIDIA官方驱动。
将驱动添加到黑名单blacklist.conf中。
```shell
sudo vim /etc/modprobe.d/blacklist.conf
```
在该文件后添加一下几行：
```
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist rivatv
blacklist nvidiafb
 ```
step3. 安装NVidia驱动
仍然在文本模式下操作。在安装过程中选择替换Ubuntu自带的X server。
```
sudo service lightdm stop
sudo sh NVIDIA-Linux-x86_64-390.59.run
sudo service lightdm start
```
step4. 检验
在shell中输入
```shell
nvidia-smi
```
可以看到显卡信息，会列出全部检测到的显卡。


### 4. 安装CUDA Toolkit

### 5. 安装NVIDIA Driver和CUDA Toolkit过程中踩过的坑（欢迎后续安装环境的小伙伴补充）

### 6. 参考
* [Ubuntu16.04 安装 Nvidia Drivers+Cuda+Cudnn](https://zhuanlan.zhihu.com/p/68069328)
* [CUDA wiki](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA)
* [CUDA Toolkit Documentation v9.2.148](https://docs.nvidia.com/cuda/archive/9.2/)
* [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)
* [ubuntu 16.04系统下GTX970显卡不支持导致无法开机或开机黑屏解决方法](https://blog.csdn.net/Good_Day_Day/article/details/74352534)
* [Ubuntu16.04 + 1080Ti深度学习环境配置教程](https://www.jianshu.com/p/5b708817f5d8?from=groupmessage)
