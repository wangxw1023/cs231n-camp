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

2.1 NVIDIA Driver run文件下载

在[NVIDIA Driver Downloads](https://www.nvidia.com/Download/Find.aspx?lang=en-us)页面输入Product Type, Product Series, Product, Operating System等查询条件，即可下载对应的NVIDIA Driver安装文件。

本文下载的NVIDIA Driver run文件名为：`NVIDIA-Linux-x86_64-418.43.run`，存储目录为`192.168.0.174`的`~/work/xiaowan`。

2.2 CUDA Toolkit run文件下载

在[CUDA Toolkit Archive Downloads](https://developer.nvidia.com/cuda-toolkit-archive)页面选择所需安装的CUDA版本，然后在点开的页面中选择Operating System, Architecture, Distribution, Version, Installer Type即可下载对应的CUDA Toolkit 安装文件。

本文下载的CUDA Toolkit run文件名为：`cuda_9.2.148_396.37_linux.run`，存储目录为`192.168.0.174`的`~/work/xiaowan`。

2.3 NVIDIA cuDNN文件下载

在[cuDNN Download](https://developer.nvidia.com/rdp/cudnn-download)页面进行文件下载。

NVIDIA cuDNN文件下载需要用NVIDIA account登录。

登录后，选择`I Agree To the Terms of the cuDNN Software License Agreement`，即可进行NVIDIA cuDNN版本的选择。点击`Archived cuDNN Releases`选项，可以下载NVIDIA cuDNN的历史发布版本。

NVIDIA cuDNN的版本选择跟CUDA Toolkit、TensorFlow、python等软件版本关联，具体可查看[经过测试的构建配置](https://www.tensorflow.org/install/source#linux).

选择了NVIDIA cuDNN版本后，还需根据操作系统选择相应的版本，

本文选择`Download cuDNN v7.6.0 (May 20, 2019), for CUDA 9.2`；

然后选择下载`cuDNN Library for Linux`；

下载的文件名为：`cudnn-9.2-linux-x64-v7.6.0.64.solitairetheme8`，存储目录为`192.168.0.174`的`~/work/xiaowan`。

`.solitairetheme8`是一种压缩文件，可以转为`tgz`文件。

```
cp  cudnn-9.2-linux-x64-v7.6.0.64.solitairetheme8 cudnn-9.2-linux-x64-v7.6.0.64.tgz
tar -xvf cudnn-9.2-linux-x64-v7.6.0.64.tgz
```

### 3. 安装NVIDIA Driver

step1. 添加`nomodeset`  
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
更新`grub`
```shell
sudo update-grub
```
重启系统。
```shell
shutdown -r now
```

step2. 禁止集成的`noubeau`驱动
Ubuntu系统集成的显卡驱动程序是`nouveau`，它是第三方为NVIDIA开发的开源驱动，我们需要先将其屏蔽才能安装NVIDIA官方驱动。
将驱动添加到黑名单`blacklist.conf`中。
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
step3. 安装NVIDIA Driver
```
sudo service lightdm stop
sudo sh NVIDIA-Linux-x86_64-418.43.run
sudo service lightdm start
```
注：在安装过程中选择替换Ubuntu自带的X server，即下面所述：
```
Would you like to run the nvidia-xconfig utility to automatically update your X configuration file so that the NVIDIA X driver will be used when you restart X?  Any pre-existing X configuration file will be backed up.
```
step4. 检验
在shell中输入
```shell
nvidia-smi
```
可以看到显卡信息，会列出全部检测到的显卡。

### 4. 安装CUDA Toolkit 和 NVIDIA cuDNN

CUDA Toolkit的安装可以参考官方的文档：[CUDA Toolkit Documentation v9.2.148](https://docs.nvidia.com/cuda/archive/9.2/)

step1. Pre-installation Actions：
- Verify the system has a CUDA-capable GPU.
- Verify the system is running a supported version of Linux.
- Verify the system has gcc installed.
- Verify the system has the correct kernel headers and development packages installed.
- Download the NVIDIA CUDA Toolkit.
- Handle conflicting installation methods.

step2. CUDA Toolkit Runfile Installation
```shell
sudo sh cuda_9.2.148_396.37_linux.run
```

注：在CUDA Toolkit的安装过程中，有一个选项一定要引起注意，如下所示，因为NVIDIA Driver已经安装过了，所以这里一定要选n。
```
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 396.37?
    (y)es/(n)o/(q)uit: n   #因为NVIDIA驱动已经安装好了，所以这里不需要
```

step3. NVIDIA cuDNN安装

在2.3中，我们下载了NVIDIA cuDNN，并对`tar`文件解压，可以在`~/work/xiaowan`目录中看到一个`cuda/`；

NVIDIA cuDNN安装即将其中的`lib64`及`include`文件夹复制到CUDA的安装路径。

CUDA默认的安装路径是`/usr/local/cuda-9.2`。
```shell
sudo cp cuda/lib64/* /usr/local/cuda-9.2/lib64/
sudo cp cuda/include/* /usr/local/cuda-9.2/include/
```

最后需要在`~/.bashrc`中配置如下的环境变量
```shell
export CUDA_INSTALL_DIR=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
执行
```shell
source ~/.bashrc
```
step4. 检验
在命令行中输入
```shell
cat /usr/local/cuda/version.txt
```
可以看到安装的CUDA Toolkit版本信息。

在命令行中输入
```shell
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
可以看到安装的NVIDIA cuDNN版本信息。

### 5. 安装NVIDIA Driver、CUDA Toolkit和NVIDIA cuDNN过程中踩过的坑（欢迎后续安装环境的小伙伴补充）

因为每台服务器的环境毕竟还是有区别的，所以参照上述流程也不一定能够顺利安装各个软件，但是基本上通过百度或者翻墙谷歌，都还是能解决的啦~

5.1 安装NVIDIA Driver的step3
- 执行`sudo service lightdm stop`
- 报错`Failed to stop lightdm.service: Unit lightdm.service not loaded.`
- 执行`sudo apt install lightdm`
- 报错`Unable to find a suitable destination to install 32-bit compatibility libraries. Your system may not be set up for 32-bit compatibility. 32-bit compatibility files will not be installed; if you wish to install them, re-run the installation and set a valid directory with the --compat32-libdir option.`
- 执行
```
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install libc6:i386
```
- 成功

5.2 安装NVIDIA Driver的step4：nvidia-smi展示结果疑问 （超级巨大的坑！！！）
- 执行`nvidia-smi`
- 结果
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.43       Driver Version: 418.43       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
```
- 分析：因为明明只安装了NVIDIA Driver，并没有装CUDA Toolkit，并且跟所有可能动174服务器的小伙伴确认，也没有人装过CUDA，那么这个CUDA Version是哪里得来的，百思不得其解……在服务器上也没有搜到任何CUDA相关的内容。。。
- 答案：排除了各种我们自己的原因后，终于在NVIDIA的官方论坛上找到了答案：[CUDA 10 installation problems on Ubuntu 18.04](https://devtalk.nvidia.com/default/topic/1045400/cuda-setup-and-installation/cuda-10-installation-problems-on-ubuntu-18-04/)
```
It looks to me like you simply haven't installed CUDA 10. You have an updated GPU driver (415.18). 
However, the fact that nvidia-smi indicates: CUDA Version: 10.0 doesn't actually mean you have CUDA 10 installed.
```
- 也就是说这个现实的CUDA Version只是一个你装的驱动推荐的CUDA版本，跟你的服务器本身装的CUDA毫无关系。。。

5.3 安装NVIDIA Driver

- `The distribution-provided pre-install script failed!  Are you sure you want to continue?`
- 解决方法：直接忽略
- 参考：[ubuntu 16 安装Nvidia显卡驱动以及cudn](https://zhuanlan.zhihu.com/p/31575356)

5.4 `nvcc --version`与`cat /usr/local/cuda/version.txt`查询结果不一致问题
- 执行：`nvcc --version`
- 结果：`The program 'nvcc' is currently not installed. You can install it by typing: apt install nvidia-cuda-toolkit`
- 习惯性根据服务器提示执行：`apt install nvidia-cuda-toolkit`
- 最后命令执行完发现NVIDIA Driver有一次将版本退回到了384.130，后来又试了一次更新到了最新的版本430.26。。。但是`nvcc --version`显示的CUDA版本还是7.5。。。
- 解决方法：sudo vim /usr/bin/nvcc，把里面的内容"exec /usr/lib/nvidia-cuda-toolkit/bin/nvcc" 改成"exec /usr/local/cuda/bin/nvcc"
- 参考：[CUDA版本检测](https://zhuanlan.zhihu.com/p/48641682)

### 6. 其他参考
* [Ubuntu16.04 安装 Nvidia Drivers+Cuda+Cudnn](https://zhuanlan.zhihu.com/p/68069328)
* [CUDA wiki](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA)
* [深度学习服务器搭建及开发环境配置](https://gitlab.tmxmall.com/tmxmall_nmt/t2t_transformer_nmt/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%9C%8D%E5%8A%A1%E5%99%A8%E6%90%AD%E5%BB%BA%E5%8F%8A%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE.md)
