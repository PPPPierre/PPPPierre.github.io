---
title: 【Linux 环境搭建】在 Win10 上搭建 wsl2 虚拟 Linux 环境
author: Stone SHI
date: 2021-09-27 09:00:00 +0200
categories: [Blogging, System]
tags: [System, Linux, wsl]
---

# WSL2 介绍

# 环境搭建步骤

## 安装 WSL2

打开微软[官方 WSL2 安装教程][WSL2_offical_installation_turorial]，按照官方教程里的安装流程安装即可。注意安装 WSL2 对于 Windows 的版本有一定要求，请事先做好检查。

也可以通过安装 [Docker][docker_official_site] 间接安装 WSL2, 在安装过程中 Docker 会要求你下载适用于 Windows 的 Linux 子系统内核，安装后即可。

## 安装 Linux 发行版

在微软商店选择一个 Linux 发行版下载就好，比如我选择的是 Ubuntu 20.04 LTS.

![Ubuntu 20.04 LTS](/pictures/ubuntu_20_04.png)

安装后打开设置用户名密码即可。

安装完毕需要进行简单检验一下，打开 Powershell，输入

```powershell
> wsl -l -v
```

看到输出类似如下：

```powershell
  NAME                   STATE           VERSION
* Ubuntu-20.04           Stopped         2
  docker-desktop         Stopped         2
  docker-desktop-data    Stopped         2
```

后面显示 **VERSION 2** 就对了。

## 安装 Windows Terminal

之后在 Windows 商店下载 **Windows Terminal**。使用比较方便，界面美观，支持一个窗口打开多个 Terminal：命令行、Powershell 和 Linux 都可以一起在一个窗口打开。同时支持自定义，用起来比较舒服，可以通过查看[官方的教程][Windows_terminal_official_customize]来自定义美化自己的 Terminal。

![Windows Terminal](/pictures/windows_t.png)

## 安装图形界面

安装完 Ubuntu 之后可以安装一个对应的图形界面，方便使用。因为目前暂时没有需求，所以只把找到的参考网站贴在这里：

[WSL2 + Ubuntu + 图形界面安装](https://www.bilibili.com/read/cv11143517)

[Windows10 WSL2 Ubuntu 图形界面环境搭建（Xfce4 、XServer）](https://cloud.tencent.com/developer/article/1820014)

推荐 Xfce4 和 VcXSrv 的组合。其中 VcXSrv 至今还一直在维护，稳定性有保证。

# 常见问题解决

这里列举我在安装过程中遇到的问题以及解决的方案。

## 与 Virtualbox 的兼容问题

安装完 WSL2 之后，会发现 Virtualbox 的虚拟机无法启动。在网上查找了资料[^hyper_v_wsl2_2]之后发现二者并不能完美兼容。在需要使用 Virtualbox 的时候，需要把 hyper-V 的虚拟功能关闭[^hyper_v_wsl2_3]，但是关闭了该功能后 WSL2 又无法启动了，目前还没有能完美解决该问题的方案[^hyper_v_wsl2_1]。

在 Powershell 中输入如下指令，停用 hyper-V ，使用 Virtualbox

```powershell
> bcdedit /set hypervisorlaunchtype off
```

在 Powershell 中输入如下指令，启用 hyper-V ，使用 WSL2

```powershell
> bcdedit /set hypervisorlaunchtype auto
```

# 参考链接

[^hyper_v_wsl2_1]:[how-to-get-oracle-vm-virtualbox-6-0-and-wsl-working-at-the-same-time](https://stackoverflow.com/questions/58031941/how-to-get-oracle-vm-virtualbox-6-0-and-wsl-working-at-the-same-time/58478688)

[^hyper_v_wsl2_2]:[unable-to-boot-windows-10-guest-using-virtualbox-with-enabled-hyper-v-and-wsl2-i](https://stackoverflow.com/questions/62184881/unable-to-boot-windows-10-guest-using-virtualbox-with-enabled-hyper-v-and-wsl2-i)

[^hyper_v_wsl2_3]: [how-to-disable-hyper-v-in-command-line](https://stackoverflow.com/questions/30496116/how-to-disable-hyper-v-in-command-line)

[WSL2_offical_installation_turorial]: https://docs.microsoft.com/fr-fr/windows/wsl/install

[docker_official_site]: https://www.docker.com)

[Windows_terminal_official_customize]: https://docs.microsoft.com/zh-cn/windows/terminal/customize-settings/startup
