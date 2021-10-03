---
title: 【专业课】 系统编程项目一 - 使用 Buildroot 编译 Linux 内核
author: Stone SHI
date: 2021-09-22 23:11:00 +0200
categories: [Blogging, Study]
tags: [System Programming, Buildroot, Linux]
---

2021年9月22日，星期三，专业周第二天，讲操作系统和系统编程。 老师在下午布置了两个小项目，第一个是使用 **Buildroot** 来编译一个 **Linux** 内核，第二个是使用 C 语言编写一个 Linux 的 Mini shell。

这篇 blog 主要讲第一个项目，第二个项目链接在此。

# 操作系统

# Linux 系统

# Buildroot

这个东西的意义在于帮助你制作一个 Linux 镜像，并且直接在里面以包的形式加上自己的软件，然后就可以将这个镜像发布出去。

# 项目：使用 Buildroot 来编译一个 Linux 内核

首先给一个 x86 的处理器编译一个 Linux 内核，然后为一个 RAM 处理器（比如树莓派）编译内核，由于处理器不同，两个编译的 Linux 内核也不同。

## 1. 环境搭建

具体参考 Buildroot 官方网站的 documentation

如果是 linux， 则只需要下载一个官方的 Buildroot 压缩包然后解压之后照着 documentation 操作就行

如果要在 windows 上操作，则需要首先创建一个 Linux 虚拟环境，或者按照官方的 doc 上说的使用 Vagrant 创建一个虚拟环境。

推荐自己创建一个。

## 2. 设置 make config

如果要自定义一个内核，那就需要对编译进行设置一下。

在 buildroot 的目录下输入

```sh
$ make menuconfig
```

然后就可以根据官方的文档对这些项目进行更改了。

但是我们这里需要首先用官方给的 x86 标准设定。

进入 build-x64 文件夹之后

设定

```sh
$ make menuconfig
```

然后就可以退出来

输入

```sh
$ make
```

进行等待就行了

## 3. 在编译的 Linux 镜像里加入自己的包

首先把自己的包的源文件加入到 build_root 文件夹的 Package 目录里

然后 为其编写两个 文件 一个是 `Config.in` 文件用于把我们的包加入到 `menuconfig` 的选项里
然后 `xxx.mk` 文件告诉 Buildroot 怎么编译。

然后进入 build-x64 文件夹里，输入 

```sh
make menuconfig
```

在 game 里选中 package

然后

```sh
make
```

## 4. 制作启动盘

将编译完的文件拷到一个U盘里，再在电脑上使用从U盘启动，这个U盘就可以作为一个Linux的启动盘了。

# 参考链接
