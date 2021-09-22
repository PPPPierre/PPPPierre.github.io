---
title: \[专业课\]系统编程小项目一 使用 Buildroot 编译 Linux 内核
author: Stone SHI
date: 2021-09-22 23:11:00 +0200
categories: [Blogging, Study]
tags: [System Programming, Buildroot, Linux]
---

2021年9月22日，星期三，专业周第二天，讲操作系统和系统编程。 老师在下午布置了两个小项目，第一个是使用 **Buildroot** 来编译一个 **Linux** 内核，第二个是使用 C 语言编写一个 Linux 的 Mini shell。

这篇 blog 主要讲第一个项目，第二个项目链接在此。

# 操作系统

# Linux 系统

# 项目：使用 Buildroot 来编译一个 Linux 内核

## 1. 环境搭建

具体参考 Buildroot 官方网站的 documentation

如果是 linux， 则只需要下载一个官方的 Buildroot 压缩包然后解压之后照着 documentation 操作就行

如果要在 windows 上操作，则需要首先创建一个 Linux 虚拟环境，或者按照官方的 doc 上说的使用 Vagrant 创建一个虚拟环境。

推荐自己创建一个。

## 2. 设置 make config

首先需要对编译进行设置一下。

在 buildroot 的目录下输入

```sh
$ make menuconfig
```

然后进行一些选项的勾选：

1. `Toolchain` 里面的 `Enable WCHAR` 勾选上
2. `Kernal` 里面的 `Kernal configuration` 选 `Use the architecture default configuration`
3. `Filsystem images` 里面的 `iso image` 勾选上
4. `Bootloaders` 里面的 `grub2` 勾选上（忘记这个是不是默认就勾选的了）

最后输入

```sh
$ make
```

就可以了，等很长一段时间就编译完毕了

## 3. 制作启动盘

将编译完的文件拷到一个U盘里，再在电脑上使用从U盘启动，这个U盘就可以作为一个Linux的启动盘了。

# 参考链接
