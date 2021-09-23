---
title: 【专业课】 系统编程小项目二 - 使用 C 语言编写一个 Linux 的 Mini shell
author: Stone SHI
date: 2021-09-22 22:58:00 +0200
categories: [Blogging, Study]
tags: [Functional Programming, C, Linux]
---

2021年9月22日，星期三，专业周第二天，讲操作系统和系统编程。 老师在下午布置了两个小项目，第一个是使用 **Buildroot** 来编译一个 **Linux** 内核，第二个是使用 **C** 语言编写一个 **Linux** 的 Mini shell。

这篇 blog 主要讲第二个项目，第一个项目链接在此。

# Shell

# 项目：使用 C 语言编写一个 Linux 的 Mini shell

## 1. 环境搭建

直接在一个 Linux 的环境里使用 C 语言编写就行了。

## 2. 代码讲解

首先导入需要的头文件

```C
#include <sys/types.h>
#include <unistd.h>
```

`sys/types.h` 是 Unix/Linux 系统的基本系统数据类型的头文件，含有size_t，time_t，pid_t等类型。

`unistd.h` 是 unix std 的意思，是POSIX标准定义的unix类系统定义符号常量的头文件，包含了许多UNIX系统服务的函数原型。unistd.h 在 unix 中类似于 windows 中的 windows.h。

# 参考链接
