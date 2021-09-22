---
title: <专业课> 系统编程小项目二 - 使用 C 语言编写一个 Linux 的 Mini shell
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

需要 include 的关键包是这个 `unistd` 包

```C
#include <unistd.h>
```

# 参考链接
