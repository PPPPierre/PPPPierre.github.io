---
title: 【专业课】 编译课项目二 - 基于 Jflex 词法分析器和 Cup 句法解析器实现编译器前端
author: Stone SHI
date: 2021-09-27 14:16:00 +0200
categories: [Blogging, Study]
tags: [Compilation, Java, Parser]
---

2021年9月30日，周四，今天的编译课程项目：在 Java 环境下基于 Jflex 和 Cup 写一个编译器前端。

由于编译器的前端是自行设计的，所以目标编程语言的语法也就是自定义了，为方便这篇 blog 讨论，就记为 **Mini Pascal**。

我们需要编译的目标代码形似汇编语言，因此这个项目的目的就是让我们更加深刻地理解编译器前端的功能。

目标语言运行的机器使用 Python 模拟的，整个程序运行过程也是直接可以在环境里完成。

# 环境搭建

运行环境是在 wsl2 的 Ubuntu 20.04 LTS 的基础上搭建的 Java 环境。

具体教程参考项目1

# 基于 Jflex 词法分析器和 Cup 句法解析器实现编译器前端

编译遇到问题

```sh
$ make test_mini_pascal.x

./asm.py < test_mini_pascal.i > test_mini_pascal.o
/usr/bin/env: ‘python3\r’: Permission denied
make: *** [Makefile:60: test_mini_pascal.o] Error 126
```

原因是在 windows 环境下创建的文件使用的是 Windows/DOS-style 行尾 （CR+LF），但是 Linux 系统下 Ubuntu 需要的是 Unix-style 行尾 （LF）。

因此下载安装一个文件行尾转换器即可

```sh
$ sudo apt install dos2unix
```

然后转换对应文件

```sh
$ dos2unix *.py

dos2unix: converting file asm.py to Unix format...
dos2unix: converting file mach.py to Unix format...
dos2unix: converting file to_nasm.py to Unix format...
```

https://askubuntu.com/questions/896860/usr-bin-env-python3-r-no-such-file-or-directory

# 参考链接
