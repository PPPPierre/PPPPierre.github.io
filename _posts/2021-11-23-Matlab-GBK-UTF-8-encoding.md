---
title: Matlab - 解决 MATLAB 打开 .m 文件后注释乱码的问题
author: Stone SHI
date: 2021-11-23 10:21:00 +0200
categories: [Blogging, Matlab]
tags: [Matlab]
---

MATLAB高版本(2017a以上)存在的两种编码格式
编码格式修改(GBK改为UTF-8)

# MATLAB高版本(2017a以上)存在的两种编码格式
GBK和UTF-8。编码格式修改后可能会导致现有m文件乱码问题解决，但原有注释乱码的问题。

# 编码格式修改(GBK改为UTF-8)
首先，命令窗口输入：slCharacterEncoding()；用于查看现在的MATLAB编码格式，将

GBK改为UTF-8步骤如下：

一、打开MATLAB安装路径下的D:\Matlab2018b\bin的lcdata_utf8.xml文件；

二、找到以下代码段，删除

```xml
    <encoding name="GBK">
        <encoding_alias name="936"/>
    </encoding>
```

三、找到以下代码段：

```xml
    <encoding name="UTF-8">
        <encoding_alias name="utf8"/>
    </encoding>
```

在第二、三行之间插入一句：`<encoding_alias name="GBK"/>`

四、文件另存为：lcdata.xml（若bin目录下原有该文件，可先将其另外重命名，因为在将

UTF-8改为GBK时可能会用到）

五、重启MATLAB即可，此时输入slCharacterEncoding()；得到ans=‘UTF-8’，修改完毕。

若将UTF-8改为GBK，反向执行上述操作即可。

注意：

1、将UTF-8改为GBK后可能导致原有m文件中的备注乱码，但英文备注一般不会乱码；

2、修改XML文件时注意输入格式(特别是双引号输入法格式)，否则会导致程序无法运行；

2、一般，UTF-8应用更为广泛，建议改为该编码格式。

# 参考：
[解决MATLAB2018b打开m文件后注释乱码的问题](https://blog.csdn.net/weixin_43973546/article/details/108398485)
