---
title: Blog - 博文撰写相关技巧
author: Stone SHI
date: 2021-10-20 2:44:00 +0200
categories: [Blogging, Blog]
tags: [Blog, Jekyll]
---

# 文件存放

`_posts` 需要发布的博客的存放位置

`_drafts` 草稿的存放位置

# 内容编辑

# 页面跳转

## 方法一、跳转到指定位置

需要点击的位置的代码：

```
[跳转文字](#label)
```

目标位置的代码：

```
<div id="label"></div>
```

`label` 可以设置为任意字符串

## 方法二、跳转到指定标题

该方法需要首先确保 markdown 引擎为 kramdown，在 _config.yml 里需要以下代码：

```
markdown: kramdown
```

然后需要点击的位置的代码同上：

```
[跳转文字](#label)
```

然后在需要跳转目标标题的代码后面加上`{#label}`即可：

```
## 1.标题一 {#label}
```

# 图片引用

# 数学公式

虽然在 GitHub Page 上使用 Jekyll 支持 Markdown 语法，但如果要显示公式，则需要借用 MathJax 帮助渲染。

方法：

1. 设置markdown引擎为kramdown，方法为在 _config.yml 里添加：

```
markdown: kramdown
```

2. 在 .md 文件或者博文开始插入以下代码段即可：

```markdown
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
```

参考：
[How to support latex in GitHub-pages?](https://stackoverflow.com/questions/26275645/how-to-support-latex-in-github-pages)

[MathJax Documentation](http://docs.mathjax.org/en/latest/)

## 特殊符号、加帽子符号、横线和波浪线
