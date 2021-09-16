---
title: 免费搭建一个基于 Jekyll 的博客站点!
author: Stone SHI
date: 2021-09-16 18:00:00 +0200
categories: [Blogging, ]
tags: [Blog, Jekyll]
---

这是该 Blog 自建立起的第一篇 Post ，主要用于记录建立这个站点的整个过程。

# 前言

这几天在研究怎么搭建一个专属的博客站点，一是用来记录自己学习知识和研究技术过程中的点点滴滴，二也是方便自己未来在需要的时候能够从中参考查阅。 其实很早以前就有这么一个想法了，但是一直没有落实。碰巧这段时间在实习之后有时间来完成这件事，同时实习过程中产生的大量的技术积累也需要进行沉淀和整理。最后呢也希望借助这个博客，自己能在未来的技术学习之路上养成一个记录、整理、反思的好习惯。

# 博客搭建

该博客是基于 [**GitHub Pages**][github-pages-homepage]  和 [**Jekyll**][jekyll-homepage] 搭建的。然后使用了 Jekyll 的第三方主题框架 [**Chirpy**][chirpy_homepage] 来进行美化和功能扩展。

## GitHub Pages 和 Jekyll

首先 GitHub Pages 的提供了一个基础的官方教程[^gp-tuto]，从配置一个博客专用的仓库，到把仓库内的静态网页内容交给 GitHub 托管, 再到结合 Jekyll 来自动化生成静态的博客框架，适合用来熟悉 GitHub Pages 以及 Jekyll 的相关知识以及操作。如果要自行搭建博客，建议过一遍这个基础教程。

Jekyll 是一个用 [**Ruby**][ruby-homepage] 语言编写的静态网站框架生成器，可以理解为一个小程序。因此在安装 Jekyll 的时候需要首先安装 Ruby。在这过程中，涉及到许多和 Ruby 编程相关的知识与术语，罗列如下[^ruby-relative]：
- `Ruby`: 一种编程语言
- `Gem`: 封装起来的 `Ruby` 应用程序或代码库，相当于 `Python` 中的 `package` 的概念 
- `Jekyll`: 基于 `Ruby` 编写的一个 `Gem`
- `RubyGems`: `Ruby` 的包管理器，相当于 `Python` 中的 `pip`，可以帮助安装、卸载、升级各种 `Gem`。在终端使用的 `gem` 命令，是指通过 `RubyGems` 管理 `Gem` 包
- `Bundle`: 相当于多个 `RubyGems` 的批处理。在配置文件 `gemfilel` 里说明你的应用依赖哪些第三方包，它会自动帮你下载安装这些包以及这些包依赖的包
- `Gemfile`: 定义你的应用依赖哪些第三方包，`bundle` 根据该配置去寻找这些包

熟悉这些概念之后，便可以结合网上的各种教程来搭建自己的博客，在这里我放一个知乎看到的教程作为例子[^zhihu]以及一个我在 Youtube 看到的非常好的基于 Jekyll 的博客搭建教程[^Jekyll-tutorial]。

## Chirpy

在熟悉搭建博客框架后，便可以上网找各种 Jekyll 主题来美化你的博客。这里我推荐一个 [**Jekyll Themes**][jekyll-themes-homepage][^jekyll-themes-homepage] 的主题网站，上面提供了大量的可供使用的主题。使用主题的教程可以参考我文章底部提供的 Youtube 的教程。各个主题框架的主页面也基本上都会有相关的使用教程。

该博客站点的主题 [**Chirpy**][chirpy-homepage] 就是在我上面提到的网站 Jekyll Themes 找到的。在该主题的 [GitHub 主页][chirpy-homepage][^chirpy-hp]上，开发者也非常耐心地讲解了如何应用该主题。在这个主题的[官方 Demo][chirpy-demo] 页面上也有文章引导使用者如何进行一些自定义的配置。

# 主题框架的应用

这里记录一下在本地部署应用这个主题框架的时候需要注意的一些要点，方便下次需要升级或者更换主题框架的时候参考。

### 部署到 GitHub Pages

本地部署的时候需要首先备份几个文件与文件夹：

- `\_config.yml` 文件里记录着站点的基本信息
- `\assets` 文件夹里存放着站点部署需要的相关资源文件
- `\posts` 文件夹里存放着所有的文章
- `\drafts` 文件夹里存放着所有没有完成的草稿

将这些文件与文件夹备份之后，按照新的主体框架的文档完成搭建，然后把需要的信息和文件资源全部复制回去即可。

### 自定义主页头像

在 `\_config.yml` 文件中可以自定义主页的头像。

只需要把 `avatar:` 后面的链接更换为自己头像的链接即可。

目前我使用的图片上传网站为 [SM.MS][smms-homepage][^smms]，比较方便。

### 自定义网页图标

该主题也支持自定义网页的图标，在 Chirpy 的[官方 Demo][chirpy-demo] 上有一篇文章讲解了如何更换图标[^favicon]，按照文章里的步骤做就可以了。

1. 首先准备一张宽高相等的需要作为网站图标的图片，大小最好在 512 × 512 以上。

2. 登陆这个专门制作图标的网站 [Favicon Generator][favicon-generator-homepage][^f-generator]，将图片上传

3. 然后按照提示逐个设置图标需要的样式，完毕后将打包文件下载下来

4. 解压，删除文件夹中的 `browserconfig.xml` 以及 `site.webmanifest` 两个文件，将剩下的文件复制到`\assets\img\favicons` 目录覆盖即可。

完成后 `push` 到线上仓库即可。

# 参考链接

[^smms]:[SM.MS 图片上传网站][smms-homepage]

[^f-generator]:[Favicon generator][favicon-generator-homepage]

[^gp-tuto]: [GitHub Pages 官方教程](https://docs.github.com/en/pages)

[^ruby-relative]: [整理 Ruby 相关的各种概念](https://henter.me/post/ruby-rvm-gem-rake-bundle-rails.html)

[^jekyll-themes-homepage]: [Jekyll Themes][jekyll-themes-homepage]

[^zhihu]: [知乎: GitHub Pages 建立个人网站详细教程](https://zhuanlan.zhihu.com/p/58229299)

[^Jekyll-tutorial]: [Youtube: Jekyll 搭建个人博客](https://www.youtube.com/playlist?list=PLK2w-tGRdrj7vzX7Y-GqKPb2QPrHCYZY1)

[^chirpy-hp]: [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy/)

[^favicon]: [Chirpy: Customize the Favicon](https://chirpy.cotes.info/posts/customize-the-favicon/)

[github-pages-homepage]: https://pages.github.com/

[ruby-homepage]: https://www.ruby-lang.org/en/

[chirpy-homepage]: https://github.com/cotes2020/jekyll-theme-chirpy/

[chirpy-demo]: https://chirpy.cotes.info/

[jekyll-themes-homepage]: http://jekyllthemes.org/

[smms-homepage]: https://sm.ms/

[favicon-generator-homepage]: https://realfavicongenerator.net/