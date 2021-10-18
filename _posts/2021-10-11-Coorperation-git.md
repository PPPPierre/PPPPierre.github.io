---
title: 【GIT】 使用 Git 进行团队开发
author: Stone SHI
date: 2021-10-11 15:14:00 +0200
categories: [Blogging, Study]
tags: [Git]
---

# Git

进入自己的分支，保证分支在最新的版本

```sh
$ git pull --rebase origin main
```

解决完冲突之后，更新线上的分支

```sh
$ git push --force
```

然后在 Github 网站上进行 Merge
