---
title: LeetCode 刷题记录 - 每日一题
author: Stone SHI
date: 2022-01-11 15:01:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode,]
---

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

# 1036. 逃离大迷宫

## 题干

在一个 $10^6 × 10^6$ 的网格中，每个网格上方格的坐标为`(x, y)`。

现在从源方格`source = [sx, sy]`开始出发，意图赶往目标方格`target = [tx, ty]`。数组`blocked`是封锁的方格列表，其中每个`blocked[i] = [xi, yi]`表示坐标为`(xi, yi)`的方格是禁止通行的。

每次移动，都可以走到网格中在四个方向上相邻的方格，只要该方格**不**在给出的封锁列表`blocked`上。同时，不允许走出网格。

只有在可以通过一系列的移动从源方格`source`到达目标方格`target`时才返回`true`。否则，返回`false`。

## 思路

这道题目的矩阵很大，直接搜索时间肯定会超时，但是有一个突破口，就是`0 <= len(blocked) <= 200`，`block`的数量是在两百个以内的。

有两个思路可供选择，第一个是**压缩矩阵**，就是除去`source`，`target`和所有`block`所在的行列，压缩所有的剩下的行列，这样就可以把问题化解为一个$200 × 200$矩阵内的广度优先搜索。

第二个思路是**广度优先搜索加提前退出**，而提前退出的条件就是利用了`block`的数量：判断`source`和`target`是否被`block`包围。

因为`block`数量有限，所以被`block`包围的格子数量也是有一个上限的，当 BFS 搜索的格子数量超过这个上限时，我们就可以认为`block`没有包围住 BFS 的起点。

而这个最大值是$(N_{block}-1)×N_{block}/2$，对应的这个情况就是所有的`block`作为斜边与两个边界在角落围成一个等腰直角三角形，此时面积最大。

所以一共有三种情况：

1. 起点被包围；

2. 终点被包围；

3. 都没有被包围。

所以根据如上思想写一个函数去判断起点和终点是否都被包围即可。

## 代码

```python
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        # 判断 src 是否被包围
        def is_surrounded(src, dst, max_area):
            # BFS 的队列
            queue = [tuple(src)]
            # 计算走过的格子数
            count = 0
            # visited 数组，这里用字典实现
            visited = dict()
            visited[tuple(src)] = True
            while queue:
                x, y = queue.pop(0)
                count += 1
                if count > max_area:
                    # 当搜索的格子数超过最大可围面积时，我们判断 src 没有被围上
                    return 1
                for dx, dy in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
                    x_next, y_next = x+dx, y+dy
                    if 0 <= x_next <= 999999 and 0 <= y_next <= 999999:
                        if [x_next, y_next] == dst:
                            # 如果找到 dst，直接返回
                            return 2
                        if (x_next, y_next) not in block_dict and (x_next, y_next) not in visited:
                            queue.append((x_next, y_next))
                            visited[(x_next, y_next)] = True
            # 队列提前终止，说明无路可走，被包围
            return 0
        
        # 广搜加提前退出
        if len(blocked) <= 1: 
            return True
        n_block = len(blocked)
        # 计算最大可包围面积
        max_area = (n_block-1)*n_block/2
        # 用字典记录 block，防止重复遍历带来的高计算量
        block_dict = dict()
        for block in blocked:
            block_dict[tuple(block)] = True
        # 首先判断起点是否被包围
        ans = is_surrounded(source, target, max_area)
        # 起点被包围
        if ans == 0: return False
        # 直接找到终点
        elif ans == 2: return True
        # 判断终点情况
        ans = is_surrounded(target, source, max_area)
        # 终点被包围
        if ans == 0: return False
        # 两个点都没被包围
        return True
```

# 747. 至少是其他数字两倍的最大数（简单）

简单题，找到数组中的最大值并确保比其他数字都大两倍以上。

同时找**最大数**的和**次大数**，然后比较最大数和次大数的两倍即可。

```python
class Solution:
    def dominantIndex(self, nums: List[int]) -> int:
        max_num = 0
        max_index = 0
        second_num = 0

        # 遍历数组，更新最大和第二大数字的值
        for i, num in enumerate(nums):
            if num > max_num:
                second_num = max_num
                max_num = num
                max_index = i
            elif num > second_num:
                second_num = num

        # 如果最大值大于等于第二大的数字的两倍则返回下标
        if max_num >= second_num * 2:
            return max_index
        else:
            return -1
```