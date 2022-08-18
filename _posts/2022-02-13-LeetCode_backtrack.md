---
title: LeetCode 刷题记录 - 回溯算法
author: Stone SHI
date: 2022-02-13 11:37:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, Backtrack]
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

# 回溯算法

## 46. 全排列

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        # 当前路径
        path = []
        # 合法路径集合
        ans = []
        # 记录访问过的节点
        visited = set()
        def trackback(nums, i):
            # 将当前节点加入路径
            path.append(nums[i])
            # 标记当前节点不可用
            visited.add(i)
            
            if len(path) == len(nums):
                # 满足返回条件返回路径
                ans.append(path.copy())
            else:
                # 未达到返回条件则寻找下一个合法节点
                for j in range(len(nums)):
                    if j not in visited:
                        # 只要节点未出现在路径里即合法
                        trackback(nums, j)
            # 将当前节点撤出路径
            path.pop()
            # 恢复当前节点可用性
            visited.remove(i)
        # 以每个节点为起点调用一遍回溯函数
        for i in range(len(nums)):
            trackback(nums, i)
        return ans
```

## 47. 全排列 II

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # 当前路径
        path = []
        # 合法路径集合
        ans = []
        # 记录访问过的节点
        visited = set()
        def trackback(nums, i):
            # 将当前节点加入路径
            path.append(nums[i])
            # 标记当前节点不可用
            visited.add(i)
            # 相比 全排列 I 还需要记录数字的重复
            repeat = set()
            if len(path) == len(nums):
                # 满足返回条件返回路径
                ans.append(path.copy())
            else:
                # 未达到返回条件则寻找下一个合法节点
                for j in range(len(nums)):
                    if j != i and j not in visited and nums[j] not in repeat:
                        # 合法节点条件
                        # 1. 前面未访问过
                        # 2. 数字未重复
                        repeat.add(nums[j])
                        trackback(nums, j)
            # 将当前节点撤出路径
            path.pop()
            # 恢复当前节点可用性
            visited.remove(i)
        # 确保不重复
        # 每个节点作为起点调用一遍回溯函数
        repeat = set()
        for i in range(len(nums)):
            if nums[i] not in repeat:
                repeat.add(nums[i])
                trackback(nums, i)
        return ans
```

## 51. N 皇后

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # 使用 valid_table 来标记行、列、对角线和反对角线上是否有皇后
        # valid_table[0]: row
        # valid_table[1]: col
        # valid_table[2]: diag
        # valid_table[3]: antidiag
        valid_table = [defaultdict(lambda: True) for _ in range(4)]
        ans = []
        path = []
        def trackback(valid_table, i, j, n):
            # 检测该位置是否可以放置皇后
            if valid_table[0][i] and valid_table[1][j] and valid_table[2][i+j] and valid_table[3][i-j]:
                # 生成代表该行的字符串
                row = '.' * j + 'Q' + '.' * (n-j-1)
                # 将该行的选择加入路径
                path.append(row)
                # 标记各个行列对角反对角线的不可用
                valid_table[0][i] = False
                valid_table[1][j] = False
                valid_table[2][i+j] = False
                valid_table[3][i-j] = False
                if i == n-1:
                    # 如果是最后一行，则将路径加入结果
                    ans.append(path.copy())
                else:
                    # 否则遍历所有下一行的选择
                    for k in range(n):
                        trackback(valid_table, i+1, k, n)
                # 遍历完分支回溯
                path.pop()
                # 重新开放行列对角反对角线的可用性
                valid_table[0][i] = True
                valid_table[1][j] = True
                valid_table[2][i+j] = True
                valid_table[3][i-j] = True

        # 遍历第一行每一列作为起点调用回溯算法
        for j in range(n):
            trackback(valid_table, 0, j, n)
        
        return ans
```

## 39. 组合总和

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res = []
        path = []
        n = len(candidates)
        def rec(i, curr_sum):
            if curr_sum > target or i == n:
                return
            if curr_sum == target:
                res.append(path.copy())
                return
            rec(i + 1, curr_sum)
            path.append(candidates[i])
            rec(i, curr_sum + candidates[i])
            path.pop()
        rec(0, 0)
        return res
```

## 40. 组合总和 II

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        path = []
        res = []
        curr_sum = 0
        n = len(candidates)
        def back_track(i, curr_sum):
            num = candidates[i]
            path.append(num)
            curr_sum += num
            visited = set()
            if curr_sum == target:
                res.append(path.copy())
            elif curr_sum < target:
                for next in range(i + 1, n):
                    if candidates[next] not in visited:
                        back_track(next, curr_sum)
                        visited.add(candidates[next])
            path.pop()
        visited = set()
        for i in range(n):
            if candidates[i] not in visited:
                back_track(i, 0)
                visited.add(candidates[i])
        return res
```