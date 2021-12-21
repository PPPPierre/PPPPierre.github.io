---
title: LeetCode 刷题记录 - Dynamic Programming
author: Stone SHI
date: 2021-10-18 3:40:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, Dynamic Programming]
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

# 32. Longest Valid Parentheses

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        dp = [0] * len(s)
        for i in range(1, len(s)):
            if s[i] == ')':
                if s[i-1] == '(':
                    dp[i] = (dp[i-2] + 2) if i >= 2 else 2
                elif i-dp[i-1]-1 >= 0 and s[i-dp[i-1]-1] == '(':
                    dp[i] = dp[i-1] + 2 + (dp[i-dp[i-1]-2] if i-dp[i-1]-2 >= 0 else 0)
        return max(dp, default=0)
```

本解法中体现的动态规划思路：

假设输入list为`s`, 初始化一个相同长度的全 0 的 list `dp`作为存放结果的表，通过遍历输入`s`对`dp`进行赋值更新。

`dp[i]`的值为当输入为`list[:i+1]`时的 Longest Valid Parentheses，即输出。

因此我们有`dp[0] = 0`。

而当我们找到从`s[i]`、`s[i-1]`以及`dp[i-1]`推出`dp[i]` 的关系时，我们也就可以通过这一递推关系构造整个`dp`数列。

# 44. Wildcard Matching 通配符匹配

当一个问题输入的大小不定时，通常我们需要考虑使用 DP 的思想，尤其是一些涉及到字符串的问题。

在此问题中，当通配符为`*`时，子问题会产生分裂。即当`p[j] == '*'`时，`helper(i, j)`的值取决于`helper(i+1, j)`和`helper(i, j+1)`。此时，需要使用递归来解决的这一产生分支的问题。

而在通配符为任意其他字符时，匹配结果不会产生分支：或不匹配返回`False`，或者取决于`helper(i+1, j+1)`。

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        @lru_cache(None)
        def helper(i, j):
            if i == len(s) and j == len(p): return True
            if i == len(s): return helper(i, j + 1) if p[j] == '*' else False
            if j == len(p): return False
            if p[j] == '?': return helper(i + 1, j + 1)
            if p[j] == '*': return helper(i, j + 1) or helper(i + 1, j)
            if s[i] != p[j]: return False
            return helper(i + 1, j + 1)
        return helper(0, 0)
```

# 45. Jump Game II 

查表法解决。

通过一个 list `jump`记录到达每一个位置`i`所需的最小步数。

使用指针`i`对 list `nums` 进行迭代就可以更新`jump`的值。

指针`j`表示目前已知可以最快到达的位置。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) == 1: return 0
        jump = [0] * len(nums)
        i, j = 0, 0
        while i < len(nums):
            r = i + nums[i]
            if r >= len(nums) - 1: return jump[i] + 1
            if r > j:
                while j < r:
                    j += 1
                    jump[j] = jump[i] + 1
            i += 1
```

# 55. Jump Game
子问题：每个 index 是否是可以到达的
状态转移方程：每个 index + 自身的 jump 数量决定了该 index 往后 jump 数量的 index 的可否到达属性。
重复子问题：每个 index 的可否到达属性被重复计算

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        
        result = [False] * len(nums)
        result[0] = True
        
        cur_index = 0
        max_reach = 0
        
        while cur_index < len(nums) and result[cur_index]:
            
            cur_reach = cur_index + nums[cur_index] + 1
            if cur_reach >= len(nums): return True
            if cur_reach > max_reach:
                for i in range(max_reach, cur_reach):
                    result[i] = True
            max_reach = cur_reach
            
            cur_index += 1
                
        return False
```

# 62. Unique Paths

子问题：到达每一个位置的路径数量
状态转移方程：每一个位置的路径数量等于其上边位置和左边位置的路径数量的和。
重复子问题：无

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        
        result = [[0]*n]*m
        result[0][0] = 1
        
        for i in range(m):
            result[i][0] = 1
        
        for j in range(n):
            result[0][j] = 1
        
        for i in range(1, m):
            for j in range(1, n):
                result[i][j] = result[i-1][j] + result[i][j-1]
                
        return result[-1][-1]
```

# 63. Unique Paths II

# 64. Minimum Path Sum

# 72. Edit Distance

关于思路，这篇[编辑距离](https://github.com/labuladong/fucking-algorithm/blob/master/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%B3%BB%E5%88%97/%E7%BC%96%E8%BE%91%E8%B7%9D%E7%A6%BB.md)文章已经讲的很清楚了，下面直接整理几个关键。

## a. 子问题

假设子字符串 `word1[:i]`, `word2[:j]`之间的最短编辑距离记为 `DP(i, j)`。那么求出所有 `i`, `j` 对应的 `DP(i, j)` 则构成子问题的集合。

## b. 状态转移方程的确定：

`DP(i, j)` 可以从 `DP(i-1, j)`, `DP(i, j-1)` 和 `DP(i-1, j-1)` 三个子问题的答案推出。

但是需要根据  `word1[i]` 和 `word2[j]` 的关系来进行分类讨论：

如果 `word1[i] == word2[j]`，因为对应字母相同，不需要任何操作。

对应代码为：

```python
DP[i][j] = DP(i-1, j-1)
```

如果 `word1[i] != word2[j]`，则对应字母不同，则需要进行操作，三个操作前状态分别对应三个操作：

`DP(i-1, j) -> DP(i, j)` 对应已知从 `word1[:i-1]` 转换到 `word2[:j]` 的最小编辑距离再加上从 `word1[:i]` 到 `word1[:i-1]` 的一步删除操作

`DP(i, j-1) -> DP(i, j)` 对应已知从 `word1[:i]` 转换到 `word2[:j-1]` 的最小编辑距离再加上从 `word2[:j-1]` 到 `word2[:j]` 的一步插入操作

`DP(i-1, j-1) -> DP(i, j)` 对应已知从 `word1[:i-1]` 转换到 `word2[:j-1]` 的最小编辑距离再加上从 `word1[:i]` 转换到 `word2[:j]` 最后一位的替换操作

最后取这三种操作中编辑距离最下的情况，代码则是：

```python
DP[i][j] = min(res[i - 1][j], res[i][j - 1], res[i-1][j-1]) + 1
```

总体代码如下：

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        
        m = len(word1)+1
        n = len(word2)+1
        
        res = [[0]*n for i in range(m)]
        
        # 初始化 DP table
        res[0][0] = 0
        for i in range(1, m):
            res[i][0] = i
        for j in range(1, n):
            res[0][j] = j
            
        for i in range(1, m):
            for j in range(1, n):

                # 状态转移方程
                if word1[i-1] == word2[j-1]:
                    res[i][j] = res[i-1][j-1]
                else:
                    res[i][j] = min(res[i - 1][j], 
                                    res[i][j - 1], 
                                    res[i-1][j-1]) + 1
                    
        return res[m-1][n-1]
```

# 84. Largest Rectangle in Histogram

关键结构：单调栈

[Leetcode 84：柱状图中最大的矩形（超详细的解法！！！）](https://blog.csdn.net/qq_17550379/article/details/85093224)

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        
        # 递增栈
        n = len(heights)
        mono_stack = []
        max_area, i = 0 ,0
        
        while i < n:
            if not mono_stack or heights[i] >= heights[mono_stack[-1]]:
                mono_stack.append(i)
                i += 1
            else:
                j = mono_stack.pop()
                max_area = max(max_area, (i - mono_stack[-1] - 1 if mono_stack else i)*heights[j])
            
        while mono_stack:
            j = mono_stack.pop()
            max_area = max(max_area, (i - mono_stack[-1] - 1 if mono_stack else i)*heights[j])
            
        return max_area
```

# 85. Maximal Rectangle

该问题可以按照行为单位拆解为若干子问题，第`i`行对应的子问题是求解一道以`matrix[i][:]`这一行为底，以`i+1`为最大柱子高度的 84 题。

创建 dp table 以构建每一个子问题的输入，可以得出`dp[i][j]`就是从`dp[0][j]`一直到`dp[i][j]`累计的柱子长度。

然后按照 84 题的构建单调栈的思路来解决每一个子问题即可。

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        
        rows, cols = len(matrix), len(matrix[0])
        # initial dp table
        dp = [[0] * cols for i in range(rows)]
        # Initial mono-stack
        monostack = list()
        res = 0
        
        for i in range(rows):
            for j in range(cols):
                
                # 状态转移方程
                if matrix[i][j] == '1':
                    dp[i][j] = dp[i-1][j] + 1 if i > 0 else 1
                else:
                    dp[i][j] = 0
                
                # 按照 84 题思路，逐个处理“柱子高度”
                while(monostack and dp[i][monostack[-1]] > dp[i][j]):
                    k = monostack.pop()
                    res = max(res, dp[i][k] * (j - monostack[-1] - 1 if monostack else j))
                monostack.append(j)
            
            # 逐个弹出单调栈中剩余的“柱子高度”并计算最大面积
            while(monostack):
                k = monostack.pop()
                res = max(res, dp[i][k] * (cols - monostack[-1] - 1 if monostack else cols))

        return res
            
```
# 300. Longest Increasing Subsequence (LIS)

方法一：DP

在这里，如何定义 DP 表内元素的意义是关键。

定义 DP 表内元素的意义 = 定义子问题

如何定义子问题很重要，子问题的定义如同桥梁，需要照顾两个方面：

1. 要足够具体（易于实现），保证可以快速找到状态转移方程；

2. 要足够贴近全局问题，使得从所有子问题的解中可以快速推导出全局问题的解。

以该题为例，将`DP[i]`定义为**以第 i 个元素为结尾的 LIS 的长度。**

那么状态转移方程就可以通过第 i 位与 从 0 到 i-1 位的大小关系来得到：

$DP[i] = Max_{0 <= k <= i-1, N[k] < N[i]}(DP[k]) + 1$

而全局问题的解就是 DP 表里所有子问题的最大值。

该解法的时间复杂度为 $O(N^2)$。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0]*n
        dp[0] = 1
        
        for i in range(1,n):
            lis = 1
            for j in range(i):
                if nums[j] < nums[i]:
                    lis = max(lis, dp[j] + 1)
            dp[i] = lis
            
        return max(dp)
```

方法二：二分法

需要数学证明的方法。通过数学证明，找到了一个与解决该问题等价的操作方法，来源于一种扑克牌游戏。

在该方法中通过应用二分法可以进一步将时间复杂度压缩到 $O(Nlog(N))$。具体请移步参考链接。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        deck = [nums[0]]
        
        for k in range(1,n):
            if nums[k] > deck[-1]:
                deck.append(nums[k])
            else:
                i = 0
                j = len(deck) - 1
                while (j > i):
                    middle = int((i + j)/2)
                    if deck[middle] < nums[k]:
                        i = middle + 1
                    else:
                        j = middle
                deck[i] = nums[k]
                
        return len(deck)
```

参考链接：
[动态规划设计：最长递增子序列](https://github.com/labuladong/fucking-algorithm/blob/master/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%B3%BB%E5%88%97/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E8%AE%BE%E8%AE%A1%EF%BC%9A%E6%9C%80%E9%95%BF%E9%80%92%E5%A2%9E%E5%AD%90%E5%BA%8F%E5%88%97.md)