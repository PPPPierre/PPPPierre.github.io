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