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

本解法中体现的动态规划思路：

假设输入 list 为 `s`, 初始化一个相同长度的全 0 list `dp` 作为存放结果的表。

通过遍历输入 `s` 对 `dp` 进行赋值更新。

`dp[i]` 的值为当输入为 `list[:i+1]` 时的 Longest Valid Parentheses，即输出。

因此我们有 `dp[0] = 0`。

而当我们找到 `s[i]`、`s[i-1]` 以及 `dp[i-1]` 推出 `dp[i]` 的关系时，我们也就可以通过这一递推关系构造整个 `dp` 数列。

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