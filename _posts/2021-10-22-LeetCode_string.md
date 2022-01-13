---
title: LeetCode 刷题记录 - String
author: Stone SHI
date: 2021-10-18 3:40:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, String]
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

# 22. Generate Parentheses

记录一个很牛逼的 Solution

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:

        stack = []
        res = []
        
        def backtrack(open_count, close_count):
            if open_count == close_count == n:
                res.append("".join(stack))
                return
            
            if open_count < n:
                stack.append("(")
                backtrack(open_count + 1, close_count)
                stack.pop()
                
            if close_count < open_count:
                stack.append(")")
                backtrack(open_count, close_count + 1)
                stack.pop()
        
        backtrack(0,0)
        return res
```

# 1044. Longest Duplicate Substring
二分法加滚动哈希