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

# 394. 字符串解码

碰到这种字符串类的题目，首先用脑子遍历一遍，如果过程中没有分叉，每遍历到一个字符得到的过渡答案是唯一的，那么就可以直接通过遍历分类讨论的方式解决。

如果碰到括号这种有优先级的符号，就结合 **Stack** 来解决。

这道题就是这样，

首先遍历，发现直接过一遍就能推算出结果。

当遇到左括号的时候需要优先计算括号内的内容，那么就将括号前的字符串中间结果和数字存储起来，等遍历到右括号的时候弹出再计算就可以了。

代码如下：

```python
class Solution:
    def decodeString(self, s: str) -> str:
        
        stack = []
        num = 0
        res = ""

        # 遍历字符串
        for c in s:
            if c.isdigit():
                num = num*10 + int(c)
            elif c == "[":
                stack.append((res, num))
                res = ""
                num = 0
            elif c == "]":
                top = stack.pop()
                res = top[0] + res * top[1]
            else:
                res += c
        
        return res
```