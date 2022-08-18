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

# 2022.08.18 - 剑指 Offer II 004. 只出现一次的数字 

## 题干

给你一个整数数组 `nums` ，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次** 。请你找出并返回那个只出现了一次的元素。

## 思路

最简单的字典思路就不说了，直接上位运算的方法。

因为题目中给的所有的数字都是用 int32 来存储的，因此可以把每一个数字按照二进制位拆开来看。

当我们对所有数字按照二进制位进行求和的时候，唯一数字二进制位为 0 的位数的和必定为 3 或者为 为0.

那么就可以通过这个方法，筛选出唯一数字为 1 的所有二进制位，再还原。

唯一的问题就是编码问题，因为最后一位（31位）在有的语言中是补码意义上的反码，所以要另外判断（比如在 Python 中）。

## 代码

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        print(-5 >> 31)
        ans = 0
        for i in range(32):
            total = sum((num >> i) & 1 for num in nums)
            if total % 3:
                # Python 这里对于最高位需要特殊判断
                if i == 31:
                    ans -= (1 << i)
                else:
                    ans |= (1 << i)
        return ans
```

## 进阶思路

请参考 [剑指 Offer II 004. 只出现一次的数字](https://leetcode.cn/problems/WGki4K/) 题解。