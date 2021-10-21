---
title: 【LeetCode】 LeetCode 刷题记录 - Array
author: Stone SHI
date: 2021-10-18 3:40:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, array]
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

# 15. 3Sum

首先将数列排序，然后分为负数和非负数两个 list，同时记录下 0 出现的数量进行分类讨论。

对情况进行分类讨论，可以发现 result 里的 3 个数只有可能是 1 个负数加两个非负数或者 2 个负数加 1 个非负数的情况。

由于 list 是有序的，因此对一个 list 遍历，另一个 list 使用**双指针**即可。总体时间复杂度为 $O(N^2)$。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) <= 2:
            return []
        nums.sort()
        nums_neg = []
        nums_pos = []
        count_0 = 0
        
        # 区分正负
        for i, num in enumerate(nums):
            if num > 0:
                nums_pos.append(num)
            elif num == 0:
                count_0 += 1
            else: 
                nums_neg.append(num)
        nums_neg.sort(reverse=True)
        if count_0 >= 1:
            nums_pos = [0] + nums_pos
        
        if count_0 >= 3:
            res.append([0,0,0])
        
        # 开始遍历
        res = []

        # 1 负 + 2 非负
        if len(nums_pos) >= 2:
            for k,num_n in enumerate(nums_neg):
                i = 0
                j = len(nums_pos) - 1
                if k > 0 and num_n == nums_neg[k - 1]:
                    continue
                rest = - num_n
                while i < j:
                    if nums_pos[i] + nums_pos[j] == rest:
                        res.append([num_n, nums_pos[i], nums_pos[j]])
                        i += 1
                        while i < j and nums_pos[i] == nums_pos[i - 1]:
                            i += 1
                    elif nums_pos[i] + nums_pos[j] < rest:
                        i += 1
                    else:
                        j -= 1
                    
        # 2 负 + 1 非负           
        if len(nums_neg) >= 2:
            for k,num_p in enumerate(nums_pos):
                i = 0
                j = len(nums_neg) - 1
                if k > 0 and num_p == nums_pos[k - 1]:
                    continue
                rest = - num_p
                while i < j:
                    if nums_neg[i] + nums_neg[j] == rest:
                        res.append([num_p, nums_neg[i], nums_neg[j]])
                        i += 1
                        while i < j and nums_neg[i] == nums_neg[i - 1]:
                            i += 1
                    elif nums_neg[i] + nums_neg[j] < rest:
                        j -= 1
                    else:
                        i += 1
        return res
```

当然此处代码还并非最优，可以继续改进。

# 18. 4Sum

思路同上一题，将输入的数列排序然后分正负（ 0 包含在正数数列）。

对情况进行分类讨论，可以分为 0 + 4 （全正或者全负）, 1 + 3, 2 + 2 三种情况。

每个情况都可以将问题化简为额外遍历加 TwoSum 或者 ThreeSum 的情况解决。这里 ThreeSum 的实现用了更简单的方案，但依旧是**双指针**。

总体复杂度 $O(N^3)$。

```python
def twoSum(nums: List[int], target: int, add_list: List[int]) -> List[List[int]]:
    nums.sort()
    res = []
    
    if len(nums) <= 1:
        return []
    
    l = 0
    r = len(nums) - 1

    while l < r:
        current = nums[l] + nums[r]
        if current == target:
            res.append([nums[l], nums[r]] + add_list)
            l += 1
            while l < r:
                if l > 0 and nums[l] == nums[l-1]:
                    l += 1
                else:
                    break
        elif current < target:
            l += 1
        else:
            r -= 1
    return res

def threeSum(nums: List[int], target: int, add_list: List[int]) -> List[List[int]]:
    nums.sort()
    res = []
    
    if len(nums) <= 2:
        return []
    
    for i, num in enumerate(nums):
        if i > 0 and num == nums[i - 1]:
            continue
        l = i + 1
        r = len(nums) - 1
        while l < r:
            current = num + nums[l] + nums[r]
            if current == target:
                res.append([num, nums[l], nums[r]] + add_list)
                l += 1
                while l < r:
                    if l > 0 and nums[l] == nums[l-1]:
                        l += 1
                    else:
                        break
            elif current < target:
                l += 1
            else:
                r -= 1
    return res

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        
        nums.sort()
        nums_pos = []
        nums_neg = []
        for i, num in enumerate(nums):
            if num >= 0:
                nums_pos.append(num)
            else:
                nums_neg.append(num)
        res = []
            
        # 0 + 4
        if target >= 0:
            for i, num in enumerate(nums_pos):
                if i > 0 and num == nums_pos[i-1]:
                    continue
                res = res + threeSum(nums_pos[i+1:], target - num, [num])
        else:
            for i, num in enumerate(nums_neg):
                if i > 0 and num == nums_neg[i-1]:
                    continue
                res = res + threeSum(nums_neg[i+1:], target - num, [num])
          
        # 1 + 3
        for i, num in enumerate(nums_pos):
            if i > 0 and num == nums_pos[i-1]:
                continue
            target_local = target - num
            if target_local < 0:
                res = res + threeSum(nums_neg, target_local, [num])
        for i, num in enumerate(nums_neg):
            if i > 0 and num == nums_neg[i-1]:
                continue
            target_local = target - num
            if target_local >= 0:
                res = res + threeSum(nums_pos, target_local, [num])
        
        # 2 + 2 
        for i, num in enumerate(nums_neg):
            if i > 0 and num == nums_neg[i - 1]:
                continue
            j = i + 1
            while j < len(nums_neg):
                target_local = target - nums_neg[i] - nums_neg[j]
                if target_local >= 0:
                    res = res + twoSum(nums_pos, target_local, [nums_neg[i], nums_neg[j]])
                j += 1
                while j < len(nums_neg):
                    if nums_neg[j] == nums_neg[j-1]:
                        j += 1 
                    else:
                        break
    
        return res
```

# 31. Next Permutation

这道题比较有意思，首先拿几个简单例子进行穷举，观察每一个数和下一个比它大的数之间的关系，就能得出结论。

假设目标数为 $a_1$ 有 $n$ 位，其下一个数为 $a_2$，他们从第 $i$ 位开始不同（$i \geq 0$）。

那么则满足条件：

- $a_1[i + 1] > a_1[i]$

- $\forall \ j \in [i+1, \ n-2] \cap \mathbb{N}, \ a_1[j] \geq a_1[j + 1] $

- $a_2[i] = \mathop{\text{min}} \\{ a_1[j], \ i<j\leq n-1 \ \text{and} \ a_1[j] > a_1[i] \\} $

- $\forall \ j \in [i+1, \ n-2] \cap \mathbb{N}, \ a_2[j] \leq a_2[j + 1] $

根据这四条关系就可以开始编写函数了：

- 按照倒序找到输入数组中最后一个比前一个数大的数，将其前一个数的位置记为 $i$

- 找到从 $i+1$ 位开始比第 $i$ 位大的最小数（候选数相同则选最靠右的），将该数与第 $i$ 位数互换

- 将数组从 $i+1$ 位开始到最后的切片 reverse

结束

```python
def swap(nums: List[int], idx1: int, idx2: int):
    temp = nums[idx1]
    nums[idx1] = nums[idx2]
    nums[idx2] = temp

class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        n = len(nums)

        # 确定进位节点
        i = n - 1
        while i > 0:
            if nums[i] > nums[i-1]:
                break
            else:
                i -= 1
        
        # 判断是否是最大情况
        if i == 0:
            nums.sort()
            return

        # 寻找比交换数
        pre = nums[i-1]
        j = i
        idx_swap = i
        while j <= n - 1:
            if nums[j] > pre and nums[j] <= nums[idx_swap]:
                idx_swap = j
            j += 1

        # 交换
        swap(nums, i-1, idx_swap)

        # reverse 剩余数列
        if i < n - 1:
            k = i
            l = n - 1
            while k < l:
                swap(nums, k, l)
                k += 1
                l -= 1
```