---
title: 【LeetCode】 LeetCode 刷题记录 - Array
author: Stone SHI
date: 2021-10-18 3:40:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, array]
---

# 15. 3Sum

首先排序输入数列

由于数列是排序的，所以使用**双指针**即可。总体时间复杂度 O(N^2)

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) <= 2:
            return []
        nums.sort()
        nums_neg = []
        nums_pos = []
        count_0 = 0
        
        for i, num in enumerate(nums):
            if num > 0:
                nums_pos.append(num)
            elif num == 0:
                count_0 += 1
            else: 
                nums_neg.append(num)
        
        res = []
        
        if count_0 >= 1:
            nums_pos = [0] + nums_pos
        
        if count_0 >= 3:
            res.append([0,0,0])
            
        nums_neg.sort(reverse=True)
        
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