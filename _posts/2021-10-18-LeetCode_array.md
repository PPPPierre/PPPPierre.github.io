---
title: LeetCode 刷题记录 - Array
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

# 33. 搜索旋转排序数组

整数数组 `nums` 按升序排列，数组中的值**互不相同**。

在传递给函数之前，`nums` 在预先未知的某个下标 `k` （ `0 <= k < nums.length` ）上进行了**旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` （下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标`3`处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你**旋转后**的数组 `nums` 和一个整数 `target` ，如果 nums 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

## 思路

提示说时间要求$O(logN)$，那肯定想到用**二分法**。

首先的思路是先用一遍二分法找到旋转的下标 `k` ，然后复原数组之后再重新用一次二分法。

但是经过仔细思考之后发现，其实只需要一次二分法也能定位 `target`，只是在每一次决定取哪半边的时候需要更加复杂的判断。

因为**旋转**数组从下标 `k` 开始被分割成左右两个递增序列，且左边序列的最小值大于右边序列的最大值。

因此在使用二分法时， 通过比较左右两指针值的大小，可以得出两指针是否在一个递增序列中，

如果在一个序列中，就变成了常规的二分法；

如果不是，则讨论**中点**落在哪一个递增序列里，然后结合 `target` 的位置决定取左边还是取右边。

## 遇到的问题

二分法最容易碰到的就是**无限循环**的问题。

我们知道当 `left` 和 `right` 只差 `1` 时，算出来的 `middle` 就等于左指针的位置。

而此时当你的代码在后续的左右指针变动中存在 `left = middle` 而不是 `left = middle + 1` 时，就存在无限循环的风险，所以需要仔细筛查。

当更新语句中赋的值为 `middle + 1` 时，该语句所在的判断情况应当对应 `if target > nums[middle]` 或者 `if target < nums[middle]`，

因为只有百分百确定 `middle` 位置的值不等于 `target` 时，我们才可以在更新时直接跳过 `middle` ,从 `middle + 1` 开始新的二分。


## 参考代码

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 直接使用二分法， 只是判断的情况比较复杂
        l, r = 0, len(nums) - 1
        while l < r:
            # 计算中点
            mid = (l + r) // 2
            # 首先对根据左右指针的值进行情况分类
            if nums[l] < nums[r]:
                # l 和 r 都在一个递增序列里
                # 常规二分法解决
                if target > nums[mid]:
                    l = mid + 1
                else:
                    r = mid
            else:
                # l 和 r 分别在两个序列里
                # 首先对中点的位置分类讨论
                if nums[mid] >= nums[l]:
                    # 中点在左边的递增区间里
                    # 这里左边界要取到，否则
                    if nums[l] <= target <= nums[mid]:
                        r = mid
                    else:
                        l = mid + 1
                else:
                    # 中点在右边的递增区间里
                    # 这里右边界要取到
                    if nums[mid] < target <= nums[r]:
                        l = mid + 1
                    else:
                        r = mid

        # 循环结束后 l == r
        if nums[l] == target:
            return l
        else:
            return -1
```

# 34. 在排序数组中查找元素的第一个和最后一个位置

也是一道用二分法的数组题。题目中给出的 `nums` 数组中的数字是单调递增的，但是数字会有重复。

要求给出 `target` 的左右边界。

思路如下：

首先用传统二分法找到第一个 `target`，然后以这个 `target` 把数组分成两部分，左右分别再用二分法寻找左右边界即可。

最后补充代码处理没找到的情况以及特殊情况即可。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return (-1, -1)
        
        # 二分法
        l , r = 0, len(nums) - 1
        found = False

        # 目标是左右指针的值都是 target
        # 或左指针等于右指针
        while (nums[l] != target or nums[r] != target) and l < r:
            # 中点指针
            mid = (l + r) // 2
            if nums[mid] > target:
                r = mid - 1
            elif nums[mid] < target:
                l = mid + 1
            else:
                # 当中点值等于 target 时，保存并跳出循环
                first_target = mid
                found = True
                break
        
        # 如果没找到 target，则左右指针一定相等
        if not found:
            if nums[l] != target:
                # 如果左指针的值不是 target，则没找到，返回 (-1, -1)
                return (-1, -1)
            else:
                # 如果是，则说明左右指针值均指向唯一的 target
                return (l, r)

        # 把边界保存下来
        left, right = l, r

        # 找到了第一个 target 之后以这个 target 为界，分别用二分法寻找左右边界
        # 寻找左边界
        r = first_target
        while (nums[left] != target) and left < r:
            # 中点指针
            mid = (left + r) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                r = mid
        
        # 寻找右边界
        l = first_target
        while (nums[right] != target) and l < right:
            # 中点指针
            mid = (l + right) // 2 + 1
            if nums[mid] > target:
                right = mid - 1
            else:
                l = mid

        # 返回左右边界
        return (left, right)

```