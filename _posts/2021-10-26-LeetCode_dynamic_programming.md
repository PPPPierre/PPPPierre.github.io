---
title: LeetCode 刷题记录 - Dynamic Programming
author: Stone SHI
date: 2021-10-18 3:40:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, Dynamic programming]
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

关于思路，这篇[编辑距离](https://labuladong.gitee.io/algo/3/24/73/)文章已经讲的很清楚了，下面直接整理几个关键。

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
# 300. 最长递增子序列 Longest Increasing Subsequence (LIS) {#problem300}

## 一、动态规划思路一

在这里，如何定义 DP 表内元素的意义是关键。

定义 DP 表内元素的意义 = 定义子问题

如何定义子问题很重要，子问题的定义如同桥梁，需要照顾两个方面：

1. 要足够具体（易于实现），保证可以快速找到状态转移方程；

2. 要足够贴近全局问题，使得从所有子问题的解中可以快速推导出全局问题的解。

思路一，将`DP[i]`定义为**以第 i 个元素为结尾的 LIS 的长度。**

那么状态转移方程就可以通过第 i 位与 从 0 到 i-1 位的大小关系来得到：

$DP[i] = Max_{0 \leq k \leq i-1, N_k < N_i}(DP[k]) + 1$

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

## 二、动态规划思路二 + 二分法

思路二，将`DP[i]`定义为**所有长为 `i+1` 的递增子序列中最后一位的最小值。**

所以 `DP` 数组的初始状态就是 `DP = [nums[0]]`。

然后按顺序遍历数组，更新 `DP`：

1. 当 `nums` 中的元素 `num` 大于 `DP[-1]` 时，说明找到了比当前最长子序列末位更大的数，因此可以将 `num` 添加到 `DP` 末尾。

2. 当 `nums` 中的元素 `num` 小于等于 `DP[-1]` 时，则需要找到 `DP[:-1]` 中最小的大于 `num` 的数，将其更新为 `num`。这里其实蕴含了**贪心**的思想。因为我们在遍历每一个元素时，总是希望该元素之前的递增子序列的末位尽可能地小，这样才能尽可能长地构造递增子序列。而相同长度的子序列只需要保留末位最小的就可以了。

通过简单的反证法可以证明 `DP` 是单调递增的，因此在第二步时可以使用二分法，进一步将时间复杂度压缩到 $O(Nlog(N))$。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i] 表示所有长为 i + 1 的递增子序列中末位的最小值
        dp = [nums[0]]
        for num in nums[1:]:
            if num > dp[-1]:
                # 找到更长的递增子序列
                dp.append(num)
            else:
                # 二分查找需要更新的递增子序列长度
                i, j = 0, len(dp) - 1
                while i < j:
                    mid = (i+j) // 2
                    if num > dp[mid]:
                        i = mid + 1
                    else:
                        j = mid
                dp[i] = num
        # 最终 DP 数组的长度就是最长递增子序列的长度
        return len(dp)
```

参考链接：
[动态规划设计：最长递增子序列](https://labuladong.gitee.io/algo/3/24/74/)

## 追加题：输出最长的上升子序列



# 491. 递增子序列

看似是递增子序列相关，但实际上是一道需要使用**回溯**技巧的题。

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        ans = []
        path = []
        def backtrack(nums, start_index):
            # 将当前位置的数字加入递增子序列
            path.append(nums[start_index])
            # 判断当前递增子序列是否满足条件
            if len(path) > 1:
                ans.append(path.copy())
            # 遍历递增子序列下一个数字的选择
            # 使用 repeated_num 来避免做出重复选择
            repeated_num = set()
            for i in range(start_index+1, len(nums)):
                # 只要是大于当前数字的数都可以作为下一个选择
                if nums[i] >= path[-1] and nums[i] not in repeated_num:
                    repeated_num.add(nums[i])
                    # 调用回溯算法
                    backtrack(nums, i)
            # 将当前位置的数字撤出递增子序列
            path.pop()
        # 每一个数都可以作为递增子序列的起点
        # 使用 visited_num 避免起点数字的重复
        visited_num = set()
        for i, num in enumerate(nums):
            if num not in visited_num:
                visited_num.add(num)
                # 对起点调用回溯算法
                backtrack(nums, i)
        return ans
```

# 334. 递增的三元子序列

面试字节跳动 Resso 业务的时候考到的题目。

使用**最长递增子序列**的两个思路都可以做，但是使用第二个更直观方便。

如果这里的**三元**改为**多元**，那么题目实质上就变成了求**最长递增子序列**。

而维护的数就不是 `num_1`，`num_2` 而是一个数组，如果求**最长递增子序列**一样用二分法来寻找需要替换更新的位置。

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        if n < 3:
            return False
        num_1, num_2 = nums[0], float('inf')
        for num in nums[1:]:
            if num > num_2:
                return True
            elif num > num_1:
                num_2 = num
            else:
                num_1 = num
        return False
```

# 87. Scramble String

借助递归进行穷举，存储中间计算过程减少重复计算。

```python
class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        n = len(s1)
        
        memo = dict()
        
        def dp(i1, j1, i2, j2):
            if (i1, j1, i2, j2) in memo: return memo[(i1, j1, i2, j2)]
            if j1 == i1: 
                memo[(i1, j1, i2, j2)] = s1[i1] == s2[i2]
                return memo[(i1, j1, i2, j2)]
            res = False
            for k in range(j1 - i1):
                res = res or (dp(i1, i1+k, i2, i2+k) and dp(i1+k+1, j1, i2+k+1, j2)) \
                        or (dp(i1, i1+k, j2-k, j2) and dp(i1+k+1, j1, i2, j2-k-1))
            memo[(i1, j1, i2, j2)] = res
            return memo[(i1, j1, i2, j2)]
        
        return dp(0, n-1, 0, n-1)
```

# 87. Super Egg Drop

在这道题里，决定问题状态的有两个变量：楼层数，鸡蛋数。

因此将 DP 数组定义为：`DP[i][j]` 表示**在鸡蛋数量为 j 时确定 i 层所需要的最少操作数**。

在状态转移时进行二分查找选择最优操作，可惜时间要求达不到。

```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        
        # 初始化
        dp = [[0]*k for i in range(n+1)]
        for i in range(n+1):
            dp[i][0] = i
        for j in range(k):
            dp[0][j] = 0
        
        # 状态转移方程
        for i in range(1, n+1):
            for j in range(1, k):
                dp[i][j] = i
                left = 1
                right = i
                
                # # 遍历方法，复杂度O(k*n*n)
                # for l in range(1, i+1):
                #     dp[i][j] = min(dp[i][j], max(dp[l-1][j-1], dp[i-l][j]) + 1)
                
                # 二分查找法，复杂度O(k*n*log(n))
                while left <= right:
                    mid = (left + right) // 2
                    broken = dp[mid-1][j-1] + 1
                    not_broken = dp[i-mid][j] + 1
                    if broken > not_broken:
                        right = mid - 1
                        dp[i][j] = min(dp[i][j], broken)
                    else:
                        left = mid + 1
                        dp[i][j] = min(dp[i][j], not_broken)
                
        return dp[n][k-1]
```

更改 DP 数组定义：

`DP[i][j]` 表示**在鸡蛋数量为 i， 操作数限制为 j 时能确定的最大楼层数**

此时在确定状态转移方程式就不需要再遍历寻找最优操作了。满足题目的时间要求。

```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        dp = [[0]*(n+1) for i in range(k+1)]
        # i: 鸡蛋总数，j: 允许测试次数
        # dp[i][j]: 能够测得的最大楼层数
        
        for i in range(k+1):
            dp[i][0] = 0
        for m in range(n+1):
            dp[0][m] = 0
            
        m = 0
        while dp[k][m] < n:
            m += 1
            for i in range(1, k+1):
                dp[i][m] = dp[i][m-1] + dp[i-1][m-1] + 1
        
        return m
```

参考链接：

[经典动态规划：高楼扔鸡蛋](https://labuladong.gitee.io/algo/3/26/89/)

[经典动态规划：高楼扔鸡蛋（进阶）](https://labuladong.gitee.io/algo/3/26/90/)

# 1143. Longest Common Subsequence

比较传统的解法，在逐个元素比对的时候，要考虑清楚状态的转移过程。

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        
        n1, n2 = len(text1), len(text2)
        
        # 初始化 DP 数组，确定初始状态
        dp = [[0]*n2 for i in range(n1)]
        for i in range(n1):
            if text1[i] == text2[0]: 
                dp[i][0] = 1
            else:
                dp[i][0] = dp[i-1][0]
        for j in range(n2):
            if text2[j] == text1[0]: 
                dp[0][j] = 1
            else:
                dp[0][j] = dp[0][j-1]
                
        # dp[i][j]: text1[:i] 与 text2[:j] 的最长公共子序列
                
        # 状态转移
        for i in range(1, n1):
            for j in range(1, n2):
                if text1[i] == text2[j]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i][j-1], dp[i-1][j])
        
        return dp[-1][-1]
```

参考链接：

[最长公共子序列](https://github.com/labuladong/fucking-algorithm/blob/master/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E7%B3%BB%E5%88%97/%E6%9C%80%E9%95%BF%E5%85%AC%E5%85%B1%E5%AD%90%E5%BA%8F%E5%88%97.md)

[经典动态规划：最长公共子序列](https://labuladong.gitee.io/algo/3/24/77/)

# 1035. 不相交的线

和上一题的思路差不多，但是添加了一个优化方法，可以将两个数组中没有交集的数直接剔除。

因为没有交集的数一定不会有连线，在数据量增大的情况下，这些数据的冗余非常影响速度。

这道题可以用双指针 `i` 和 `j ` 分别从两个数组起点开始遍历的思路来理解，指针只能前进不能后退，那么每次状态转移的自由度为 2，指针 `i` 前进，或者指针 `j` 前进。

而当 `nums1[i] == nums2[j]` 的时候，连线其实是必须的，因为按照规则，无法回头，所以当下连线一定是最好的选择。

那么就可以直接定义一个二维的 dp 数组来进行状态转移，

这里的 dp 表格的定义为： `dp[i][j]` 为数组 `nums[:i+1]` 和数组 `nums[:j+1]` 的最大连线数。

当 `nums1[i] == nums2[j]` 时，我们一定会连线，所以最大连线数等于 `dp[i][j] = dp[i-1][j-1] + 1`，也就是 `nums[:i]` 和 `nums[:j]` 之间的最大连线数加上新连的这根线。

当 `nums1[i] != nums2[j]` 时，我们不连线，所以 `dp[i][j]` 取决于上一个状态，`i` 退一步或者 `j` 退一步的最大连接数，即 `dp[i][j] = max(dp[i][j-1], dp[i-1][j])`。

代码参考如下：

```python
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        # 优化1：取交集，不是交集的不可能有连线，如果冗余过多导致速度很慢
        nums = set(nums1) & set(nums2)

        nums1, nums2 = [i for i in nums1 if i in nums], [j for j in nums2 if j in nums]
        if not nums1 or not nums2:
            return 0

        # 获取数列基本长度信息
        n_row, n_col = len(nums1), len(nums2)

        # 初始化dp矩阵
        dp = [[0] * n_col for _ in range(n_row)]
        if nums1[0] == nums2[0]:
            dp[0][0] = 1

        # 状态转移
        for i in range(1, n_row):
            if nums1[i] != nums2[0]:
                dp[i][0] = dp[i-1][0]
            else:
                dp[i][0] = 1
        for j in range(1, n_col):
            if nums1[0] != nums2[j]:
                dp[0][j] = dp[0][j-1]
            else:
                dp[0][j] = 1
        for i in range(1, n_row):
            for j in range(1, n_col):
                if nums1[i] == nums2[j]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[-1][-1]
```

# 583. Delete Operation for Two Strings

思路同上

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        
        # 初始化 dp 数组
        n1, n2 = len(word1), len(word2)
        dp = [[0]*(n2+1) for i in range(n1+1)]
        for i in range(1, n1+1):
            dp[i][0] = dp[i-1][0] + 1
        for j in range(1, n2+1):
            dp[0][j] = dp[0][j-1] + 1
            
        # dp[i][j]: word1[:i] 与 word2[:j] 的最小操作数
        
        # 状态转移
        for i in range(1, n1+1):
            for j in range(1, n2+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + 1
                    
        return dp[-1][-1]
```

# 712. Minimum ASCII Delete Sum for Two Strings

思路同上

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        
        # 初始化dp数组
        n1, n2 = len(s1), len(s2)
        dp = [[0]*(n2+1) for i in range(n1+1)]
        for i in range(1, n1+1):
            dp[i][0] = dp[i-1][0] + ord(s1[i-1])
        for j in range(1, n2+1):
            dp[0][j] = dp[0][j-1] + ord(s2[j-1])
        
        # dp[i][j]: s1[:i] 和 s2[:j] 的最小删除和
        
        # 状态转移
        for i in range(1, n1+1):
            for j in range(1, n2+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i][j-1] + ord(s2[j-1]), dp[i-1][j] + ord(s1[i-1]))
        
        return dp[-1][-1]
```

# 516. Longest Palindromic Subsequence

使用二维 DP 来定义子问题：`DP[i][j]` 表示 `s[i..j]` 的最长回文子序列长度。

主要不同点就在于这是一个斜向初始化和遍历的 DP 数组。

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        
        # 初始化DP数组
        n = len(s)
        dp = [[0]*n for i in range(n)]
        for i in range(n):
            dp[i][i] = 1
            
        # DP[i][j]: s[i..j] 的最长回文子序列长度
        
        # 状态转移（斜向遍历）
        for l in range(1, n):
            for i in range(n-l):
                j = l + i
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i][j-1], dp[i+1][j])
        
        return dp[0][n-1]
```

斜向遍历的坏处就是无法把二维 DP 数组压缩到一维，所以我们从下往上（i 逆序），从左往右（j 正序）的顺序遍历。

通过借助 `pre`, `temp` 等中间变量，将二维的 DP 数组压缩到一维。可以节省更多的空间复杂度。

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        
        # 初始化DP数组，把状态压缩到一维
        n = len(s)
        dp = [1]*n 
            
        # DP[j]: s[i..j] 的最长回文子序列长度
        
        # 状态转移（按行逆序遍历）
        for i in range(n-2, -1, -1):
            pre = 0
            for j in range(i+1,n):
                temp = dp[j]
                if s[i] == s[j]:
                    dp[j] = pre + 2
                else:
                    dp[j] = max(dp[j-1], dp[j])
                pre = temp
        
        return dp[n-1]
```

# 931. Minimum Falling Path Sum

推荐 DP 数组解法。

```python
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        
        # 初始化DP数组
        n = len(matrix)
        dp = [[0]*n for i in range(n)]
        for j in range(n):
            dp[0][j] = matrix[0][j]
        
        # dp[i][j] 落到 matrix[i][j] 位置时的最小路径和
        
        # 状态转移
        for i in range(1, n):
            for j in range(n):
                if j == 0:
                    min_pre = min(dp[i-1][j], dp[i-1][j+1])
                elif j == n-1:
                    min_pre = min(dp[i-1][j-1], dp[i-1][j])
                else:
                    min_pre = min(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1])
                dp[i][j] = min_pre + matrix[i][j]
                
        return min(dp[-1])
```

递归 + 备忘录版本，要比 DP 数组慢很多。

```python
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        
        n = len(matrix)
        memo = [[10001]*n for i in range(n)]
        
        def dp(i, j):
            if j < 0 or j >= n:
                return 10002
            if i == 0:
                return matrix[0][j]
            if memo[i][j] != 10001:
                return memo[i][j]
            memo[i][j] = min(dp(i-1, j-1), dp(i-1, j), dp(i-1, j+1)) + matrix[i][j]
            return memo[i][j]
            
        res = 10001
        for j in range(n):
            res = min(res, dp(n-1,j))
        
        return res
```

# 416. Partition Equal Subset Sum

按照 0-1 背包问题，定义 DP 数组：`dp[i][j]` 当背包容量剩余 j 时，前 i 个物品能否刚好装满背包。数组元素为 `True` 或 `False`

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        
        total = sum(nums)
        if total % 2 != 0: return False
        
        target = int(total / 2)
        n = len(nums)
        
        # 初始化DP数组
        dp = [[False] * (target+1) for i in range(n)]
        for i in range(n):
            dp[i][0] = True
        for j in range(1, target+1):
            if nums[0] == j:
                dp[0][j] = True
        
        # dp[i][j] 当背包容量剩余 j 时，前 i 个物品能否刚好装满背包
        
        # 状态转移
        for i in range(1, n):
            for j in range(1, target+1):
                if nums[i] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j-nums[i]] or dp[i-1][j]
        
        return dp[-1][-1]
```

根据数组的特性，可以进行状态压缩。

只是需要注意遍历顺序，j 需要逆序遍历，否则上一个 i-1 的状态会被覆盖掉

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        
        total = sum(nums)
        if total % 2 != 0: return False
        
        target = int(total / 2)
        n = len(nums)
        
        # 初始化DP数组
        dp = [False] * (target+1)
        dp[0] = True
        for j in range(1, target+1):
            if nums[0] == j:
                dp[j] = True
        
        # dp[i][j] 当背包容量剩余 j 时，前 i 个物品能否刚好装满背包
        
        # 状态转移（i 正序，j 逆序遍历）
        for i in range(1, n):
            for j in range(target, 0, -1):
                if nums[i] <= j:
                    dp[j] = dp[j-nums[i]] or dp[j]
        
        return dp[-1]
```

# 494. Target Sum

参考链接：
[动态规划和回溯算法到底谁是谁爹？](https://labuladong.gitee.io/algo/3/23/71/)

## 一、回溯算法

回溯算法的思路比较直观了当，直接按思考顺序进行穷举，配合备忘录来减少子问题的重复计算。

但是代价就是相对比较高的时间和空间复杂度。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        
        n = len(nums)
        memo = dict() # 备忘录
        
        # backtrack(i, rest) 的意义为从第 i 个数起，能凑出和为 Target - rest 的组合数

        def backtrack(i, rest):
            
            if (i, rest) in memo: return memo[(i, rest)]
            
            # 结束条件：当凑出 Target 时，返回 1；否则返回 0
            if i == n:
                if rest == 0:
                    memo[(i, rest)] = 1
                    return memo[(i, rest)]
                else:
                    memo[(i, rest)] = 0
                    return memo[(i, rest)]
            
            # 可选操作
            # backtrack(i + 1, rest - nums[i]): 加上 Nums[i] 之后剩余可能的组合数
            # backtrack(i + 1, rest + nums[i]): 减去 Nums[i] 之后剩余可能的组合数
            memo[(i, rest)] = backtrack(i + 1, rest - nums[i]) + backtrack(i + 1, rest + nums[i])
            
            return memo[(i, rest)]
        
        return backtrack(0, target)
```

## 二、动态规划 

这个问题可以转化为[416.分割等和子集问题](https://leetcode-cn.com/problems/partition-equal-subset-sum/)，即本文上一题，也是背包问题中的一种。

根据题目中元素前的符号我们可以将元素分成两个集合，假设 N 为总集合，A 为“加”集合，B 为“减”集合，T 为目标和，那么则有

$ Sum(A) - Sum(B) = T $

进而可以推出（省略一点中间步骤）

$ Sum(A) = (T + Sum(N))/2 $

至此，问题被转化为：

**从集合 N 中选出元素组合成集合 A 满足 A 的元素和为 $(T + Sum(N))/2$，请问一共有多少种选法？**

使用 DP 数组法，和子集分割问题不同的地方在于，这里的状态转移情况稍微不同，需要将“A包括i”和“A不包括i”两种情况的可能情况相加。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        
        # 判断特殊情况
        s = sum(nums)
        if s < abs(target) or (s + target) % 2 == 1:
            return 0
        
        # DP方法

        # 初始化DP数组
        n = len(nums)
        dp_target = (s + target) // 2 # 转换成子集分割问题时对应的目标和
        dp = [[0] * (dp_target + 1) for i in range(n)]

        # dp[i][j]: 前 i 个元素能构成满足和为 j 的子集合数

        # 初始状态
        dp[0][0] += 1
        if nums[0] <= dp_target:
            dp[0][nums[0]] += 1
        
        # 状态转移
        for i in range(1, n):
            for j in range(dp_target, -1, -1):
                if nums[i] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] + dp[i-1][j - nums[i]]
        
        return dp[-1][-1]
```

同样，状态可以进一步压缩以降低空间复杂度，但是一样要注意 j 要逆序遍历。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        
        s = sum(nums)
        if s < abs(target) or (s + target) % 2 == 1:
            return 0
        
        # 初始化DP数组
        n = len(nums)
        dp_target = (s + target) // 2 # 转换成子集分割问题时对应的子集和
        dp = [0] * (dp_target + 1)
        
        dp[0] += 1
        
        if nums[0] <= dp_target:
            dp[nums[0]] += 1
        
        # 状态转移
        for i in range(1, n):
            for j in range(dp_target, -1, -1):
                if nums[i] <= j:
                    dp[j] = dp[j] + dp[j - nums[i]]
        
        return dp[-1]
```

# 354. Russian Doll Envelopes

参考链接：
[二维递增子序列：信封嵌套问题](https://labuladong.gitee.io/algo/3/24/75/)

面对这道题，要做的第一件事就是**排序**。

因为套娃过程是一个本质上信封高度或者宽度递增的过程，但是显然题目中给出的信封宽高是没有规律的。所以如果不对信封首先进行一定程度的排序预处理，那么在对信封的遍历过程中一定会遇到对目前已有的套娃方案进行中途插入的问题。因此以后面对类似的题目，也要首先注意题目是否需要排序这一操作。

而这一题的难点也就在于如何对二元数组的序列进行排序。解法如下：

**将信封首先按宽度升序排序，宽度相同的时候按照高度降序排序。然后把信封所有的高度作为一个数组，然后在这个数组上计算最长递增子序列就可以得到本题答案。**

实际上就是通过排序将问题转化成了最长递增子序列的问题，该问题的解法可以参考本文的[300. Longest Increasing Subsequence (LIS)](#problem300)。

代码如下：

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        
        # 按照 width 进行升序排列，如果 width 相等，则按照 height 降序排列
        envs_sorted = sorted(envelopes, key=lambda env: (env[0], -env[1]))
        h_list = [env[1] for env in envs_sorted]
        
        # 按照最长递增子序列问题进行处理：二分的扑克牌方法
        deck = [h_list[0]]
        for poker in h_list[1:]:
            if poker > deck[-1]:
                deck.append(poker)
            else:
                i, j = 0, len(deck) - 1
                while i < j:
                    mid = (i + j)//2
                    if poker > deck[mid]:
                        i = mid + 1
                    else:
                        j = mid
                deck[i] = poker
        
        return len(deck)
```

# 53. Maximum Subarray

参考链接：

[动态规划设计：最大子数组](https://labuladong.gitee.io/algo/3/24/76/)

# 509.

# 322.

# 518. Coin Change 2

参考链接：

[经典动态规划：完全背包问题](https://labuladong.gitee.io/algo/3/25/82/)

这道题可以看作是一个背包问题的变体：

**有一个背包，最大容量为 `amount`，有一系列物品 `coins`，每个物品的重量为 `coins[i]`，每个物品的数量无限。请问有多少种方法，能够把背包恰好装满？**

在这里，关键的问题是每个物品数量是无限的。

思路：也是使用动态规划，注意明确「状态」和「选择」

状态：「背包容量」和「可选择的物品」

选择：「装进背包」和「不装进背包」

DP 数组的定义：**若只使用 `coins` 中的前 `i` 个硬币的面值，若想凑出金额 `j`，有 `dp[i][j]` 种凑法。**

状态转移方程：`dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i-1]]`

在确定状态转移方程的过程中，我们需要保证子问题得到的最大组合数不会重复。`dp[i - 1][j]` 意味着我们的选择不把该硬币凑进我们的组合里，同时之后只限制在前`i-1`个面值的硬币里作选择。另一种情况`dp[i][j - coins[i-1]]`意味着我们在该轮选择接受该硬币，并且继续保留选择第`i`个面值的硬币的权利。这样两种情况下第`i`个面值的硬币的数量就是不同的，进而可以将他们对应的最大组合数相加得到当前状态下的最大组合数。

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        
        # 特殊输入处理
        if amount == 0: return 1
        if coins[0] > amount: return 0
        
        # 初始化dp数组
        n_coins = len(coins)
        dp = [[0]*(amount + 1) for i in range(n_coins)]
        
        # dp[i][j] 当背包容量为 i 时，面对是否要往里装一个 j 硬币时的最多组合数
        
        # 初始状态
        for i in range(n_coins):
            dp[i][0] = 1
        for j in range(1, amount+1):
            if coins[0] > j:
                dp[0][j] = 0
            else:
                dp[0][j] = dp[0][j-coins[0]]
        
        # 状态转移
        for i in range(1, n_coins):
            for j in range(1, amount+1):
                if coins[i] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-coins[i]]
            
        return dp[-1][-1]
```

# 174. Dungeon Game

参考链接：

[动态规划帮我通关了《魔塔》](https://labuladong.gitee.io/algo/3/26/85/)

知道这道题要用动态规划解，但是起初无论怎么定义 DP 数组，都没办法实现合理的状态转移。

因为我一开始都把 DP 数组定义成类似于：

**从左上角（`dungeon[0][0]`）走到`dungeon[i][j]`至少需要`dp[i][j]`的生命值**

**从左上角（`dungeon[0][0]`）走到`dungeon[i][j]`能保留的最大生命值为`dp[i][j]`**

等等……

希望能通过这些定义最终推出结果，但是很显然，这道题跟那些求「最大路径和」之类的题目不一样，因为在这样的DP数组定义下，这道题目的局部最优并不等于全局最优。

当我们考虑状态`dp[i][j]`，不论我们基于何种原则（上一个状态剩下的血量，或者上一个路线的历史最低生命值）去选择上一个状态（`dp[i-1][j]`或者`dp[i][j-1]`），我们都不能保证在接下来的路线中不会遇到非常强大的怪物（绝对值非常大的负数），从而导致初始生命值需要非常大的值。

因此，我们无论如何不能从已有路线中得到足够的信息从而进行状态转移，我们会发现**信息藏在未来的路径里**，而究其原因，根本上是因为增减机制的不对称性：即**勇者先碰到怪物和先吃血瓶对于结果的影响是不同的。**

**信息藏在未来的路径里**，所以我们需要反向定义我们的 DP 数组，把终点公主（`dungeon[-1][-1]`）作为初始状态，反推到起点（`dungeon[0][0]`）：

**从`dungeon[i][j]`走到右下角(`dungeon[-1][-1]`)至少需要`dp[i][j]`的生命值**

至此，DP 数组定义完毕，从终点开始反向遍历，便会发现自然而然就推出了状态转移方程。

以后做动态规划之前不妨先想一想，**这题需要正着来还是反着来？**，**我正着推能掌握足够的信息进行状态转移吗？**

代码如下：

```python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        
        # 初始化dp数组
        height, width = len(dungeon), len(dungeon[0])
        dp = [[0] * width for i in range(height)]
        
        # 初始状态
        dp[-1][-1] = max(1, 1 - dungeon[-1][-1])
        for i in range(height-2, -1, -1):
            dp[i][-1] = max(1, dp[i+1][-1] - dungeon[i][-1])
        for j in range(width-2, -1, -1):
            dp[-1][j] = max(1, dp[-1][j+1] - dungeon[-1][j])
            
        # dp[i][j][1]：从(1,j)开始走到右下角至少需要多少生命值
        
        # 状态转移
        for i in range(height-2, -1, -1):
            for j in range(width-2, -1, -1):
                dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])
                
        return max(1, dp[0][0])
```

# 514. Freedom Trail

## DP 递归函数

参考链接：

[动态规划帮我通关了《辐射4》](https://labuladong.gitee.io/algo/3/26/86/)

按照参考的思路，通过构建递归 DP 函数来解决该问题。

首先明确「状态」和「选择」：

状态：「当前`ring`上指针指向的位置」即`ring[i]`，「当前需要输入的字符」即`key[j]`

选择：「如何拨动指针得到待输入的字符」

通过分析，可以知道，当目标字符`key[j]`在`ring`中是唯一的时候，我们只有一个选择，即从顺时针或者逆时针中选择路径最短的那一个方向转过去。

而当目标字符存在不止一个的时候，子问题就会发生分裂：我们需要通过穷举来比较选择哪一个位置作为下一个目标字符时得到的最终操作数最小。

因此我们可以更加细化「选择」：

选择：「转到圆盘上重复的目标字符中的哪一个」

于是状态转移方程就比较好写了，伪代码如下:

```python
# 当前状态 ring[i], 目标字符key[j]
def dp(i,j):
    if j >= len(key): return 0 # 基础状态
    res = ... 
    for k in range(len(ring)):
        res = min(res, min(abs(k-i), len(ring)-abs(k-i)) + dp(k, j+1))
    return res
```

其中`min(abs(k-i), len(ring)-abs(k-i))`是从`i`到`k`的最少旋转次数，

`dp(k, j+1))`是以`ring[k]`为初始状态，`key[j+1]`为目标字符的最小旋转次数。

## 更快，更高，更强

然而我们可以发现，即便结合备忘录，上述方法的时间复杂度也非常高。

而压缩问题的关键往往在于深刻理解这一问题状态转移的本质。

在 LeetCode 的讨论里发现了一个很厉害的方案：

基本逻辑和上述方案一样，也是使用 DP 函数递归，「状态」相同，但是在「选择」方面做出了很大的改进。

**选择：选择「顺时针最近的」或者「逆时针最近的」目标字符作为该轮的选择**

事实也是如此，对于状态`(i,j)`来说，最佳的`key[j]`一定只能是`ring[i]`的左边第一个或是右边第一个`key[j]`，只能是这两者之一。

而对于一个圆盘来说，我们可以**用字典来把圆盘上每一个位置向左或者向右碰到的第一个其他字符的位置存下来**，这样就不需要遍历来寻找这些目标字符。

直接将递归算法内的复杂度从`len(ring)`降到了常数级别，整整降低了一个数量级。

总结，备忘录法提供的只是剪枝，整体复杂度的数量级很难下来。而根据问题本身性质对状态的「选择」进行数量级级别的减少复杂度，能对算法起更关键的优化效果。

代码如下：

```python
# 递归法
class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        
        len_ring = len(ring)
        
        # 把每一个字符按照顺时针和逆时针顺序碰到的第一个其他字符的索引记下来
        # ring[i] 左边第一个字符 c 的索引为 lefts[i][c]
        lefts = [None for _ in range(len_ring)]
        cur = dict()
        for i in range(-len_ring + 1, len_ring, 1):
            cur[ring[i]] = i
            if i >= 0:
                lefts[i] = cur.copy()
        
        # ring[i] 右边第一个字符 c 的索引为 rights[i][c]
        rights = [None for _ in range(len_ring)]
        cur = dict()
        for i in range(2*len_ring-1, -1, -1):
            cur[ring[i%len_ring]] = i
            if i <= len_ring-1:
                rights[i] = cur.copy()
        
        # 当前状态ring[idx_ring], 下一个目标key[idx_key], 最少需要的旋转次数为 dp(idx_ring, idx_key)
        @functools.cache
        def dp(idx_ring, idx_key):
            
            if idx_key >= len(key): return 0
            
            left_target = lefts[idx_ring][key[idx_key]]
            right_target = rights[idx_ring][key[idx_key]]
            
            return min(
                abs(idx_ring - left_target) + dp(left_target%len_ring, idx_key+1),
                abs(idx_ring - right_target) + dp(right_target%len_ring, idx_key+1)
            )
        
        # 最终结果加上拍按钮的次数
        return dp(0,0) + len(key)
```