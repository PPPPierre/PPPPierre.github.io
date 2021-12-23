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
[动态规划设计：最长递增子序列](https://labuladong.gitee.io/algo/3/24/74/)

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

这个问题可以转化为背包问题中的子集分割问题，根据题目中元素前的符号我们可以将元素分成两个集合，假设 N 为总集合，A 为“加”集合，B 为“减”集合，T 为目标和，那么则有

$ Sum(A) - Sum(B) = T $

进而可以推出（省略一点中间步骤）

$ Sum(A) = (T + Sum(N))/2 $

至此，问题被转化为：

**从集合 N 中选出元素组合成集合 A 满足 A 的元素和为 $(T + Sum(N))/2$，请问一共有多少种选法？**

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