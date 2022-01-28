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

# 2022.01.11 - 1036. 逃离大迷宫

## 题干

在一个 $10^6 × 10^6$ 的网格中，每个网格上方格的坐标为`(x, y)`。

现在从源方格`source = [sx, sy]`开始出发，意图赶往目标方格`target = [tx, ty]`。数组`blocked`是封锁的方格列表，其中每个`blocked[i] = [xi, yi]`表示坐标为`(xi, yi)`的方格是禁止通行的。

每次移动，都可以走到网格中在四个方向上相邻的方格，只要该方格**不**在给出的封锁列表`blocked`上。同时，不允许走出网格。

只有在可以通过一系列的移动从源方格`source`到达目标方格`target`时才返回`true`。否则，返回`false`。

## 思路

这道题目的矩阵很大，直接搜索时间肯定会超时，但是有一个突破口，就是`0 <= len(blocked) <= 200`，`block`的数量是在两百个以内的。

有两个思路可供选择，第一个是**压缩矩阵**，就是除去`source`，`target`和所有`block`所在的行列，压缩所有的剩下的行列，这样就可以把问题化解为一个$200 × 200$矩阵内的广度优先搜索。

第二个思路是**广度优先搜索加提前退出**，而提前退出的条件就是利用了`block`的数量：判断`source`和`target`是否被`block`包围。

因为`block`数量有限，所以被`block`包围的格子数量也是有一个上限的，当 BFS 搜索的格子数量超过这个上限时，我们就可以认为`block`没有包围住 BFS 的起点。

而这个最大值是$(N_{block}-1)×N_{block}/2$，对应的这个情况就是所有的`block`作为斜边与两个边界在角落围成一个等腰直角三角形，此时面积最大。

所以一共有三种情况：

1. 起点被包围；

2. 终点被包围；

3. 都没有被包围。

所以根据如上思想写一个函数去判断起点和终点是否都被包围即可。

## 代码

```python
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        # 判断 src 是否被包围
        def is_surrounded(src, dst, max_area):
            # BFS 的队列
            queue = [tuple(src)]
            # 计算走过的格子数
            count = 0
            # visited 数组，这里用字典实现
            visited = dict()
            visited[tuple(src)] = True
            while queue:
                x, y = queue.pop(0)
                count += 1
                if count > max_area:
                    # 当搜索的格子数超过最大可围面积时，我们判断 src 没有被围上
                    return 1
                for dx, dy in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
                    x_next, y_next = x+dx, y+dy
                    if 0 <= x_next <= 999999 and 0 <= y_next <= 999999:
                        if [x_next, y_next] == dst:
                            # 如果找到 dst，直接返回
                            return 2
                        if (x_next, y_next) not in block_dict and (x_next, y_next) not in visited:
                            queue.append((x_next, y_next))
                            visited[(x_next, y_next)] = True
            # 队列提前终止，说明无路可走，被包围
            return 0
        
        # 广搜加提前退出
        if len(blocked) <= 1: 
            return True
        n_block = len(blocked)
        # 计算最大可包围面积
        max_area = (n_block-1)*n_block/2
        # 用字典记录 block，防止重复遍历带来的高计算量
        block_dict = dict()
        for block in blocked:
            block_dict[tuple(block)] = True
        # 首先判断起点是否被包围
        ans = is_surrounded(source, target, max_area)
        # 起点被包围
        if ans == 0: return False
        # 直接找到终点
        elif ans == 2: return True
        # 判断终点情况
        ans = is_surrounded(target, source, max_area)
        # 终点被包围
        if ans == 0: return False
        # 两个点都没被包围
        return True
```

# 2022.01.13 - 747. 至少是其他数字两倍的最大数（简单）

简单题，找到数组中的最大值并确保比其他数字都大两倍以上。

同时找**最大数**的和**次大数**，然后比较最大数和次大数的两倍即可。

```python
class Solution:
    def dominantIndex(self, nums: List[int]) -> int:
        max_num = 0
        max_index = 0
        second_num = 0

        # 遍历数组，更新最大和第二大数字的值
        for i, num in enumerate(nums):
            if num > max_num:
                second_num = max_num
                max_num = num
                max_index = i
            elif num > second_num:
                second_num = num

        # 如果最大值大于等于第二大的数字的两倍则返回下标
        if max_num >= second_num * 2:
            return max_index
        else:
            return -1
```

# 2022.01.15 - 1716. 计算力扣银行的钱

简简单单的等差数列计算。

```python
class Solution:
    def totalMoney(self, n: int) -> int:
        weeks = n // 7
        days = n % 7
        money = 0
        for i in range(weeks):
            money += (i + 4) * 7
        if days != 0:
            money += (2 * weeks + 1 + days) * days / 2
        return int(money)
```

# 2022.01.16 - 382. 链表随机节点

这道题的目的主要在于考察蓄水池抽样算法。

如果我们可以通过遍历得知数据大小，那么可以通过均匀分布生成一个随机索引返回答案就行了。

但是实际场景是：

给定一个数据流，数据流长度 `N` 很大，且N直到处理完所有数据之前都不可知，请问如何在只遍历一遍数据（$O(N)$）的情况下，能够随机选取出 `m` 个不重复的数据。

有三个事情值得注意：

1. 数据流长度 `N` 很大且不可知，所以不能一次性存入内存，也不能一次性得知数据的长度，数据大小也总是在变动的；
2. 时间复杂度为$O(N)$，所以不能预先存储再用索引取出；
3. 随机选取 `m` 个数，每个数被选中的概率为 `m/N` ，随机性的保证。

算法思路如下：

1. 如果接收的数据量小于 `m` ，则依次放入蓄水池。
2. 当接收到第 `i` 个数据时， `i >= m` ，在 `[0, i]` 范围内取以随机数 `d` ，若 `d` 的落在 `[0, m-1]` 范围内，则用接收到的第 `i` 个数据替换蓄水池中的第 `d` 个数据。
3. 重复步骤2。

算法的精妙之处在于：当处理完所有的数据时，蓄水池中的每个数据都是以 `m/N` 的概率获得的。可以通过简单的手算来证明。

这道题目相当于 `m` 等于 1。

核心代码如下：

```python
class Solution:

    def __init__(self, head: Optional[ListNode]):
        # 蓄水池算法
        self.head = head

    def getRandom(self) -> int:
        node = self.head
        res_node = node
        count = 1
        # 从第二个数据开始替换
        while node.next:
            node = node.next
            count += 1
            # 生成基于目前遍历数据数目的随机数
            rand = random.randrange(count)
            if rand == 0:
                # 如果随机到的数为 0，则替换
                res_node = node
        return res_node.val
```

CPU 的性能总是有限的，当数据大小继续增加的时候，为了提高效率，我们可以应用分布式技术。

参考资料：

[蓄水池抽样算法（Reservoir Sampling）](https://www.jianshu.com/p/7a9ea6ece2af)

# 2022.01.17 - 1220. 统计元音字母序列的数目

今天的每日一题很有意思，首先看完想到的就是**动态规划**，建立 5 行 `n` 列的 dp 表，其中 `dp[i][j]` 表示**最后一个字符是第 j 个字符的长度为 i 的字母序列的数量**。

由于定义，同一列之间的字母序列是没有重复的，所以长度为 `k` 的所有序列的数量为 dp 表该列的数字之和。

定义好了 dp 表之后，状态转移表就很简单，因为题目已经给出了每一列存储数值和上一列的关系。

算完之后发现才 beat 5%，不能忍，马上又想到一个新的想法。来源于今天面试字节的斐波那契数列计算的 $O(log{n})$ 的算法。

就是使用**矩阵**来表示状态之间的转移，这个方法叫做**矩阵快速幂**。

比如斐波那契数列问题中，直接用如下 `N1` 作为初始状态，`M` 作为状态转移矩阵，然后就能发现，不断地用矩阵 `M` 点乘向量 `N`，就可以得到后续的结果。

```python
N1 = [1, 1]
M = [[1, 1], 
     [1, 0]]
```

而时间复杂度降低的诀窍就在于，矩阵之间点乘可以直接构造出对应更高步数的状态转移矩阵。

比如 $M_2 = M \dot M$ 得到的是表示横跨两位的状态转移方程，比如 $N_5 = M_2 \dot N_3$。

利用这一性质，结合递归运算，就可以直接将复杂度减少到 $O(log(n))$ 了， 具体看代码：

```python
class Solution:
    def countVowelPermutation(self, n: int) -> int:
        # 特殊情况
        if n == 1:
            return 5

        # 初始状态
        n1 = [1] * 5

        # 定义状态转移矩阵
        self. M = [[0, 1, 1, 0, 1],
                   [1, 0, 1, 0, 0],
                   [0, 1, 0, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 1, 0]]
        
        # 获得 n 次方状态转移矩阵
        M_n = self.get_M_n(n-1)

        # 矩阵运算获得最终结果
        res = 0
        for i in range(5):
            for k in range(5):
                res += M_n[i][k] * n1[k]

        # 返回结果
        return res % (10**9 + 7)

    def get_M_n(self, n):
        # 获得状态转移矩阵的 M 次方
        # 时间复杂度 O(logn)
        if n == 1:
            return self.M
        half_M = self.get_M_n(n // 2)
        res_M = self.product(half_M, half_M)
        if n % 2 == 1:
            res_M = self.product(res_M, self.M)
        return res_M

    def product(self, M1, M2):
        # 5 * 5 的矩阵乘法
        M = [[0]*5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    M[i][j] += M1[i][k]*M2[k][j]
        return M
```

# 2022.01.18 - 539. 最小时间差

求时间之间的最小差值，主要思路就是去重，数据转换，排序，首尾相接，然后差分求最小值。

如果有重复的，那么就可以直接返回 0.

否则就要在排序完之后，将最小的加上 24*60 然后添加到队尾，实现一个首位相接的效果。

参考代码如下：

```python
class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        # 去重
        time_unique = list(set(timePoints))
        if len(time_unique) != len(timePoints):
            return 0
        # 数据转换
        min_list = []
        for time in time_unique:
            hour, minute = time.split(':')
            min_list.append(int(hour) * 60 + int(minute))
        # 排序
        min_list = sorted(min_list)
        # 首尾相接
        min_list.append(24*60 + min_list[0])
        # 差分求最小
        diff = [min_list[i] - min_list[i-1] for i in range(1, len(min_list))]
        return min(diff)
```

# 2022.01.22 - 1332. 删除回文子序列

脑经急转弯，拿来练 C++ 了。

因为题目只有 `'a'` `'b'` 两种字符，所以最多只需要两次操作就能删除干净了，第一次删 `'a'`，第二次删 `'b'`。

所以只需要判断整个字符串是不是回文字符串即可，如果是，返回 `1`，如果不是就返回 `2`。

```C++
class Solution {
public:
    int removePalindromeSub(string s) {
        
        int i = 0;
        int j = s.length() - 1;
        while (i < j){
            if (s[i] == s[j]){
                i += 1;
                j -= 1;
            }
            else{
                return 2;
            }

        }
        return 1;
    }
};
```

# 2022.01.24 - 2045. 到达目的地的第二短时间

这道题红绿灯的设置就是障眼法，因为所有的红绿灯的变化周期是相等的，所以的路径时间消耗又是一样的，所以如果用 BFS，一个路线在等红灯，其他路线也都在等红灯。

所以实际上这道题目的目标是如何求出**到达目的地的严格次小步数**。

求最小步数很简单， DFS 就可以了。但是如果求严格次小步数，就需要一些特殊的设计。

这里把以前用的 `visited` 数组用一个 `dist` 数组代替了，其中：

- `dist[i][0]` 表示从点 `i` 到起点的**最短距离**；

- `dist[i][1]` 表示从点 `i` 到起点的**严格次小距离**。

只有更新**最短距离**，才能知道**严格次小距离的下限**，而我们可以通过这两个值来对入队的节点进行条件限制。

```python
                if step + 1 < dist[next_index][0]:
                    # 当某节点第一次符合上述条件入队时
                    # 对应的 step 一定是最小值
                    dist[next_index][0] = step + 1
                    q.append([next_index, step + 1])
                elif dist[next_index][0] < step + 1 < dist[next_index][1]:
                    # 当某节点第一次符合上述条件入队时
                    # 对应的 step 一定是严格次小小值
                    dist[next_index][1] = step + 1
                    q.append([next_index, step + 1])
```

上面第一个条件相当于原来的 `visited` 数组，在 BFS 的情况下每个节点只有第一次访问时会入队。

而第二个条件则保证了每一个节点第二次入队时的步数一定为**严格次小步数**，因为步数等于 `dist[next_index][0]` 的情况被排除在外了。

这样并不会导致漏掉部分情况，因为对同一节点相同步数的访问只是无用的重复访问而已。

最后讨论时间转换的情况，一共两种方法：

1. 通过分类讨论直接计算结果；

2. 通过模拟每一步更新总时间。

两者各有优劣，第一种时间快但是思考起来比较复杂；第二种直观好理解，坏处就是要遍历。

```python
class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        # 使用邻接表来表示图
        graph = [[] for _ in range(n)]
        for edge in edges:
            vertex_1 = edge[0] - 1
            vertex_2 = edge[1] - 1
            graph[vertex_1].append(vertex_2)
            graph[vertex_2].append(vertex_1)
        # 使用队列结构实现 BFS
        q = deque()
        # dist[i][0] 表示从点 i 到起点的最短距离
        # dist[i][1] 表示从点 i 到起点的严格次小距离
        dist = [[float(inf)] * 2 for _ in range(n)]
        dist[0][0] = 0
        q.append([0, 0])

        # BFS 
        while dist[-1][1] == float(inf):

            index, step = q.popleft()

            # BFS 可以保证每次节点入队时对应的 step 是非递减的
            for next_index in graph[index]:
                if step + 1 < dist[next_index][0]:
                    # 当某节点第一次符合上述条件入队时
                    # 对应的 step 一定是最小值
                    dist[next_index][0] = step + 1
                    q.append([next_index, step + 1])
                elif dist[next_index][0] < step + 1 < dist[next_index][1]:
                    # 当某节点第一次符合上述条件入队时
                    # 对应的 step 一定是严格次小小值
                    dist[next_index][1] = step + 1
                    q.append([next_index, step + 1])
        
        # 使用函数来将步数转换为时间
        return self.time_calculate_1(dist[-1][1], time, change)
    
    def time_calculate_1(self, step, time, change):
        # 分类讨论公式法
        if time < change:
            move_per_change = change // time
            if change % time == 0:
                move_per_period = move_per_change
            else:
                move_per_period = move_per_change + 1
            num_period = step // move_per_period
            if step % move_per_period == 0:
                return (num_period - 1) * change * 2 + time * move_per_period
            else:
                return num_period * change * 2 + time * (step % move_per_period)
        else:
            if (time // change) % 2 == 1:
                time_pre_step = time + (change - time % change)
                return time_pre_step * step - (change - time % change)
            else:
                rest_time = time % change
                if rest_time == 0:
                    return (time - rest_time) * step
                else:
                    return (time - rest_time) * step + self.time_calculate(step, rest_time, change)

    def time_calculate_2(self, step, time, change):
        # 模拟法
        total_time = 0
        for _ in range(step):
            if total_time % (2*change) >= change:
                total_time += 2 * change - total_time % (2 * change)
            total_time += time
        return total_time
```

# 2022.01.25 - 1688. 比赛中的配对次数

可以用递归，但是有更简单的方法：

每次匹配会淘汰一个队伍，最终只剩一个队伍，所以淘汰了 `n-1` 个队伍，所以匹配了 `n-1`，答案就是 `n-1`。

代码放递归的：

```python
class Solution:
    def numberOfMatches(self, n: int) -> int:

        def rec(n):
            if n == 1:
                return 0
            if n == 2:
                return 1
            
            match_num = n // 2

            if n % 2 == 0:
                return rec(match_num) + match_num
            else:
                return rec(match_num + 1) + match_num

        return rec(n)
```

# 2022.01.27 - 2047. 句子中的有效单词数

这道题用**正则匹配**更加锻炼自己，两点需要注意：
1. 模式字符串想清楚：`([a-z]*[!|.|,]?)?` 表示的是不包括 `'-'` 的合法单词，而 `'([a-z]+[-][a-z]+)'` 则表示带有 `'-'` 的部分，所以将两者结合就是能够匹配所有合法单词的模式字符串了。
2. `pattern.match(word)` 的返回是从整个字符串的第一个字符开始匹配的结果，如果有连续的符合条件的匹配项，会返回多个匹配结果。所以要判断整个 `word` 是不是满足匹配条件，需要将匹配结果的第一个 `result[0]` 和整个 `word` 进行比较。

```python
class Solution:
    def countValidWords(self, sentence: str) -> int:
        words = sentence.split()
        count = 0
        pattern = re.compile(r'([a-z]+[-][a-z]+)?([a-z]*[!|.|,]?)?')
        for word in words:
            result = pattern.match(word)
            if result and result[0] == word:
                count += 1
        return count
```

# 2022.01.28 - 1996. 游戏中弱角色的数量

你正在参加一个多角色游戏，每个角色都有两个主要属性：**攻击** 和 **防御**。给你一个二维整数数组 `properties` ，其中 `properties[i] = [attack_i, defense_i]` 表示游戏中第 `i` 个角色的属性。

如果存在一个其他角色的攻击和防御等级**都严格高于**该角色的攻击和防御等级，则认为该角色为**弱角色**。更正式地，如果认为角色 `i` **弱于**存在的另一个角色 `j` ，那么 `attack_j > attack_i` 且 `defense_j > defense_i` 。

返回**弱角色**的数量。

两种方法，直接**排序**和**单调栈**，其核心思想一样，都是：

根据两个特征按照一定优先级进行排序。

目的是比大小，那么先排序一定没有错。

首先可以想到的是按照攻击力从大到小排序，然后从头开始遍历，用一个临时变量存储遍历过的角色的最大防御值 `max_defense`。

当目前遍历的角色的防御力 `defense` 小于 历史最大防御力 `max_defense` 的时候，则 `count` 加一。

这样的判定存在的问题就是历史防御力最高的角色可能和当前角色攻击力相同，于是就追加排序条件：当攻击力相同的时候，防御力升序排列，这样就可以避免前面所述的情况了。

参考代码如下：

```python
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        # 首先按照攻击力降序排列，然后按照防御力升序排列
        properties_sorted = sorted(properties, key=lambda x: (-x[0], x[1]))
        # 维护历史最大防御力
        max_defence = properties_sorted[0][1]
        # 计算弱角色数量
        count = 0
        # 从攻击力最高者开始遍历
        for attack, defense in properties_sorted:
            if defense < max_defence:
                # 当防御力低于历史最大防御力时，弱角色数量加一
                count += 1
            elif defense > max_defence:
                # 当防御力大于历史最大防御力时，更新历史最大防御力
                max_defence = max(defense, max_defence)
        return count
```