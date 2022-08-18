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

# 2022.01.30 - 884. 两句话中的不常见单词

题目可以理解成：**在两个字符串拼接的合成字符串中搜索返回只出现一次的单词**

方法一：

比较 Pythonic 的方法，将字符串连接再 `split` 之后，直接使用数组的 `count` 方法筛选出只出现一次的单词。

```python
class Solution:
    def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
        words_dict = dict()
        words = (s1 + ' ' + s2).split()
        return [word for word in words if words.count(word) == 1]
```

方法二：

同样将字符串连接再 `split` ，但是这次是自己实现快速的单词频率计算。

用一个 `appeared_words` 的 `list` 结构来存储出现过的单词，利用 `dict` 结构来对单词在 `appeared_words` 内的索引进行记录。

当单词重复出现时，就从 `appeared_words` 里将对应的单词改为 `None`，最后遍历一遍 `appeared_words` 将剩余的单词输出即可。

时间复杂度为 $O(N)$。

```python
class Solution:
    def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
        words_dict = dict()
        words = (s1 + ' ' + s2).split()
        appeared_words = []
        
        count = 0
        for word in words:
            if word not in words_dict: 
                appeared_words.append(word)
                words_dict[word] = count
                count += 1
            else:
                appeared_words[words_dict[word]] = None

        res = []
        for word in appeared_words:
            if word:
                res.append(word)

        return res
```

# 2022.01.31 - 1342. 将数字变成 0 的操作次数

利用移位操作，在模拟过程中可以不需要在奇数情况下减一。

每一次判断当前数字是奇数还是偶数，奇数就操作数加二（除 2 和减 1），偶数就只加一（除 2）。

只是这个计算只能算到 1，所以循环直到数字为 1 ，最后结果再加一即可。

```python
class Solution:
    def numberOfSteps(self, num: int) -> int:
        # 特殊情况
        if num == 0:
            return 0
        res = 0
        # 计算直到 1 时的操作数
        while num > 1:
            if num % 2 == 1:
                # 如果当前为奇数：则操作两次
                res += 2
            else:
                # 非奇数操作一次
                res += 1
            # 每次向右移位一位，无视奇偶性，结果一样
            num = num >> 1
        # 最后操作数 +1
        return res + 1
```

# 2022.02.01 - 1763. 最长的美好子字符串

一个比较难的简单题，可以动态规划，也可以用分治思想递归，分治比较快一些，代码如下：

```python
class Solution:
    def longestNiceSubstring(self, s: str) -> str:
        # 递归
        def rec(i, j):
            for k in range(i, j):
                # 遍历 s[i:j], 把第一个不满足大小写均在 s[i:j] 内的位置作为分割点
                if s[k].upper() not in s[i:j] or s[k].lower() not in s[i:j]:
                    # 在左侧的字符串切片上调用递归
                    # 返回左边的最大美好子字符串位置
                    l_i, l_j = rec(i, k)
                    # 在右侧的字符串切片上调用递归
                    # 返回右边的最大美好子字符串位置
                    r_i, r_j = rec(k+1, j)
                    # 比较左右子字符串的长度，选择较长者返回
                    if (r_j - r_i) > (l_j - l_i):
                        return (r_i, r_j)
                    else:
                        return (l_i, l_j)
            return (i, j)

        i, j = rec(0, len(s))
        return s[i:j]
```

# 2022.02.02 - 2000. 反转单词前缀

关键就是如何将未找到 `ch` 的情况和找到的情况结合在一步代码里。

```python
class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        # 找到反转点
        pos = word.find(ch) + 1
        # 将 word 分两部分，处理后再拼接
        return word[:pos][::-1] + word[pos:]
```

# 2022.02.03 - 1414. 和为 K 的最少斐波那契数字数目

首先基于动态规划生成小于 `k` 的所有斐波那契数字的列表 `num_list`。

然后根据斐波那契数字的性质，可以基于**贪心**的原则来从大到小寻找构成 `k` 的每个数字：

使用一个临时变量 `rest_sum` 来存储目标，每次寻找 `num_list[:r]` 中小于 `rest_sum` 的最大的数。

`rest_sum` 初始化为 `k`，`r` 初始化为 `len(num_list) - 1`。

假设每次寻找到的数字下标为 `i`，更新 `rest_sum = rest_sum - num_list[i]`,

则根据斐波那契数列的性质，下次寻找的目标必定存在于 `num_list[:i-1]`，因此更新 `r = i - 1` 即可。

如果使用线性搜索，时间复杂度为 $O(N)$，参考代码如下：

```python
class Solution:
    def findMinFibonacciNumbers(self, k: int) -> int:
        # 先生成所有小于 k 的斐波那契数列
        num_list = [1, 1]
        new_num = num_list[-1] + num_list[-2]
        while new_num <= k:
            num_list.append(new_num)
            new_num = num_list[-1] + num_list[-2]
        
        # 初始化
        max_index = len(num_list) - 1
        rest_sum = k
        res = 0

        # 从大到小线性搜索
        while rest_sum > 0:
            if rest_sum >= num_list[max_index]:
                rest_sum -= num_list[max_index]
                res += 1
            max_index -= 1

        return res 
```

可以使用二分搜索，但是时间并没有减少太多，因为线性搜索搜索到了之后会缩小搜索空间，而二分搜索是从中间开始，反而会产生一些重复的搜索步骤。

参考代码如下：

```python
class Solution:
    def findMinFibonacciNumbers(self, k: int) -> int:
        # 先生成所有小于 k 的斐波那契数列
        num_list = [1, 1]
        new_num = num_list[-1] + num_list[-2]
        while new_num <= k:
            num_list.append(new_num)
            new_num = num_list[-1] + num_list[-2]
        
        # 初始化
        r = len(num_list) - 1
        rest_sum = k
        res = 0

        while rest_sum > 0:
            # 重置左端点
            l = 0
            # 使用二分法确定小于 rest_num 的最大元素的位置
            while l < r:
                mid = (l + r + 1) // 2
                if num_list[mid] > rest_sum:
                    r = mid - 1
                else:
                    l = mid

            # 减去找到的元素
            rest_sum -= num_list[r]
            # 更新右端点和结果
            res += 1
            r = r - 1

        return res 
```

# 2022.02.04 - 1725. 可以形成最大正方形的矩形数目

思路简单。

```python
class Solution:
    def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
        # 生成所有切好的矩形边长的数组
        lens = [min(rect) for rect in rectangles]
        # 返回最长边长的数量
        return lens.count(max(lens))
```

# 2022.02.05 - 1219. 黄金矿工

这道题应该用**回溯算法**，开始的时候用错了，用了 BFS，在此阐述一下个人认为的 BFS 和回溯算法在目标上的不同点：
1. BFS 的目标在于遍历所有的点，只关心每一个点，不关心路径，搜索空间是**所有的点构成的空间**；
2. 回溯算法是遍历所有的路径，相比起 BFS，既关心点，也关心路径，搜索空间是**所有的可能路径构成的空间**。
这道题显然跟路径有关，不同路径返回的最大采矿量也不同，因此应该会**回溯算法**。

```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        # 初始化 visited 网格
        m, n = len(grid), len(grid[0])
        max_gold = 0
        visited = [[False]*n for _ in range(m)]

        # 定义回溯函数
        # 使用 visited 作为路径辅助
        def backtracking(x, y, gold):
            nonlocal max_gold
            # 将该点标记在路径上
            visited[x][y] = True
            # 更新当前最大采矿量
            new_gold = gold + grid[x][y]
            max_gold = max(max_gold, new_gold)
            # 对非路径上的邻点调用回溯算法
            for new_x, new_y in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                if 0 <= new_x < m and 0 <= new_y < n and grid[new_x][new_y] != 0 and not visited[new_x][new_y]:
                    backtracking(new_x, new_y, new_gold)
            # 回退，从路径上消除该点
            visited[x][y] = False
        
        # 对每一个有矿的格子调用回溯算法
        for i in range(m):
            for j in range(n):
                if grid[i][j] != 0:
                    backtracking(i, j, 0)
                    
        return max_gold
```

# 2022.02.06 - 1748. 唯一元素的和

可以直接用 `list` 类的内置 `count` 函数判断一个元素是否是唯一元素：

```python
class Solution:
    def sumOfUnique(self, nums: List[int]) -> int:
        return sum([elem for elem in nums if nums.count(elem) == 1])
```

或者自己构造映射来计算元素出现的次数：

```python
class Solution:
    def sumOfUnique(self, nums: List[int]) -> int:
        # 初始化计算出现次数的 dict
        count_dict = {}
        # 遍历元素，计算出现次数
        for elem in nums:
            if elem in count_dict:
                count_dict[elem] += 1
            else:
                count_dict[elem] = 1
        # 将唯一的元素累加
        res = 0
        for elem in count_dict:
            if count_dict[elem] == 1:
                res += elem
        return res
```

# 2022.02.07 - 1405. 最长快乐字符串

这一题主要基于**贪心算法**，每一步都选择剩余最多的字母加入答案的字符串中，因为最小的字母要承担作为分隔符的任务。

如果目标字母已经连续出现两次，那么选择第二多的字母。

通过连续局部最优得到全局最优。

参考代码如下：

```python
class Solution:
    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        ans = []
        cnt = [[a, 'a'], [b, 'b'], [c, 'c']]
        while True:
            cnt.sort(key = lambda x: -x[0])
            # 判断是否符合快乐字符串条件
            if len(ans) >= 2 and ans[-1] == cnt[0][1] and ans[-2] == cnt[0][1]:
                idx = 1
            else:
                idx = 0
            # 判断是否已达到最长
            if cnt[idx][0] <= 0:
                return ''.join(ans)
            # 将符合条件的最多的字母加入 list
            ans.append(cnt[idx][1])
            # 对应剩余数量-1
            cnt[idx][0] -= 1
```

# 2022.02.08 - 1001. 网格照明

难点在于如何储存记录每个位置灯的亮灭情况已经每个格子的点亮情况。

测试范例中有一个很大的矩阵，所以存储的时候如果需要初始化一个大矩阵就会直接超时。

因此用集合、词典或者哈希表来存储是最好的选择。

代码参考如下：

```python
class Solution:
    def gridIllumination(self, n: int, lamps: List[List[int]], queries: List[List[int]]) -> List[int]:
        # 初始化灯状态集合
        lamps_on = set()
        # 初始化亮度字典，表示被点亮的行，列，对角线，反对角线
        row, col, diag, antidiag = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)

        # 遍历 lamps，得到在查询前的矩阵和字典的状态
        for i, j in lamps:
            # 如果对应位置灯未点亮则将其点亮并更新亮度词典
            if (i,j) not in lamps_on:
                # 将该位置加入点亮的灯的词典
                lamps_on.add((i,j))
                # 灯辐射到的位置亮度 +1
                row[i] += 1
                col[j] += 1
                diag[i - j] += 1
                antidiag[i + j] += 1

        # 遍历 queries，获取结果
        ans = []
        for i, j in queries:
            # 查询亮度词典判断是否为照亮状态
            if row[i] or col[j] or diag[i - j] or antidiag[i + j]:
                ans.append(1)
            else:
                ans.append(0)
                continue
            
            # 熄灭范围内的灯并更新亮度词典
            for x, y in [(i+1,j+1),(i,j+1),(i-1,j+1),(i+1,j),(i,j),(i-1,j),(i+1,j-1),(i,j-1),(i-1,j-1)]:
                if 0 <= x < n and 0 <= y < n and (x, y) in lamps_on:
                    # 将该位置从点亮的灯集合移除
                    lamps_on.remove((x, y))
                    # 灯辐射到的位置亮度 -1
                    row[x] -= 1
                    col[y] -= 1
                    diag[x - y] -= 1
                    antidiag[x + y] -= 1
        return ans                          
```

# 2022.02.09 - 2006. 差的绝对值为 K 的数对数目

先排序后计算，比较简单。

```python
class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i+1, n):
                if nums[j] - nums[i] == k:
                    ans += 1
        return ans
```

# 2022.02.10 - 1447. 最简分数

比较简单的一道题：关键就是判断两个数互质。

这里调用了 Python 中的 `gcd` 函数。

实际上只需要用**递归调用**来实现**辗转相除法**即可。

```python
# # 判断两数最大公约数的辗转相除算法
# def gcd(i, j):
#     # 调整数字的大小顺序
#     if i > j:
#         num_1, num_2 = i, j
#     else:
#         num_1, num_2 = j, i
#     # 当余数为 0 ，找到最大公约数
#     if num_1 % num_2 == 0:
#         return num_2
#     # 当余数不为 0，再用余数除除数
#     else:
#         return gcd(num_2, num_1 % num_2)

class Solution:
    def simplifiedFractions(self, n: int) -> List[str]:
        if n == 1:
            return []
        if n == 2:
            return ["1/2"]
        ans = ["1/2"]
        for i in range(2,n):
            # 只需要遍历前一半的分子即可
            for j in range((i+1)//2):
                # 判断两数是否互质
                # 等价于最大公约数是否为 1
                if gcd(j+1,i+1) == 1:
                    # 将该分数加入答案
                    ans.append(str(j+1) + "/" + str(i+1))
                    # 将 1 与该分数的差加入答案
                    ans.append(str(i-j) + "/" + str(i+1))
        return ans
```

# 2022.02.11 - 1984. 学生分数的最小差值

排序 + 滑动窗口

```python
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        ans = float('inf')
        nums.sort()
        for i in range(len(nums)-k+1):
            diff = nums[i+k-1] - nums[i]
            if diff < ans:
                ans = diff
        return ans
```

# 2022.02.12 - 1020. 飞地的数量

BFS 或者 DFS 中“过河拆桥”思想的运用。

由于需要统计不连接边界的白色格子的数量，那么先以所有边界为 `1` 的格子为起点，遍历所有连接的值为 `1` 格子，并将其置为 `0`。

这么一番操作过后，剩下的所有为 `1` 的格子，就都是满足条件需要计算的格子了。

```python
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m, n = len(grid) ,len(grid[0])

        # 定义 BFS 递归函数
        def bfs(grid, i, j):
            q = deque()
            q.append((i, j))
            while q:
                x, y = q.popleft()
                if grid[x][y] == 1:
                    # “过河拆桥”
                    # 将遍历后的格子置为零
                    grid[x][y] = 0
                    for new_x, new_y in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                        if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y]:
                            q.append((new_x, new_y))
        
        # 对所有边界的格子调用 BFS 函数
        for i in range(m):
            bfs(grid, i, 0)
            bfs(grid, i, n-1)
        for j in range(n):
            bfs(grid, 0, j)
            bfs(grid, m-1, j)

        # 计算剩下的格子中为 1 的数量
        ans = 0
        for i in range(1, m-1):
            for j in range(1, n-1):
                if grid[i][j] == 1:
                    ans += 1
        
    return ans
```

# 2022.02.14 - 540. 有序数组中的单一元素

题目要求 $O(logN)$ 的时间复杂度，直接提示我们使用二分法来解决这个问题。

因为题目给定的数组是有序数组，在所有出现两次的元素中寻找一个只出现一次的元素。

传统的二分法是根据二分点的数值大小和目标数的大小来决定后续锁定处理的分块。

而这道题则有一点不同，是根据二分位置的值的左右两部分的元素数量来决定的。

根据二分点的值我们总能按照元素将当前区块分成两份，而目标元素总是在元素数量为奇数的那一个区块中。

根据这个性质，我们就可以设计二分法的方案了：

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l+r) // 2
            if nums[mid] == nums[mid-1]:
                # 二分值等于其左边的值的情况
                if (mid - l + 1) % 2 == 0:
                    # l 和 mid 之前（包含 mid）有偶数个数
                    # 说明目标数字在 mid 右侧
                    l = mid + 1
                else:
                    r = mid
            elif nums[mid] == nums[mid+1]:
                # 二分值等于其右边的值的情况
                if (mid - l + 1) % 2 == 1:
                    # l 和 mid 之前（包含 mid）有奇数个数
                    # 说明目标数字在 mid 右侧
                    l = mid
                else:
                    r = mid - 1
            else:
                # 两边都不相等，说明是只出现一遍的数
                return nums[mid]
        return nums[l]
```

# 2022.02.16 - 1719. 重构一棵树的方案数

号称 LeetCode 史上周赛最难题，几千人里只有十几个人做出来，思维难度非常之变态。

```python
class UF:
    def __init__(self, n):
        self.__count = n
        self.parent = [i for i in range(n)]
        self.weight = [1] * n
        self.roots = set(self.parent)

    def union(self, node1, node2):
        if not self.connected(node1, node2):
            if self.weight[node1] <= self.weight[node2]:
                node_light, node_heavy = node1, node2
            else:
                node_light, node_heavy = node2, node1
            parent_light = self.find(node_light)
            parent_heavy = self.find(node_heavy)
            self.parent[parent_light] = parent_heavy
            self.weight[parent_heavy] += self.weight[parent_light]
            self.roots.remove(parent_light)
            self.__count -= 1

    def find(self, node):
        while node != self.parent[node]:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node
    
    def connected(self, node1, node2):
        return self.find(node1) == self.find(node2)
    
    def get_group(self):
        count = 0
        root_2_idx = dict()
        groups = [[] for _ in range(self.__count)]
        for node in range(len(self.parent)):
            root = self.find(node)
            if root not in root_2_idx:
                root_2_idx[root] = count
                count += 1
            groups[root_2_idx[root]].append(node)
        return groups

class Solution:
    def checkWays(self, pairs: List[List[int]]) -> int:
        
        def rec(pairs):
            cnt = []
            num_2_idx, idx_2_num = dict(), dict()
            for pair in pairs:
                if pair[0] not in num_2_idx:
                    idx_2_num[len(cnt)] = pair[0]
                    num_2_idx[pair[0]] = len(cnt)
                    cnt.append(1)
                else:
                    cnt[num_2_idx[pair[0]]] += 1
                if pair[1] not in num_2_idx:
                    idx_2_num[len(cnt)] = pair[1]
                    num_2_idx[pair[1]] = len(cnt)
                    cnt.append(1)
                else:
                    cnt[num_2_idx[pair[1]]] += 1

            roots = [idx_2_num[i] for i in range(len(cnt)) if cnt[i] == (len(cnt) - 1)]
            root_num = len(roots)
            if root_num >= 1:
                
                res = 0
                root = roots[0]
                # 对每一个可能的 root 都做一次选择
                # 把包含 root 的 pair 都去除
                idx_2_num = dict()
                num_2_idx = dict()
                count = 0

                new_pairs = [pair for pair in pairs if root not in pair]

                for pair in new_pairs:
                    if pair[0] not in num_2_idx:
                        num_2_idx[pair[0]] = count
                        idx_2_num[count] = pair[0]
                        count += 1
                    if pair[1] not in num_2_idx:
                        num_2_idx[pair[1]] = count
                        idx_2_num[count] = pair[1]
                        count += 1
                
                uf = UF(count)
                for pair in new_pairs:
                    uf.union(num_2_idx[pair[0]], num_2_idx[pair[1]])
                
                # print("new_pairs: ", new_pairs, count)

                root_res = 1
                
                # print("uf.group: ", uf.get_group())

                for idx_group in uf.get_group():
                    group = [idx_2_num[i] for i in idx_group]
                    pairs_group = []
                    for pair in new_pairs:
                        if pair[0] in group or pair[1] in group:
                            pairs_group.append(pair)
                    sub_res = rec(pairs_group)

                    # print("pairs_group: ", pairs_group, sub_res)

                    if sub_res == 0:
                        root_res = 0
                        break
                    else:
                        root_res = min(root_res * sub_res, 2)
                res = min(root_res * root_num, 2)
                return res
            else:
                return 0
        return rec(pairs)
```

# 2022.02.16 - 1719. 重构一棵树的方案数

LeetCode 史上最难周赛题目

```python
```

# 2022.02.17 - 688. 骑士在棋盘上的概率

仔细鉴别什么时候用回溯什么时候用动态规划。

这道题用回溯和动态规划都能做，但是显然，回溯会浪费大量的时间。

找到一定的判据来判断究竟什么时候用回溯，什么时候用动态规划。

```python
```

# 2022.02.21 - 838. 推多米诺

## 方法一：广度优先搜索

以所有初始时刻被推倒的骨牌为起点，按时间发展顺序模拟骨牌的推倒，搜索出所有被推倒的骨牌。

使用数组 `times` 来记录所有骨牌被推倒的时间，这个时间最大为 `len(dominoes)`。

当遍历到一个骨牌 `i` 时，根据其下一个被推倒方向的骨牌 `ni` 的被推倒时间 `times[ni]` 来决定是否继续搜索：

1. 如果 `times[ni] == -1` 说明是没有被推倒的竖直骨牌，那么将其记为推倒，更新 `times[ni]`，并且加入队列继续搜索。

2. 如果 `times[ni] == times[i] + 1` 说明该骨牌在同一时刻受到另一个方向的推倒，那么根据规则它将保持竖直，`ans[ni] = '.'`，并且中止搜索。

```python
# 方法一： 模拟
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        n = len(dominoes)
        ans = ['.'] * n
        times = [-1 for _ in range(n)]
        directions = {'L': -1, 'R': 1}
        q = deque()

        # 把初始起点加入队列
        for i, d in enumerate(dominoes):
            if d != '.':
                ans[i] = d
                q.append((i, d))
                times[i] = 0

        # DFS
        while q:
            i, d = q.popleft()
            ni = i + directions[d]
            if 0 <= ni < n:
                if times[i] + 1 == times[ni] and ans[ni] != '.':
                    ans[ni] = '.'
                elif times[ni] == -1:
                    ans[ni] = d
                    times[ni] = times[i] + 1
                    q.append((ni, d))
        
        return ''.join(ans)
```

## 方法二： 双指针

双指针搜索，任何连续竖立的多米诺骨牌构成的子串在最后的状态仅取决于该字串两端的多米诺骨牌：
1. 如果两端的骨牌倒向相同方向，则这一串骨牌均倒向该方向；
2. 如果两端的骨牌相向而倒，则这一串骨牌也相向而倒；
3. 如果两端的骨牌方向相反，则这一串骨牌保持竖立。
根据这一规则，使用双指针，寻找所有连续的竖立骨牌的区间，对区间两端点加以分类讨论，然后对区间内部竖立的骨牌处理即可。

为了方便，可以在骨牌 list 两端分别加上向外倒的两个骨牌，不影响结果，但是方便代码计算过程。

```python
# 方法二： 双指针
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        # 两端分别加上一张向外倒的骨牌降低程序复杂性
        ans = ['L'] + list(dominoes) + ['R']
        n = len(ans)
        i = 0 
        while i < n - 1:
            # 寻找左端点的骨牌 i
            if ans[i] == '.':
                i += 1
                continue
            # 寻找右端点的骨牌 j
            j = i + 1
            while ans[j] == '.':
                j += 1
                continue
            # 如果中间存在区间，则开始分类讨论
            if i < j - 1:
                if ans[i] == ans[j]:
                    # 两个端点骨牌都倒向一个方向
                    for k in range(i+1, j):
                        ans[k] = ans[i]
                elif ans[i] == 'R':
                    # 两个端点骨牌相向而倒
                    for k in range((j-i-1)//2):
                        ans[i+k+1] = 'R'
                        ans[j-k-1] = 'L'
            # 终点变成下一个区间的起点
            i = j
        
        # 返回答案时去掉初始加上的两端的骨牌
        return ''.join(ans[1:-1])
```

# 2022.02.22 - 1994. 好子集的数目