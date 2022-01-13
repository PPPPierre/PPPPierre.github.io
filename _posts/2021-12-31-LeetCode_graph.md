---
title: LeetCode 刷题记录 - 数据结构之图（Graph） 
author: Stone SHI
date: 2021-12-31 08:57:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, Graph]
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

# 图论基础

主要组成：**节点**和**边**

主要存储方式：
1. **邻接表**：占用空间少，但无法快速判断两节点是否相邻；
2. **邻接矩阵**：占用空间大，但是可以快速判断两节点是否相邻。

重要概念：
1. **有向图**与**无向图**：**无向**就等于**双向**；
2. **有环图**与**无环图**：有环图需要`visited`数组来辅助遍历；
3. **加权图**：存储的**邻接矩阵**内的元素从**布尔值**变为**整数**或**浮点数**即可。

参考链接：

[图论基础](https://labuladong.gitee.io/algo/2/19/34/)

# 797. All Paths From Source to Target

这道题输入一幅**有向无环图**，这个图包含`n`个节点，标号从`0`到`n-1`，要求计算出所有从节点`0`到节点`n-1`的路径。

题目输入的`graph`是用「邻接表」表示的一幅图，其中`graph[i]`存储着节点`i`的所有邻居节点。

思路比较简单：以`0`为起点遍历图，同时记录遍历过的路径，当遍历到终点时将路径记录下来即可。

因为输入的图是无环的，所以不需要`visited`数组辅助，每条路径都是一定有终点的。

参考代码如下：

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        # 获得节点总数
        n = len(graph)
        path = []
        res = []
        def traverse(i):
            # 把当前节点加入列表
            path.append(i)
            # 如果是终点，则把路径加入结果列表
            if i == n-1:
                res.append(path.copy())
            else:
                # 遍历所有邻居节点
                for neighbor in graph[i]:
                    traverse(neighbor)
            # 离开时将当前节点从路径中删除
            path.pop()
            
        traverse(0)
        return res
```

# 207. Course Schedule

经典的判断有向图是否有环的问题。

问题设定在一个选课的背景下。给出`numCourses`为课程的数量，以及`prerequisites: List[List[int]]`存储了课程与课程之间的先决条件，给出这些课程是否可以学完，也就是确定由课程构成的图中是否有环。

思路比较简单，首先生成对应的邻接表，之后写好递归遍历的函数，结合`visited`减少重复遍历，最后靠`onPath`数组表示路径来判断是否有环。

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 用邻接表生成graph
        graph = [[] for _ in range(numCourses)]
        for prereq in prerequisites:
            graph[prereq[1]].append(prereq[0])
        
        # 初始化辅助用数组
        visited = [False for _ in range(numCourses)]
        onPath = [False for _ in range(numCourses)]
        hasCycle = False
        
        # 定义遍历函数
        def traverse(node):
            nonlocal hasCycle
            # 如果节点在路径上，则有环
            if onPath[node]: 
                hasCycle = True
                return
            # 如果访问过或者有环，停止遍历
            if visited[node] or hasCycle:
                return
            # 前序遍历操作
            visited[node] = True
            onPath[node] = True
            for neighbor in graph[node]:
                traverse(neighbor)
            # 后序遍历操作
            onPath[node] = False
        
        for node in range(numCourses):
            traverse(node)
        
        return not hasCycle
```

# 210. Course Schedule II

经典图论的拓扑排序问题，问题依旧是选课问题。给出`numCourses`为课程的数量，以及`prerequisites: List[List[int]]`存储了课程与课程之间的先决条件，给出为了学完所有课程所安排的学习顺序。

而直观地说，拓扑排序（Topological Sorting）的定义就是：**把一幅图「拉平」，而且这个「拉平」的图里面，所有箭头方向都是一致的。**

相当于一个以节点优先级作为标准的排序问题。

对于这个问题，**如果把课程抽象成节点，课程之间的依赖关系抽象成有向边，那么这幅图的拓扑排序结果就是上课顺序。**

首先实现上一题的代码判断是否有环，**而拓扑排序的结果，其实就是后序遍历结果的反转。**

为什么？有一个非常直观的解释：

在后序遍历中，只有遍历完所有邻居节点（在这道题里是前置课程）之后才会执行后序遍历位置的代码。换句话说，当邻居节点都被装到结果列表里面了，当前节点才会被装进去。

后序遍历的这一特点很重要，**之所以拓扑排序的基础是后序遍历，是因为一个任务必须在等到所有的依赖任务都完成之后才能开始开始执行。**

至于为什么是后续遍历结果的反转，这取决与实际问题对应的图中**边的方向的意义**，如果是低优先级指向高优先级（一门课程的先修课程指向该课程，也就是学习顺序），那么就需要反转。如果是高优先级指向低优先级（一门课程指向其先修课程），那么实际上就并不需要反转，主要还是要结合实际情况分析。

参考代码如下：

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        
        # 遍历函数
        def traverse(node):
            # 如果节点在路径里，判断有环
            nonlocal hasCycle
            if path[node]:
                hasCycle = True

            # 如果存在环或者访问过就跳过
            if visited[node] or hasCycle:
                return
            
            # 前序遍历操作
            path[node] = True
            visited[node] = True 

            # 遍历
            for neighbor in graph[node]:
                traverse(neighbor)
            
            # 后序遍历操作
            path[node] = False
            res.append(node)
        
        # 生成图的邻接表
        graph = [[] for _ in range(numCourses)]
        for prec in prerequisites:
            graph[prec[1]].append(prec[0])
        
        # 初始化变量
        path = [False for _ in range(numCourses)]
        visited = [0 for _ in range(numCourses)]
        hasCycle = False
        res = []
        
        for node in range(numCourses):
            traverse(node)
        
        if hasCycle: return []

        # 翻转后序遍历结果
        res.reverse()
        
        return res
```

还有一个很厉害的思路，不用到递归方法，也快很多，就是直接从课程的前置条件信息中，将课程的优先级计算出来，然后逐渐遍历零优先级的课程（也就是不需要前置条件的课程），去降低对应后续课程的优先级，逐渐扩充结果列表。

具体代码如下：

```python
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        # 生成邻接表和优先级表
        adjacency_list = [[] for i in range(numCourses)]
        in_degrees = [0] * numCourses
        for courses in prerequisites:
            course, prereq = courses
            # 前置课程指向后续课程
            adjacency_list[prereq].append(course)
            # 后续课程的优先级更高
            in_degrees[course] += 1
        # 过滤出所有不需要前置条件的课程，也就是零优先级课程
        zero_indegree_courses = [i for i in range(numCourses) if in_degrees[i] == 0]
        # 遍历所有零优先级课程
        for course in zero_indegree_courses:
            for adjacent_course in adjacency_list[course]:
                # 将该课程对应的后续课程的优先级减一
                in_degrees[adjacent_course] -= 1
                # 当该后续课程优先级为零时将其加入结果列表
                if in_degrees[adjacent_course] == 0:
                    zero_indegree_courses.append(adjacent_course)
        # 最后返回零优先级课程列表
        # 如果存在环，则零优先级列表的元素数量一定低于课程数量
        # 因为环上的所有课程优先级至少为一，无法减为零
        return zero_indegree_courses if len(zero_indegree_courses) == numCourses else []
```

参考：

[拓扑排序，YYDS！](https://labuladong.gitee.io/algo/2/19/35/)

# 785. Is Graph Bipartite?

这道题涉及的知识是图论里的**二分图**定义。

什么是二分图？百度百科定义如下：

> 二分图的顶点集可分割为两个互不相交的子集，图中每条边依附的两个顶点都分属于这两个子集，且两个子集内的顶点不相邻。

其实该定义等价于「双色问题」：

> 如果一个图是二分图，当且仅当可以只用两种颜色将图中的所有顶点着色，且使得任意一条边的两个端点的颜色都不相同。

所以这道题的思路就比较简单，我们可以一边遍历图一遍进行染色，碰到遍历过的节点就比较其颜色和当前节点是否相同。

这里提供 DFS 和 BFS 两种方法：

DFS:

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        # visited 数据： -1 未访问，0 和 1 分别代表两种颜色
        visited = [-1 for _ in range(n)]
        # 是否染色失败
        fail = False
        # DFS 递归函数
        def traverse(node, color):
            nonlocal fail
            if fail: return
            # 染色
            visited[node] = color
            # 访问该节点每一个邻居节点
            for neighbor in graph[node]:
                if visited[neighbor] == color:
                    # 如果邻居节点访问过且颜色和当前节点相同，则判定失败
                    fail = True
                    return
                if visited[neighbor] == -1:
                    # 如果未访问则调用递归
                    traverse(neighbor, 1- color)

        # 对每一个节点都调用一次递归
        for node in range(n):
            if visited[node] == -1:
                traverse(node, 0)
        
        return not fail
```

BFS:

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        
        n = len(graph)
        # 这里存储访问状态和颜色的数组要分开
        visited = [False for _ in range(n)]
        colors = [False for _ in range(n)]
        # 辅助 BFS 的队列
        queue = []
        for node in range(n):
            # 对未访问的节点进行遍历
            if not visited[node]:
                queue.append(node)
                while len(queue) != 0:
                    # 从队列取出节点
                    node = queue.pop(0)
                    # 标记访问
                    visited[node] = True
                    for neighbor in graph[node]:
                        if visited[neighbor]:
                            # 如果邻居节点访问过，则进行颜色判定
                            if colors[node] == colors[neighbor]:
                                return False
                        else:
                            # 未访问则染色，且把邻居节点加入队列
                            colors[neighbor] = not colors[node]
                            queue.append(neighbor)
        
        return True
```

参考：

[二分图判定](https://labuladong.gitee.io/algo/2/19/36/)

# 886. Possible Bipartition

和上一题一样的思路，就是多一步构建`graph`的过程。

```python
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        # 创建图
        graph = [[] for _ in range(n)]
        for dislike in dislikes:
            graph[dislike[0]-1].append(dislike[1]-1)
            graph[dislike[1]-1].append(dislike[0]-1)
        
        # 直接返回上一题的二分图解法
        return self.isBipartite(graph)
        
    def isBipartite(self, graph: List[List[int]]) -> bool:
        # ... 参考上一题
```

# 323. Number of Connected Components in an Undirected Graph

这一题考察的是 Union-Find 算法，也就是「并查集算法」的使用，主要用于解决图论中的「动态连通性」问题的。

## 动态连通性

在图论中，动态连通性主要指的是实现一种数据结构，能够满足以下需求：

> Given a set of N objects.
> Union command: connect two objects.
> Find/connected query: is there a path connecting the two objects?

即能够实现检查两个节点是否连通，同时可以在任意两个不连通的节点之间添加连接。

所以 Union-Find 算法主要需要实现这两个 API：

```python
class UF:
    # ...

    # 将 p 和 q 连接
    def union(self, p: int, q: int) -> None:
        
    # 判断 p 和 q 是否连通# 判断 p 和 q 是否连通
    def connected(self, p:int, q: int) -> bool:
        
    # 返回图中有多少个连通分量
    def count(self) -> int:
```

这里所说的「连通」是一种等价关系，具有如下三个性质：

1、自反性：节点`p`和`p`是连通的。

2、对称性：如果节点`p`和`q`连通，那么`q`和`p`也连通。

3、传递性：如果节点`p`和`q`连通`q`和`r`连通，那么`p`和`r`也连通。

从「连通」的概念我们可以推出「连通分量」的概念，即所有**连通的节点构成的集合称为一个连通分量**。

动态连通性问题的实际应用很多，比如：检查通信网络中计算机之间是否连通、电子电路中的触点是否连接或者社交网络中的人是否相识等等。

至此我们也搞清楚了，「并查集」中的「并」的意思，就是连接、合并两个连通分量；「查」就是查找判断两个节点是否属于一个连通分量。

Union-Find 算法的关键就在于 union 和 connected 函数的效率。

## 数据结构

我们使用**树**结构来表示每一个连通分量，其中数的节点就是图的节点，实现的具体数据结构是**数组**。

我们规定每一个节点有一个指针指向其父节点，表示该节点和父节点连通，如果没有父节点，则指向自己。

那么一个连通分量内的所有节点都共享一个根节点，该根节点也就代表了整个连通分量。

现在假设有`N`个节点，数组的第`i`个元素则代表节点`i`所指向的父节点的索引，于是我们就可以用一个数组来表示整个随机分量的森林了：

```python
class UF:
    # 构造函数
    def __init__(self, n: int):
        
        # 一开始互不连通
        self.__count = n
        # 父节点指针初始指向自己
        self.parent = [i for i in range(n)]

    # 将 p 和 q 连接
    def union(self, p: int, q: int) -> None:
        
    # 判断 p 和 q 是否连通# 判断 p 和 q 是否连通
    def connected(self, p:int, q: int) -> bool:
        
    # 返回图中有多少个连通分量
    def count(self) -> int:
        return self.__count
```

**如果某两个节点被连通，则让其中的（任意）一个节点的根节点接到另一个节点的根节点上：**

```python
    # 将 p 和 q 连接
    def union(self, p: int, q: int) -> None:
        rootP = self.find(p)
        rootQ = self.find(q)
        if (rootP != rootQ):
            # 合并两棵树
            self.parent[rootP] = rootQ
            # 分量数减一
            self.__count -= 1

    # 返回某个节点的根节点
    def find(self, x: int) -> int:
        # 根节点的parent等于它自己
        while(self.parent[x] != x):
            x = self.parent[x]
        return x
```

**判断节点 p 和 q 是否连通，只要检查它们是否拥有相同的根节点即可：**

```python
    # 判断 p 和 q 是否连通# 判断 p 和 q 是否连通
    def connected(self, p:int, q: int) -> bool:
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ
```

至此， Union-Find 算法基本完成。其复杂度主要取决于`connected`和`union`两个 API 的复杂度，而这两个 API 的复杂度主要取决于 `find` 函数的复杂度，而 `find` 函数的复杂度则取决于树的大小和平衡情况。在最坏情况下一棵树会退化成一个链表，而此时`find`函数需要寻找到根节点就需要遍历整棵树， 时间复杂度为$O(N)$。对于社交网络等一些数据规模庞大的对象，这样的复杂度是十分不理想的。因此我们需要在生成树的过程做一些优化改进，使得生成的树结构更加平衡。

## 算法改进 1. 加权

添加一个`weight`数组，记录每一课树的“重量”，其实也就是节点数量，代表树的权重。在`union`函数中，总是让权重小的树的根节点指向权重大的树的根节点，这样可以一定程度上保证树结构的平衡性。

这一优化可以使得时间复杂度降低到$O(logN)$左右。

```python
class UF:
    # 构造函数
    def __init__(self, n: int):
        
        # 一开始互不连通
        self.__count = n
        # 父节点指针初始指向自己
        self.parent = [i for i in range(n)]
        # 初始权重都是 1
        self.weight = [1 for i in range(n)]

    # 将 p 和 q 连接
    def union(self, p: int, q: int) -> None:
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ: return
        # 合并两棵树
        # 小树接到大树下面，更平衡
        if self.weight[rootP] < self.weight[rootQ]:
            self.parent[rootP] = rootQ
            self.weight[rootQ] += self.weight[rootP]
        else:
            self.parent[rootQ] = rootP
            self.weight[rootP] += self.weight[rootQ]
        # 分量数减一
        self.__count -= 1
```

## 算法改进 2. 路径压缩

在`find`函数中加上一行代码：

```python
    # 返回某个节点的根节点
    def find(self, x: int) -> int:
        # 根节点的parent等于它自己
        while(self.parent[x] != x):
            # 路径压缩
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
```

每一次`find`函数执行过程中，都可以压缩树结构，**使得同一个连通分量里的每一个节点都直接指向其根节点**。

在这样一个平的树的结构下，时间复杂度可以降低到$O(1)$。

## 最终代码

```python
class UF:
    # 构造函数
    def __init__(self, n: int):
        
        # 一开始互不连通
        self.__count = n
        # 父节点指针初始指向自己
        self.parent = [i for i in range(n)]
        # 初始权重都是 1
        self.weight = [1 for i in range(n)]

    # 将 p 和 q 连接
    def union(self, p: int, q: int) -> None:
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ: return
        # 合并两棵树
        # 小树接到大树下面，更平衡
        if self.weight[rootP] < self.weight[rootQ]:
            self.parent[rootP] = rootQ
            self.weight[rootQ] += self.weight[rootP]
        else:
            self.parent[rootQ] = rootP
            self.weight[rootP] += self.weight[rootQ]
        # 分量数减一
        self.__count -= 1

    # 判断 p 和 q 是否连通# 判断 p 和 q 是否连通
    def connected(self, p:int, q: int) -> bool:
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ
        
    # 返回图中有多少个连通分量
    def count(self) -> int:
        return self.__count
    
    # 返回某个节点的根节点
    def find(self, x: int) -> int:
        # 根节点的parent等于它自己
        while(self.parent[x] != x):
            # 路径压缩
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        uf = UF(n)
        for edge in edges:
            uf.union(edge[0], edge[1])
        
        return uf.count()
```

## 参考资料:

[UNION-FIND算法详解](https://labuladong.gitee.io/algo/2/19/37/)

[算法4（Algorithms4）- Part 1 动态连通性（Dynamic Connectivity）1](https://www.jianshu.com/p/44541a3fe8b3)

# 130. Surrounded Regions

第 130 题，被围绕的区域：给你一个 M × N 的二维矩阵，其中包含字符 `X` 和 `O`，让你找到矩阵中**四面**被 `X` 围住的 `O`，并且把它们替换成 `X`。

根据题意，四面，不包括斜角，因此边和角上的`O`一定不会被围住。所以这道题的基本思路是从边角上的`O`出发，找到跟这些`O`连通的其他`O`，替换剩下的`O`即可。

具体有两种方法，第一种是正常的 DFS 的方案，先用 for 循环遍历棋盘的**四边**，用 DFS 算法把那些与边界相连的`O`换成一个特殊字符，比如`#`；然后再遍历整个棋盘，把剩下的`O`换成`X`，把`#`恢复成`O`。这样就能完成题目的要求，时间复杂度$O(MN)$:

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 获取棋盘的宽高
        m = len(board)
        n = len(board[0])
        visited = [[False] * n for _ in range(m)]

        # 深度优先遍历
        def dfs(i,j):
            # 不符合范围的索引或者访问过的节点直接返回
            if i < 0 or i > m-1: return
            if j < 0 or j > n-1: return
            if visited[i][j]: return

            # 前序遍历操作
            visited[i][j] = True
            
            # 仅当目前为止为'O'时继续递归
            if board[i][j] == 'O':
                # 将'O'替换成'#'
                board[i][j] = '#'
                # 遍历与其直接相邻的其他节点
                dfs(i+1, j)
                dfs(i-1, j)
                dfs(i, j+1)
                dfs(i, j-1)
            else:
                return
        
        # 对所有边角的'O'调用dfs
        for i in range(m):
            dfs(i, 0)
            dfs(i, n-1)  
        for j in range(n):
            dfs(0, j)
            dfs(m-1, j)

        # 将'#'替换回'O'，将剩下的'O'换成'X'
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == '#':
                    board[i][j] = 'O'
```

第二种方法，就是使用上一题提到的 Union-Find 算法解决，虽然实现复杂一些，甚至效率也略低，但这是使用 Union-Find 算法的通用思想，值得一学。

在这道题中，这些`O`就是单独的节点，「连通性」就是指节点与节点之间是否是直接相邻。

那么我们就可以直接使用上一题的代码来解决这个问题，只要首先创建一个虚拟头节点`dummy`，然后将所有的边角上的`O`都指向该节点，那么这些节点都归属于同一个连通分量了。再将所有的`O`和其邻接的`O`连通，所有的`O`就会分属于不同的连通分量，而和`dummy`处在同一个连通分量里的`O`都不需要替换；替换其他`O`即可。

最后剩下的问题就是，怎么把二维的数据投射到一维的数组上，最简单常用的方法就是`parent[i × n + j] = board[i][j]`其中`n`是`board`的宽度，也就是索引`j`对应的上限。然后让我们的虚拟头指针占用索引`m × n`，我们刚好需要构建一个大小为`m × n + 1`的数组。

参考代码：

```python
class UF:
    # Union-Find 的实现
    # 参考上题
    # ...

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(board)
        n = len(board[0])
        # 给dummy留一个位置
        uf = UF(m * n + 1)
        # 定义虚拟头节点
        dummy = m * n
        # 将首行和末行的'O'和dummy连通
        for i in range(m):
            uf.union(i*n, dummy)
            uf.union(i*n + n - 1, dummy)
        # 将首列和末列的`O`和dummy连通
        for j in range(n):
            uf.union(j, dummy)
            uf.union((m-1)*n + j, dummy)

        # 定义方向数组
        directions = [(0,1), (1,0), (-1,0), (0,-1)]

        # 遍历连通所有的'O'
        for i in range(1, m-1):
            for j in range(1, n-1):
                if board[i][j] == 'O':
                    curr_node = i * n + j
                    # 将该'O'和上下左右的'O'连通
                    for dir_i, dir_j in directions:
                        if board[i+dir_i][j+dir_j] == 'O':
                            neighbor = curr_node + dir_i * n + dir_j
                            uf.union(curr_node, neighbor)
        
        # 将所有不和dummy连通的'O'改为'X'
        for i in range(1, m-1):
            for j in range(1, n-1):
                if board[i][j] == 'O':
                    curr_node = i * n + j
                    if not uf.connected(curr_node, dummy):
                        board[i][j] = 'X'
```

参考文章：

[UNION-FIND算法应用](https://labuladong.gitee.io/algo/2/19/38/)

# 990. Satisfiability of Equality Equations

这一题使用 Union-Find 算法解就比较优雅了。

题目给定一个数组`equations`，装着若干字符串表示的算式。每个算式`equations[i]`长度都是 4，而且只有等式或不等式两种情况：`a==b`或者`a!=b`，其中 a,b 可以是任意小写字母。算法要求是，如果`equations`中所有算式都不会互相冲突，返回`true`，否则返回`false`。

思路很明显，等式即是一种最简单的等价关系，满足自反性、对称性和传递性三种性质，`a==b`就等价于`a`和`b`连通。因此把 26 个字母作为节点，相等关系作为连通性，首先处理所有的等式划分好连通分量，然后使用不等式去验证关系即可：

```python
class UF:
    # Union-Find 的实现
    # 参考上上题
    # ...

class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        # 一共有26个字母
        uf = UF(26)
        # 暂存'a'的 ASCII 码
        id_a = ord('a')
        # 首先根据等式构建连通分量
        for eq in equations:
            if eq[1] == '=':
                # 连通等式两端两个字母对应的节点
                uf.union(ord(eq[0])-id_a, ord(eq[3])-id_a)
        
        # 遍历不等式进行验证
        for eq in equations:
            # 如果不等式两端字母连通则说明矛盾
            if eq[1] == '!' and uf.connected(ord(eq[0])-id_a, ord(eq[3])-id_a):
                return False
         
        return True
```

参考文章：

[UNION-FIND算法应用](https://labuladong.gitee.io/algo/2/19/38/)

# KRUSKAL 最小生成树算法

最小生成树算法主要有 Prim 算法和 Kruskal 算法两种，这两种算法都运用了贪心思想，但从实现上来说存在挺大差异，本部分主要使用的 Kruskal 算法

Kruskal 算法其实很容易理解和记忆，其关键是要熟悉并查集算法，如果不熟悉，建议先看下前几道题的 Union-Find 并查集算法。

## 最小生成树定义

「树」和「图」的根本区别：**树不会包含环，图可以包含环。**

如果一幅图没有环，完全可以用一棵树的形式来表示。专业点说，树就是「无环连通图」。

图的「生成树」就是在图中找一棵包含图中的所有节点的树。专业点说，生成树是含有图中所有顶点的「无环连通子图」。

对于加权图，每条边都有权重，所以每棵生成树都有一个权重和。

那么「最小生成树」就很好理解了，**在所有可能的生成树中，权重和最小的那棵生成树就叫「最小生成树」**。

> 一般来说，我们都是在**无向加权图**的情境下计算最小生成树的，所以现实场景中，图的边权重一般代表成本、距离这样的标量。

## Union-Find 并查集算法

图的生成树是含有其所有顶点的「无环连通子图」，最小生成树是权重和最小的生成树。

Union-Find 并查集算法，是用来用来高效处理图中连通分量的问题。

Union-Find 并查集算法的实现可以参考前几道题，主要运用`size`数组和路径压缩技巧提高算法效率。

## Kruskal 算法

所谓最小生成树，就是图中若干边的集合（后文称为`mst`，最小生成树的英文缩写），算法要保证这些边：

1. 包含图中的所有节点。

2. 形成的结构是树结构（即不存在环）。

3. 权重和最小。

Union-Find 算法主要负责做到前两条，关键在于第 3 点，如何保证得到的这棵生成树权重和最小。

这里主要用到的是贪心思路：

将所有边按照权重从小到大排序，从权重最小的边开始遍历，如果这条边和`mst`中的其他边不会形成环，则这条边是最小生成树的一部分，将它加入`mst`集合；否则，这条边不是最小生成树的一部分，不加入`mst`集合。

按照这个思路，最后`mst`集合中的边就形成了最小生成树。

下面用三个例题来应用 Kruskal 算法。

## 261. Graph Valid Tree

这道题其实就是判断一张无向图中是否有环。

遍历所有的边，首先判断边的两个节点是否在同一个连通分量中，如果是，则这条边的建立会形成环，返回`False`。

遍历完成之后连通分量的数量就是树的数量，所以最后需要检查连通分量的数量是否为 1。

```python
class UF:
    # 参考 Union-Find 算法部分

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        uf = UF(n)
        # 遍历所有边
        for edge in edges:
            if uf.connected(edge[0], edge[1]):
                # 如果两个节点已经连通，则会产生环
                return False
            else:
                # 否则，连通两个节点
                uf.union(edge[0], edge[1])

        # 保证最后只剩一个连通分量，所有节点都在一棵树里
        return uf.count() == 1
```

## 1135. Connecting Cities With Minimum Cost

这道题就可以转换为无向加权图中的最小生成树计算问题了，城市就是节点，而成本就是边的权重。

```python
class UF:
    # 参考 Union-Find 算法部分

class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        connections = sorted(connections, key=lambda x: x[2])
        uf = UF(n)
        min_cost = 0
        for city1, city2, cost in connections:
            if not uf.connected(city1-1, city2-1):
                # 如果两个城市未连通，则连通这条边
                uf.union(city1-1, city2-1)
                min_cost += cost
        
        # 最后检查所有城市是否是连通的
        if uf.count() > 1:
            return -1

        return min_cost
```

## 1584. Min Cost to Connect All Points

这道题和上一题的区别就在于边需要自己生成，但是也并不难。

遍历生成所有的边，然后直接应用 Kruskal 算法即可。

唯一的问题就是在这里点的坐标只是用来计算成本的，在数据结构中还是用输入列表里的索引来表示。

```python
class UF:
    # 参考 Union-Find 算法部分

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        # 计算点的数量
        n = len(points)
        uf = UF(n)
        # 遍历生成带权重的边
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                distance = abs(xj - xi) + abs(yj - yi)
                edges.append((i, j, distance))

        # 对边按照权重进行排序
        edges = sorted(edges, key=lambda x: x[2])

        # 贪心算法
        min_cost = 0
        for point1, point2, cost in edges:
            if not uf.connected(point1, point2):
                # 如果两个点没有连通，则将连通该条边
                uf.union(point1, point2)
                min_cost += cost

        return min_cost
```

# 277. Find the Celebrity

题目提供`n`个人之间的社交关系（两个人之间是否认识），设计算法找出这些人中的「名人」。

所谓「名人」有两个条件：

1. 所有其他人都认识「名人」；

2. 「名人」不认识其他任何人。

这是一个比较贴近实际的图相关的算法问题，社交关系，本质上可以抽象成一幅图。

我们可以把每个人看做图中的节点，「认识」就可以看做是节点之间的有向边，相当于社交软件上的「关注」，那么名人就是该图中一个特殊的节点。

题目提供了一个`knows(i, j)` API 来返回`i`是否认识`j`，本质上是在访问「邻接矩阵」。

最终题目要求返回名人的索引，或者`-1`来表示没有名人的情况。

## 暴力解法

简单的思路就是进行暴力穷举，把每个人遍历一遍，判断他和其他人的关系，最后筛选出名人。

## 时间优化

可以简单根据两个人之间的认识关系来对名人的候选名单进行筛选。

两个人之间的认识情况无非四种，以两个人`i`和`j`来举例：

1. 如果`i`认识`j`，`j`不认识`i`，那么`i`一定不是名人；

2. 如果`i`不认识`j`，`j`认识`i`，那么`j`一定不是名人；

3. 如果两个互相认识，那么两个人都不是名人；

4. 如果两个人互相不认识，那么两个人都不是名人。

根据如上四条逻辑，可以先任意比较`n-1`对人之间的关系，每次比较都能筛掉至少一个人，如果图里有名人，那么`n-1`次比较之后剩下的人一定是名人，然后我们验证一下最后这个人和其他所有人的关系即可。

优化后时间复杂度为$O(N)$。

## 空间优化

根据上一条时间优化的结果，我们可以只设定两个变量`cand`和`other`，两者之间淘汰之后交替赋值即可。优化后空间复杂度为$O(1)$。

## 参考代码

```python
# The knows API is already defined for you.
# return a bool, whether a knows b
# def knows(a: int, b: int) -> bool:

class Solution:
    def findCelebrity(self, n: int) -> int:
        # 初始化候选人 cand
        cand = 0
        # 遍历所有人
        for other in range(1, n):
            if knows(cand, other):
                # 只需要验证 cand 是否认识 other
                # 如果认识，直接淘汰 cand
                # 否则，淘汰 other
                cand = other
        
        # 验证最后的 cand 是否是真正的名人
        for other in range(n):
            if cand == other:
                # 跳过自己的情况
                continue
            if (not knows(other, cand)) or knows(cand, other):
                # 如果存在 cand 认识其他人或者有人不认识 cand 的情况
                # 返回 -1
                return -1

        return cand 
```

参考链接：

[众里寻他千百度：名流问题](https://labuladong.gitee.io/algo/2/19/41/)

# DIJKSTRA 算法

DIJKSTRA 算法 的输入是一幅图`graph`和一个起点`start`，目的是返回一个记录从 start 起点到所有节点的最短路径权重的数组。

函数签名如下：

```python
def dijkstra(start: int, graph: List[List[tuple[int]]]) -> List[int];
```

其中`graph`的两种实现方式，邻接表和邻接矩阵均可，这里使用的是「带权重的邻接表」数据结构。只要能正确提供邻居节点和权重的信息即可。

其实现代码如下：

```python
def dijkstra(start: int, graph: List[List[tuple[int]]]) -> List[int]:
    ''' 输入一个起点 start，计算从 start 到其他节点的最短距离
    Args:
        start: 输入的起点
        graph: 包含权重的邻接表
    Returns:
        包含从 start 到其他节点的最短距离的列表
    '''
    # 定义：distTo[i] 的值就是起点 start 到达节点 i 的最短路径权重
    n = len(graph)
    dist_to = [float('inf') for _ in range(n)]

    # base case，start 到自己的距离为 0
    dist_to[start] = 0
    
    # 使用优先级队列决定节点遍历优先级，distFromStart 较小的排在前面
    # 从起点开始 BFS
    pqueue = [(0, start)]
    heapify(pqueue)

    while pqueue:
        curr_state = heapq.heappop(pqueue)
        curr_node_id = curr_state[1]
        curr_dist_from_start = curr_state[0]
        if curr_dist_from_start > dist_to[curr_node_id]:
            continue
        
        # 将 curNode 的相邻节点装入队列
        for neighbor in graph[curr_node_id]:
            neighbor_id = neighbor[0]
            dist_to_next_node = dist_to[curr_node_id] + neighbor[1]
            # 更新 dist_to 列表
            if dist_to[neighbor_id] > dist_to_next_node:
                dist_to[neighbor_id] = dist_to_next_node
                heapq.heappush(pqueue, (dist_to_next_node, neighbor[0]))

    return dist_to
```

## 保存路径权重

该算法可以看作是一个 BFS 算法的加权图特化版。比起无权图中会使用`visited`数组记录节点的访问情况，加权图中因为不同路径走到同一个点的权重和会不同（即便运用了贪心思想），我们无法保证第一次访问某一节点时对应的路径就是最小权重的路径，所以没有必要再使用`visited`数组，取而代之的是需要保存每条路径的权重和，用于计算后续路径的权重，以及比较和更新从起点到某一节点的最小权重和。

这里保存权重有很多种方法，参考链接提供了一种构造`Stage`类来存储节点`node_id`和到该节点的距离`dist_from_start`的方法。但是在 Python 里可以不构造类`Stage`，比如上面提供的代码，直接将元组`(node_id, dist_from_start)`加入队列就行了。

## 优先级队列

比起用在无权图上的 BFS，Dijkstra 算法使用**优先级队列**取代了普通队列，其好处是**每次出队列的都是待选节点中距离起点最近的节点**。也就是说，该算法每一步迈出的都是最小步子，延伸至最近的节点，事实上就是 BFS 的概念。如果不用优先级队列，Dijkstra 算法就不具备 BFS 的「广度优先」属性。 

这一特性保证了算法平等地开拓各条路径，这样就能更早找到权重最小路径，而越早发现权重最小路径，根据算法，就会有越多路径不需要遍历，就能节省更多时间。

在上述代码中，优先级队列是用最小堆的数据结构实现的，使用的 Python 包是`heapq`包。

## 只计算起点 start 到某一个终点 end 的最短路径

脑子里过一遍 Dijkstra 算法，我们可以发现，在某一个节点`i`第一次入队列的时候，所对应的那条路径并不一定是最短路径。因为在节点`i`入队列的时候，遍历的是节点`i`当前路径上的前一个节点。我们设这个节点为`i0`，节点`i0`与起点的距离才是所有当前路径中的最小距离路径。因此从节点`i0`到节点`i`时，两者间的权重`weight(i0, i)`可能导致该条路径的总权重高于其他路径，也就是说，此时其他路径有可能能以更小的代价能到达节点`i`。而根据算法，接下来出队列的就会是那些距离起点更近的节点。

总结就是，**某节点第一次入队列的时候找到的不一定是最短路径，但是当该节点第一次出队列时，对应的`dist_to_next_node`就一定是最短路径了**。因为是「第一次」出队列，所以目前路径中没有经过该节点的存在，而因为队列是优先级队列，所以该节点是目前路径中距离起点最近的路径，因此一定是最短路径。

所以如果只计算起点 start 到某一个终点 end 的最短路径，代码只需要加一行：

```python
# 输入起点 start 和终点 end, 返回起点到终点的最短距离
def dijkstra(start: int, end:int, graph: List[List[tuple[int]]]) -> int:
    
    # ...

    while pqueue:

        curr_dist_from_start, curr_node_id = heapq.heappop(pqueue)

        # 在这里加一个判断即可
        if curr_node_id == end:
            return curr_dist_from_start

        if curr_dist_from_start > dist_to[curr_node_id]:
            continue
        
        # ...

    # 如果运行到这里，说明 start 无法走到 end
    return -1
```

## 是否需要 Visited 数组？

在网上看到有的 Dijkstra 算法里会加上`visited`数组，然后在节点出队列时记录访问，在入队的时候检查节点的访问情况，像下面这样：

```python
def dijkstra(start: int, graph: List[List[tuple[int]]]) -> List[int]:

    # ...

    # 初始化 visited 集合
    visited = set()

    while pqueue:
        
        # ...
       
        # 如果访问过该节点则跳过
        if curr_node_id in visited:
            continue

        # 出队的时候记录节点访问
        visited.add(curr_node_id)

        if curr_dist_from_start > dist_to[curr_node_id]:
            continue
        
        # 将 curNode 的相邻节点装入队列
        for neighbor in graph[curr_node_id]:
            neighbor_id = neighbor[0]
            # 如果访问过则不入队
            if neighbor_id not in visited:
                dist_to_next_node = dist_to[curr_node_id] + neighbor[1]
                # 更新 dist_to 列表
                if dist_to[neighbor_id] > dist_to_next_node:
                    dist_to[neighbor_id] = dist_to_next_node
                    heapq.heappush(pqueue, (dist_to_next_node, neighbor[0]))

    return dist_to
```

事实上并没有必要，因为这和`dist_to`数组的应用是等价的。上个部分说到当某一节点第一次出队的时候，就说明得到了从起点到该节点的权重最小路径。而在每一个节点出队的时候，我们都有一个条件判断

```python
        if curr_dist_from_start > dist_to[curr_node_id]:
            continue
```

来判断当前路径是否大于最小路径，事实上等价于判断该节点是否访问过了。

在之后邻居节点入队的条件判断

```python
        if dist_to[neighbor_id] > dist_to_next_node:
                dist_to[neighbor_id] = dist_to_next_node
                heapq.heappush(pqueue, (dist_to_next_node, neighbor[0]))
```

也是同样的道理，一旦访问过该节点，那么其对应的最小权重路径已经找到，其在`dist_to`里的值一定是最小值，因此该判断一定为否，相当于存在于`visited`数组里了。

即便使用了`visited`数组，节点还是有可能会被多次访问的，这取决于图的具体情况。所以用`visited`数组稍微有点误导性。而这里我们可以看到`dist`数组的更新是在节点入队的时候就完成了。结合入队的判断条件可以在该节点入队后出队前减少一些该节点重复入队的情况，故而避免不必要的堆排序，比起使用`visited`数组的方法时间消耗更少。


## 时间复杂度分析

Dijkstra 算法的复杂度随着其内部数据结构的实现的不同而不同。假设图中的节点数量为`V`，边的数量为`E`，代码中的优先队列是用二叉堆实现的。那么由于在最差情况下，节点可能会重复进队，而这个队列内元素数量则和边的数量成正比，因此构造二叉堆的时间复杂度就是$O(Elog(E))$。同时邻接表中所有的边的信息都会被遍历一遍，因此总的时间复杂度就是：

$O(E+Elog(E))=O(Elog(E))$

## 743. Network Delay Time

这道题就是很简单的 Dijkstra 算法的应用，其中的优先级队列的实现是用 Python 中的 heapq 库，速度更快一些。

根据题目输入构造完图的邻接表之后，找到所有节点距离起点最远的那一个，就是所有节点连通的时间了。

```python
def dijkstra(start: int, graph: List[List[tuple[int]]]) -> List[int]:
    ''' 输入一个起点 start，计算从 start 到其他节点的最短距离
    Args:
        start: 输入的起点
        graph: 包含权重的邻接表
    Returns:
        包含从 start 到其他节点的最短距离的列表
    '''
    # 定义：distTo[i] 的值就是起点 start 到达节点 i 的最短路径权重
    n = len(graph)
    dist_to = [float('inf') for _ in range(n)]

    # base case，start 到自己的距离为 0
    dist_to[start] = 0
    
    # 使用优先级队列决定节点遍历优先级，distFromStart 较小的排在前面
    # 从起点开始 BFS
    pqueue = [(0, start)]
    heapify(pqueue)

    while pqueue:
        curr_dist_from_start, curr_node_id = heapq.heappop(pqueue)
        if curr_dist_from_start > dist_to[curr_node_id]:
            continue
        
        # 将 curNode 的相邻节点装入队列
        for neighbor in graph[curr_node_id]:
            neighbor_id = neighbor[0]
            dist_to_next_node = dist_to[curr_node_id] + neighbor[1]
            # 更新 dist_to 列表
            if dist_to[neighbor_id] > dist_to_next_node:
                dist_to[neighbor_id] = dist_to_next_node
                heapq.heappush(pqueue, (dist_to_next_node, neighbor[0]))

    return dist_to

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # 构造邻接表
        # 因为节点编号从 1 开始，表大小为 n + 1
        graph = [[] for _ in range(n+1)]
        for time in times:
            graph[time[0]].append((time[1], time[2]))
        
        # 调用 dijkstra 算法计算以节点 k 为起点到其他节点的最短路径
        dist_to = dijkstra(k, graph)

        # 找到最长的路径
        res = 0
        for dist in dist_to[1:]:
            if dist == float('inf'):
                # 如果有节点不可到达，返回 -1
                return -1
            res = max(res, dist)
        
        return res
```

## 1514. Path with Maximum Probability

这道题对比于常规的 Dijkstra 算法来说，只是把权重相加变成了概率相乘。

同时由于概率要取最大，而数据结构有时只能限制使用最小堆，所以需要把从起点到各个节点的概率取反即可。

```python
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        
        # 构造邻接表
        graph = [[] for _ in range(n)]
        for i, edge in enumerate(edges):
            proba = succProb[i]
            graph[edge[0]].append((edge[1], proba))
            graph[edge[1]].append((edge[0], proba))
        
        # 定义：proba_to[i] 的值就是起点 start 到达节点 i 的中最大概率
        proba_to = [0 for _ in range(n)]

        # base case，start 到自己的概率为 -1
        # 因为 Python 不提供最大堆的实现
        # 所以所有概率用对应的负数表示
        # 最终输出最小值
        proba_to[start] = -1

        # 使用优先级队列决定节点遍历优先级，proba_from_start 较小的排在前面
        pqueue = [(-1, start)]
        heapify(pqueue)

        while pqueue:
            curr_proba_from_start, curr_node_id = heapq.heappop(pqueue)
            
            if curr_node_id == end:
                return - curr_proba_from_start
            
            if curr_proba_from_start > proba_to[curr_node_id]:
                continue

            # 将 curNode 的相邻节点装入队列
            for neighbor in graph[curr_node_id]:
                neighbor_id = neighbor[0]
                proba_to_next_node = - abs(proba_to[curr_node_id] * neighbor[1])
                # 更新 proba_to 列表
                if proba_to[neighbor_id] > proba_to_next_node:
                    proba_to[neighbor_id] = proba_to_next_node
                    heapq.heappush(pqueue, (proba_to_next_node, neighbor[0]))

        return 0
```

## 1631. Path With Minimum Effort

这道题也是使用 Dijkstra 算法做，稍微对代码进行更改即可。

首先，问题变成二维，对于节点的描述变成二元坐标`(i, j)`。

在路径前进的过程中，需要记录的是路径上的体力消耗，Effort，也就是最大高度差。

更改结果数组`min_effort`的定义：**`min_effort[i][j]`保存的是从起点到点`(i,j)`的最小体力消耗**，也就是需要从所有路径中挑出体力消耗最少的那一条。颇有动态规划的味道。

而实际上这就是动态规划。

```python
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        
        rows = len(heights)
        cols = len(heights[0])
        
        # min_effort[i][j] 的值就是起点 (0,0) 到达位置 (i,j) 的最小体力消耗
        min_effort = [[float('inf')]*cols for _ in range(rows)]

        # base case，到达起点的最小 effort 是 0
        min_effort[0][0] = 0
        
        # 使用优先级队列决定节点遍历优先级，effort 较小的排在前面
        # 从起点开始 BFS
        pqueue = [(0, 0, 0)]
        heapify(pqueue)

        while pqueue:
            # 取出队列中最小 effort 的坐标
            curr_effort, curr_row, curr_col = heappop(pqueue)

            if curr_row == rows - 1 and curr_col == cols - 1:
                return curr_effort

            # 如果已经存在消耗更小的路径，则跳过
            if curr_effort > min_effort[curr_row][curr_col]:
                continue

            # 将相邻上下左右四个位置装入队列
            for dist_row, dist_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # 计算下个位置的坐标
                next_row, next_col = curr_row + dist_row, curr_col + dist_col
                if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                    # 如果坐标不合法则跳过
                    continue
                
                # 计算当前位置和下个位置的高度差
                new_diff = abs(heights[next_row][next_col] - heights[curr_row][curr_col])
                # 计算到下一个位置的 effort
                effor_next_step = max(new_diff, curr_effort)

                # 更新 min_effort 列表， 如果已存在 effort 更小的路径则不入队
                if min_effort[next_row][next_col] > effor_next_step:
                    min_effort[next_row][next_col] = effor_next_step
                    heappush(pqueue, (effor_next_step, next_row, next_col))
        
        return min_effort[-1][-1]
```

## 参考链接

[我写了一个模板，把 DIJKSTRA 算法变成了默写题](https://labuladong.gitee.io/algo/2/19/42/)