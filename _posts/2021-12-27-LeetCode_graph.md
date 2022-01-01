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