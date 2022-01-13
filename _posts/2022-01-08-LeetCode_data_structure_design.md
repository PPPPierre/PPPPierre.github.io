---
title: LeetCode 刷题记录 - 数据结构设计
author: Stone SHI
date: 2022-01-08 15:28:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, Data structure design]
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

# 146. LRU Cache

这道题的主要目标是构造一个数据结构来实现 LRU 缓存机制。

当内存的空间满了之后，我们总是需要按照一定的原则来删除内存中的数据，为新数据腾出空间。

LRU 就是这样一种机制，LRU 的全称是 Least Recently Used，也就是根据内容的最后一次使用时间来选择数据删除的优先级。

相比起最近使用过的数据，我们会优先选择删除好久没用过的数据。

所以这道题的数据结构设计思路就是：**双链表**加上**字典**的组合。

1. 使用**链表**是为了实现优先级的结构，即将最近访问的数据提到链表最前端，自然未使用的数据就会向链表末端下沉的。同时链表也能实现时间复杂度为$O(1)$的插入。双头的原因是为了方便删除节点。节点内需要同时存储上数据的`key`和`value`。存储`key`的原因在于删除节点时需要同时删除字典内的`key`，如果在节点中不存储`key`的话，在删除节点时是做不到删除字典中对应数据的。

2. **字典**是为了实现时间复杂度为$O(1)$的访问。注意字典中`key`对应存储的不是数据的`value`，而是对应的节点，这样才能在访问的同时实现将对应的节点推到链表的最前端。

参考代码如下：

```python
class Node:
    def __init__(self, key=None, val=0, next_node=None, pre_node=None):
        # 双链表结构，存储数据的键值对
        self.key = key
        self.val = val
        self.next = next_node
        self.pre = pre_node

class LRUCache:

    def __init__(self, capacity: int):
        # 初始化字典
        self.dict = dict()
        # 初始化头节点和尾节点
        self.head = Node(val='head')
        self.tail = Node(val='tail', pre_node=self.head)
        self.head.next = self.tail
        # 初始化容量和数据数量
        self.count = 0
        self.capacity = capacity

    def get(self, key: int) -> int:
        if not key in self.dict:
            # 如果不存在 key，返回 -1
            return -1
        else:
            # 如果存在，使用字典返回 value，时间复杂度O(1)
            curr_node = self.dict[key]
            # 将该节点推到双链表最前端
            self.put_first(curr_node)
            return curr_node.val

    def put(self, key: int, value: int) -> None:
        if key not in self.dict:
            # 如果 key 不存在则在最前端创建新的节点
            # 首先判断存储量是否到上限
            if self.count == self.capacity:
                # 如果存储量到上限，则删除最后一个节点
                delete_node = self.tail.pre
                delete_key = delete_node.key
                self.tail.pre = delete_node.pre
                self.tail.pre.next = self.tail
                self.dict.pop(delete_key)
                del(delete_node)
            else:
                # 未到达上限，则 count +1
                self.count += 1
            # 创建新节点
            new_node = Node(key=key, val=value, next_node=self.head.next, pre_node=self.head)
            self.dict[key] = new_node
            self.head.next.pre = new_node
            self.head.next = new_node

        else:
            # 如果 key 早已存在，更改对应的 value 并且将该节点插入到链表最前端
            curr_node = self.dict[key]
            curr_node.val = value
            self.put_first(curr_node)

    def put_first(self, node):
        # 将节点插入链表最前端
        node.pre.next = node.next
        node.next.pre = node.pre
        node.next = self.head.next
        self.head.next.pre = node
        self.head.next = node
        node.pre = self.head
```


参考链接：

[算法就像搭乐高：带你手撸 LRU 算法](https://labuladong.gitee.io/algo/2/19/43/)

# 460. LFU Cache

这一题和上一题相似，也是实现一个缓存机制，但是这次是 LFU 机制：Last Frequently Used，也就是按照使用频率顺序删除数据。

当内存达到上限的时候，我们优先删除访问次数最少的数据，如果碰到多个数据具有相同的使用频率，则删除最旧的数据。

首先按照题目要求（和上一题一样），我们需要实现以下功能：

1. 时间为$O(1)$的插入或更改数据并更新使用频率；
2. 时间为$O(1)$的访问数据；
3. 当内存已满，插入新数据时，以$O(1)$的时间复杂度删除使用频率最低数据删除；
4. 当删除数据时遇到多个数据，要能以$O(1)$时间找出最旧的数据删除；

上一题 LRU 机制可以满足第 1 条和第 2 条，但是没法满足 3 和 4。

显然需要对结构进行改进。

## 数据结构

构造如下数据结构来满足我们的需求：

1. 使用一个`dict`存储`key`到`value`的映射，用以访问数据
2. 使用一个`dict`存储`key`到`freq`的映射，用以更新数据的使用频率
3. 使用一个`dict`存储`freq`到`key`的映射，用以按照时序删除`key`
4. 使用临时变量`min_freq`来存储目前的最小使用频率

这里需要注意，从`freq`到`key`的映射需要满足如下三个条件：

1. 需要满足**一对多**的映射
2. 能够反映`key`之间的**时序**关系
3. 能够以$O(1)$的时间插入或删除`freq`对应的一个`key`

Python 的`collections`包中有一个数据结构能够满足上述条件，那就是`collections.OrderedDict`。

我们构建一个`dict`，使得`freq`为键，对应的值是一个`OrderedDict`，其中存储的是所有频率为`freq`的`key`。

`OrderedDict`是一个类似于我们上一题构造的既可以满足**时序**要求同时也可以以常数时间访问的结构。可以理解为链表和哈希表的结合体。

它的`popitem(last=True)`方法可以弹出最新存储的键值对，如果设置参数`last=False`则是弹出最旧的键值对。

为了方便初始化从`freq`到`key`映射的默认值，我使用了`collections.defaultdict`这个结构，主要是当`key`不存在的时候可以直接初始化一个默认的`OrderedDict`变量。

## 代码实现

剩下的就是代码部分了，首先从`put`和`get`两个基本功能开始写，其中能抽象的部分先抽象成高级函数。

然后写出高级函数的函数头，其中用注释注明思路和需要实现的步骤。

然后一个模块一个模块实现并测试，就可以条理清晰地写出难出 bug 的代码了。

```python
class LFUCache:

    def __init__(self, capacity: int):
        # 需要构造三个表来实现功能
        self.key_to_val = dict()
        self.key_to_freq = dict()
        self.freq_to_key = collections.defaultdict(collections.OrderedDict)
        self.min_freq = 0
        self.count = 0
        self.capacity = capacity

    def get(self, key: int) -> int:
        if self.capacity == 0:
            return -1

        if key in self.key_to_val:
            # 如果 key 存在
            # key 的访问频率加一
            self.increase_freq(key)
            # 返回对应 val
            return self.key_to_val[key]
        else:
            # 不存在返回 -1
            return -1

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.key_to_val:
            # 如果 key 存在
            # 更改 key 的 val
            self.key_to_val[key] = value
            # key 的访问频率加一
            self.increase_freq(key)

        else:
            # 如果 key 不存在
            # 判断容量是否到达上限
            if self.count == self.capacity:
                # 删除最低频率使用过的 key
                self.delete_min_freq_key()
            else:
                self.count += 1
            # 增加新 key
            self.add_new_key(key, value)
    
    def increase_freq(self, key):
        # 增加 key 的访问频率
        curr_freq = self.key_to_freq[key]
        # KF 字典中，需要改变 key 对应的 freq
        self.key_to_freq[key] += 1
        # FK 字典中，需要将 key 从 freq 的 OrderedDict 中删除
        self.freq_to_key[curr_freq].pop(key)
        # 将 key 添加到 freq+1 的 OrderedDict 中
        self.freq_to_key[curr_freq+1][key] = key
        # 判断 freq 是不是 min_freq
        if curr_freq == self.min_freq:
            # 如果是，则判断原 OrderedDict 中是否还有元素
            if not self.freq_to_key[curr_freq]:
                # 如果原 dict 为空，min_freq += 1
                self.min_freq += 1
    
    def delete_min_freq_key(self):
        # 删除使用频率最低的 key
        # 根据 min_freq 找到最小的 freq 对应的 key list
        # 使用 OrderedDict 的 pop() 方法删除最近的 key 并得到值
        delete_key = self.freq_to_key[self.min_freq].popitem(last=False)[0]
        # 删除另外两个 dict 中的 key
        self.key_to_freq.pop(delete_key)
        self.key_to_val.pop(delete_key)
        # 由于调用该函数之后会紧接着新创建一个数据
        # 该数据的 freq 必为 1，所以不用更新 min_freq

    def add_new_key(self, key, value):
        # 添加一个新的值为 value， freq 为 1 的 key
        # KV 字典中，新增 key-val 键值对
        self.key_to_val[key] = value
        # KF 字典中，新增 key-1 键值对
        self.key_to_freq[key] = 1
        # VF 字典中，在 freq 为 1 的 OrderedDict 中增加 key
        self.freq_to_key[1][key] = key
        # 将 min_freq 记为 1
        self.min_freq = 1
```

## 参考链接

[算法就像搭乐高：带你手撸 LFU 算法](https://labuladong.gitee.io/algo/2/19/44/)

