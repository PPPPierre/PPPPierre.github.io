---
title: LeetCode 刷题记录 - 数据结构之链表（Linked List） 
author: Stone SHI
date: 2021-12-27 22:26:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, Linked list]
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

参考链接：

[一文搞懂单链表的六大解题套路](https://labuladong.gitee.io/algo/2/17/16/)

# 21. Merge Two Sorted Lists

方法：双指针

小技巧：代码中还用到一个链表的算法题中是很常见的「虚拟头结点」技巧，也就是 `dummy` 节点。你可以试试，如果不使用 `dummy` 虚拟节点，代码会复杂很多，而有了 `dummy` 节点这个占位节点，可以避免处理空指针的情况，降低代码的复杂性。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        
        curr = dummy = ListNode()
        pt1 = list1
        pt2 = list2
        
        while  pt1 and pt2:
            if pt1.val > pt2.val:
                curr.next = pt2
                pt2 = pt2.next
            else:
                curr.next = pt1
                pt1 = pt1.next
            curr = curr.next
        
        if pt1:
            curr.next = pt1
        if pt2:
            curr.next = pt2
        
        return dummy.next
```

# 23. Merge k Sorted Lists

思路同上题，关键是如何每一步从`k`个`ListNode`里面选出最小的那一个。

这里使用了一个数据结构，叫做[优先级队列](https://labuladong.gitee.io/algo/2/20/50/)。

这个队列是基于二叉堆（最小堆）实现的，可以通过从队列中取出最小的`Node`然后把`Node.next`再添加进队列里来保证持续获得最小的`Node`。

`queue`这个包就提供了可以直接调用的`PriorityQueue`类。

由于官方的`ListNode`类没有重载比较符，所以直接把链表节点加入队列会报错。

为了保证队列能正常运作，使用**三元元组**来作为存储对象，保证第二个元素唯一即可（这里用的是链表索引）。

因为元组的比较规则是按照元素逐个比较，当前元素相等的话会比较第二个，所以只要第二个元素唯一，就不会比较第三个元素，即我们的`ListNode`。

参考代码如下：

```python
import queue

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        
        pqueue = queue.PriorityQueue()
        curr = dummy = ListNode()
        
        for i, head in enumerate(lists):
            if head is not None:
                pqueue.put((head.val, i, head))
        
        while not pqueue.empty():
            _, i, curr.next = pqueue.get()
            curr = curr.next
            if curr.next is not None:
                pqueue.put((curr.next.val, i, curr.next))
            
        return dummy.next
```

其实使用`list`类加上`heapq`包也可以实现优先级队列， 并且这个方法更快。代码参考如下，这里元组第二个元素用`count`来保证唯一性：

```python
import heapq

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        curr = head = ListNode(0)
        queue = []
        count = 0
        for l in lists:
            if l is not None:
                count += 1
                heapq.heappush(queue, (l.val, count, l))
        while len(queue) > 0:
            _, _, curr.next = heapq.heappop(queue)
            curr = curr.next
            if curr.next is not None:
                count += 1
                heapq.heappush(queue, (curr.next.val, count, curr.next))
        return head.next 
```

参考资料：

[优先级队列和堆（一）](https://zhuanlan.zhihu.com/p/355317948)

# 19. Remove Nth Node From End of List

最简单的思路就是遍历两遍，第一遍确定节点数量，第二遍寻找倒数第`n`个节点。

但是我们肯定是希望只遍历一遍的。思路并不难，用两个指针去遍历，当第一个指针已经走了`n`步之后，再让第二个指针从头指针跟上，这样一来，当第一个指针走到末尾的时候，第二个指针刚好走到倒数第`n`个节点的位置。

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # 为方便代码，创建一个虚拟头指针
        dummy = ListNode()
        dummy.next = head
        
        # 两个指针的起始点都是dummy
        pt1 = dummy
        pt2 = dummy
        
        # 首先让指针pt1走出n步
        count = 0
        while count < n:
            pt1 = pt1.next
            count += 1
        
        # 指针pt2和pt1开始同步走
        while pt1.next:
            pt1 = pt1.next
            pt2 = pt2.next
        
        # 删除节点并返回
        pt2.next = pt2.next.next
        return dummy.next
```

# 876. Middle of the Linked List

技巧：快慢指针

快指针一次走两步，慢指针一次走一步。

```python
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(next=head)
        
        pt_fast = pt_slow = dummy
        
        while pt_fast.next and pt_fast.next.next:
            pt_fast = pt_fast.next.next
            pt_slow = pt_slow.next
            
        return pt_slow.next
```

# 141. Linked List Cycle

技巧：快慢指针

当快指针追上慢指针的时候，说明存在环，否则快指针会遍历到链表末尾。

``` python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        
        dummy = ListNode(next=head)
        
        pt_slow = pt_fast = dummy
        
        while pt_fast.next and pt_fast.next.next:
            pt_fast = pt_fast.next.next
            pt_slow = pt_slow.next
            # 快慢指针相遇
            if pt_fast == pt_slow:
                return True
            
        return False
```

# 142. Linked List Cycle II

和上一题思路一样，快慢指针，只是在两指针相遇之后还需要求得环开始的节点。

这里需要一点点数学证明，可以得到的结论是：

当快慢指针相遇时，让其中一个指针返回`dummy`节点，两个指针再同速继续前进，那么再次相遇的位置就是环开始的节点。

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        dummy = ListNode(next=head)
        
        pt_slow = pt_fast = dummy
        
        while pt_fast.next and pt_fast.next.next:
            pt_fast = pt_fast.next.next
            pt_slow = pt_slow.next
            # 快慢指针相遇
            if pt_fast == pt_slow:
                # 慢指针返回dummy节点
                pt_slow = dummy
                # 两指针同速前进
                while pt_fast.next:
                    pt_fast = pt_fast.next
                    pt_slow = pt_slow.next
                    # 再次相遇即为环开始节点
                    if pt_fast == pt_slow:
                        return pt_slow
            
        return None
```

# 160. Intersection of Two Linked Lists

有两个思路，第一个比较传统，但是效果好：

各遍历一遍两个链表，计算出长度差，然后让两个链表从同一个起点开始前进，就能同一时间进入交叉节点。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        # 计算A链表长度
        a_len = 0
        a = headA
        while a:
            a_len += 1
            a = a.next

        # 计算B链表长度
        b_len = 0
        b = headB
        while b:
            b_len += 1
            b = b.next
        
        # 统一起始节点
        a = headA
        b = headB
        lag = a_len - b_len
        if lag > 0:
            for i in range(lag): a = a.next
        else:
            for i in range(abs(lag)): b = b.next

        # 同时前进
        while a and b and a != b:
            a = a.next
            b = b.next
        return a
```

第二个思路也是相似，同时进入交叉点。

让指针`p1`遍历完`headA`之后去遍历`headB`，然后`p2`遍历完`headB`去遍历`headA`，这样两个指针就能同时进入交叉节点了。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        
        p1 = headA
        p2 = headB
        
        while p1 != p2:
            if p1 is None:
                p1 = headB
            else:
                p1 = p1.next
            if p2 is None:
                p2 = headA
            else:
                p2 = p2.next
        
        return p1
```

# 206. Reverse Linked List

常规思路很简单，直接遍历，用临时变量存储前后节点，每一个循环里反转前后节点的指向关系。

```python 
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        node = head
        pre_node = None
        
        # 循环遍历
        while node:
            
            # 临时存储后一个节点
            curr_head = node.next

            # 当前节点指向前一个结点
            node.next = pre_node

            # 更新临时节点
            pre_node = node
            node = curr_head
        
        return pre_node
```

同时也可以用递归实现，虽然比较慢，但是对思维是很好的锻炼。

递归函数`reverse`定义：**输入一个节点`head`，将「以`head`为起点」的链表反转，并返回反转之后的头结点。**

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        if head is None: return None
        
        # 使用递归函数
        def reverse(node):

            # base case
            if node.next is None:
                return node
            
            # 反转以下一个节点为头节点的链表
            last = reverse(node.next)

            # 更改指向
            node.next.next = node
            node.next = None
            return last
        
        return reverse(head)
```

# 92. Reverse Linked List II

上一题的进阶版，引入参数`left`和`right`来翻转节点索引从`left`到`right`范围内的链表（头节点索引为 1 ）。

需要分步实现两个递归函数：

1. `reverseN(head, n)`：翻转链表前`N`个节点。

2. `reverseBetween(head, m, n)`：给出一个索引区间 `[m,n]`（索引从 1 开始），仅仅反转区间中的链表节点的函数。

两个函数的实现思路如下，

`reverseN(head, n)`的思路是将`n`作为递归函数的参数传递下去，每次调用在当前节点进行翻转，并将`n`减一，到达`n == 1`的 base case 时返回并记录**后驱节点**，为了将翻转后的链表的尾巴接上去。

`reverseBetween(head, m, n)`的思路是持续前进到反转的起点，也就是直到触发`m == 1`的 base case，从当前的`head`位置调用`reverseN(head, n)`。

```python
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        
        successor = None
        
        # 翻转前N个节点
        def reverseN(head, n):

            # 后驱节点
            nonlocal successor

            # base case
            if n == 1:
                successor = head.next
                return head
            
            # 逐节点翻转
            last = reverseN(head.next, n-1)
            head.next.next = head
            head.next = successor
            return last
        
        # 翻转从m到n的节点
        def reverseBetween(head, m, n):
            if m == 1:
                return reverseN(head, n)
            else:
                head.next = reverseBetween(head.next, m-1, n-1)
                return head
        
        return reverseBetween(head, left, right)
```

递归操作链表并不高效。和迭代解法相比，虽然时间复杂度都是 O(N)，但是迭代解法的空间复杂度是 O(1)，而递归解法需要堆栈，空间复杂度是 O(N)。

所以递归操作链表可以作为对递归算法的练习，但是考虑效率的话还是使用迭代算法更好。

# 25. Reverse Nodes in k-Group

使用迭代遍历的思路来做，比较简单。

首先实现上上题**206. Reverse Linked List**的翻转整个链表的函数，然后遍历链表。

使用`count`来计数，当满足条件`count == k`时触发翻转，但是需要提前存下上一个`group`的尾节点以及下一个`group`的头节点。

翻转完后返回新的当前`group`的头节点和末节点，再和前后两个`group`重新建立连接即可。

整体时间复杂度为$O(N)$。

参考代码如下：

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
         
        # 翻转整个链表的函数
        def reverse(head):
            curr_node = head
            pre = None
            while curr_node:
                temp = curr_node.next
                curr_node.next = pre
                pre = curr_node
                curr_node = temp
            return pre, head
        
        # 创建虚拟头节点
        dummy = ListNode(next = head)
        curr_node = end_of_last_group = dummy
        # 使用count计数
        count = 0
        while curr_node.next:
            curr_node = curr_node.next
            count += 1
            # 当count等于k时触发翻转
            if count == k:
                # 保存下一个group的头节点
                head_of_new_group = curr_node.next
                # 断开两个group之间的链接
                curr_node.next = None
                # 翻转整个group的节点
                curr_head, curr_end = reverse(end_of_last_group.next)
                # 上一个group的尾节点接上反转后的group
                end_of_last_group.next = curr_head
                # 连上下一个group的头节点
                curr_end.next = head_of_new_group
                # 重置count
                count = 0
                # 现在的group的尾节点成为新的end_of_last_group
                end_of_last_group = curr_node = curr_end
        
        return dummy.next
```

# 234. Palindrome Linked List

判断链表是不是回文链表。

这道题大可以直接把链表的元素存储到数组中再判断，但是这样就失去了链表的意义同时时间复杂度也不低。

作为一个练习，其实这道题可以很好地应用之前关于链表的操作知识：寻找中点，翻转链表等等。

思路也不难：

1. 首先使用快慢指针找到链表中点

2. 然后翻转后半段链表

3. 同时遍历两个链表，逐个节点比较即可

```python
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        
        # 找到链表中点
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # 现在慢指针所指向的是后半段链表
        # 翻转后半段链表
        prev = None
        while slow:
            slow.next, slow, prev = prev, slow.next, slow
        
        # 现在 prev 是后半段链表的头节点
        first = head
        second = prev
        while first and second:
            if first.val != second.val:
                return False
            first = first.next
            second = second.next
        
        return True
```

# 24. Swap Nodes in Pairs

一道相对简单的题目，每两个 node 交换一下位置。

思路就是创建虚拟头指针，从虚拟指针开始，以步长为二遍历链表，每次都交换指针后两个链表的位置。

基本要注意的就是交换的顺序问题。

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        # 创建虚拟头指针
        node = dummy = ListNode(next=head)
        # 遍历直到倒数第二个或者最后一个节点
        while node.next and node.next.next:
            # 交换后两个节点的位置
            next_node = node.next.next
            node.next.next = next_node.next
            next_node.next = node.next
            node.next = next_node
            # 跳过一个节点
            node = next_node.next

        return dummy.next
```

## 61. Rotate List

读题后可以使用**前后双指针**解决。

唯一需要注意的细节就是`k`的取值上限远大于链表的长度上限，所以需要先取模求出去掉重复旋转的实际偏转值`k_pure`。

然后使用前后相隔`k_pure`步的两个指针遍历链表，当走得快的指针到底之后，把两个指针之间的节点全部接在链表头部即可完成任务。

```python
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 特殊情况
        if k == 0 or not head:
            return head
        # 虚拟头指针
        dummy = ListNode(next=head)
        # 计算节点数量
        count = 0
        node = dummy.next
        while node:
            node = node.next
            count += 1
        # 计算取余后的实际旋转节点数
        k_pure = k % count
        # 如果旋转数为 0，则返回原链表
        if k_pure == 0:
            return head
        # 使用前后相隔 k_pure 个节点的双指针
        # 指针 1 走了 k_pure 步之后指针 2 开始走
        count = 0
        node_1 = dummy
        while (count < k_pure):
            count += 1
            node_1 = node_1.next
        node_2 = dummy
        # 同步前进
        while(node_1.next):
            node_1 = node_1.next
            node_2 = node_2.next
        # 指针1到头时将两个指针之间的部分链表前部
        node_1.next = dummy.next
        dummy.next = node_2.next
        node_2.next = None
        return dummy.next
```