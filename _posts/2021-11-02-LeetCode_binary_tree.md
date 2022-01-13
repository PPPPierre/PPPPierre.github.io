---
title: LeetCode 刷题记录 - Binary Tree
author: Stone SHI
date: 2021-11-02 5:47:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, Binary tree]
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

# 95. Unique Binary Search Trees II

给定一个整数 n，生成一个包含所有由值为 1 到 n 的节点组成的 BST 的 list。

已知 BST 的性质：每一个节点的左子树节点的值都小于该节点，其右子树节点的值都大于等于该节点。

那么根据该性质，则可以构建一个递归函数`helper`，该函数以两个指针`i`，`j`作为输入，输出由`range(i,j+1)`这个范围的数构成的所有可能 BST 的 list。

在该递归函数内部，首先遍历`range(i,j+1)`，选择一个数`k`作为`root`，然后遍历`helper(i, k-1)`输出的元素作为左子树，遍历`helper(k+1, j)`输出的元素作为右子树，然后就可以构造新的节点`TreeNode(val=k, left=left_node, right=right_node)`。最后将这些新构造的节点逐个添加到 list 即可。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:        
        def helper(i: int, j: int) -> List[Optional[TreeNode]]:
            if i == j: return [TreeNode(val=i)]
            
            if i > j: return [None]
            
            node_list = []
        
            for k in range(i, j+1):
                left_list = helper(i, k-1)
                right_list = helper(k+1, j)
                for left_node in left_list:
                    for right_node in right_list:
                        node_list.append(TreeNode(val=k, left=left_node, right=right_node))
            return node_list
        return helper(1, n)
```

# 96. Unique Binary Search Trees

根据整数 n，给出所有由值为从 1 到 n 的节点组成的 BST 的数量。

与上一题思路相似，设计一个`_rec(i:int, j:int) -> int`的递归函数。其中两个指针`i`、`j`指示目标 BST 对应的节点值的头和尾。

事实上总的 BST 的数量只和 `j+1-i`也就是与数组的长度有关，因此根据 DP 的思路，建一个表，将计算的内容存储到表内可以大量节省时间。

```python
class Solution:
    def numTrees(self, n: int) -> int:
        
        # 创建储存结果的表
        num_dict = {}
        
        def _rec(i:int, j:int) -> int:
            
            if i > j+1 or j+1 <= i or i == j: return 1 # 无节点或者只有一个节点返回 1
            
            if j+1-i in num_dict: return num_dict[j+1-i] # 如果结果已计算则查表
            else:
                # 如果结果未计算，则遍历i到j+1，再对其子树对应数组调用递归，返回数量为左子树数量乘以右子树的数量
                num_total = 0
                for k in range(i, j+1):
                    num_left_tree = _rec(i, k-1)
                    num_right_tree = _rec(k+1, j)
                    num_total += num_left_tree * num_right_tree
                    
                # 将结果存入表
                num_dict[j+1-i] = num_total
                return num_total
            
        return _rec(0, n-1)
```

# 98. Validate Binary Search Tree

判断一个树是否是一个 BST。

利用 BST 的中序遍历结果是一个有序数组的特性，对其进行中序遍历，将遍历的每个节点的值插入于一个`val_list`的末尾。

同时遍历到每一个节点时都判断其值是否大于`val_list`中的最后一个元素，否则输出`False`。

若于该节点的判断没有问题，则最后递归函数的结果取决于左子树和右子树的遍历结果的与运算。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        val_list = []
        def _rec(l: List[int], root: Optional[TreeNode]) -> bool:

            # 判断是否为空节点
            if not root:
                return True

            # 递归左子树
            is_BST_left = _rec(val_list, root.left)

            # 中序遍历操作
            if val_list and root.val <= val_list[-1]:
                return False
            val_list.append(root.val)

            # 递归右子树
            is_BST_right = _rec(val_list, root.right)
            
            return is_BST_left and is_BST_right

        return _rec(val_list, root)
```

# 99. Recover Binary Search Tree

找出 BST 中唯一交换的两个节点并复原。

思路如上一题，中序遍历二叉树，通过将节点的值与前一个节点的值进行比较，找到出现问题的两个节点，将其交换即可。

所谓的有问题，其实是节点的值小于中序遍历的上一个节点。

唯一需要注意的是当两个节点在中序遍历的结果中相邻时，其交换后只能找到一个有问题的节点。

因此在遍历结束进行判断并追加操作即可。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        
        def _rec(root: Optional[TreeNode]):
            
            nonlocal node_1, node_2, pre_node
            
            if not root:    
                return
            
            # 遍历左子树
            _rec(root.left)
            
            # 中序遍历操作
            if pre_node and root.val < pre_node.val:
                if not node_1:
                    # 如果node_1没赋值，说明找到的是第一个交换节点
                    node_1 = pre_node
                    node_2 = root
                else:
                    # 如果node_1已经赋值，则说明目前遍历的是第二个节点
                    # 于是可以在此交换两节点的值，并结束递归
                    temp = node_1.val
                    node_1.val = root.val
                    root.val = temp
                    node_2 = None
                    return
            pre_node = root
            
            # 遍历右子树
            _rec(root.right)
            
        node_1, node_2, pre_node  = None, None, None
            
        _rec(root)
        
        if node_2:
            temp = node_1.val
            node_1.val = node_2.val
            node_2.val = temp
```

# 102. Binary Tree Level Order Traversal

思路：队列 + 前序遍历

```Python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        
        node_queue = []
        res_list = []
        
        def _rec(root: Optional[TreeNode], lv:int):
            
            if not root: return
            
            if len(res_list) < lv + 1:
                res_list.append([root.val])
            else:
                res_list[lv].append(root.val)
            
            node_queue.append((root.left, lv + 1))
            node_queue.append((root.right, lv + 1))
            
            _rec(*node_queue.pop(0))
            _rec(*node_queue.pop(0))
            
        _rec(root, 0)
        return res_list
```

# 103. Binary Tree Zigzag Level Order Traversal

与上一题思路一致

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        
        res_list = []
        node_queue = []
        
        def _rec(root: Optional[TreeNode], lv:int):
            
            if not root: return
            
            if lv >= len(res_list):
                res_list.append([root.val])
            else:
                if lv%2 == 0:
                    res_list[lv].append(root.val)
                else:
                    res_list[lv].insert(0, root.val)
            
            node_queue.append((root.left, lv + 1))
            node_queue.append((root.right, lv + 1))
            
            _rec(*node_queue.pop(0))
            _rec(*node_queue.pop(0))
            
        _rec(root, 0)
        return res_list
```

# 105. Construct Binary Tree from Preorder and Inorder Traversal

# 106. Construct Binary Tree from Inorder and Postorder Traversal

# 107. Binary Tree Level Order Traversal II

# 113. Path Sum II

给定一个二叉树和一个`targetSum`，以数组形式返回所有从根节点到叶节点的值相加等于`targetSum`的路径。

因此从根节点出发，遍历其到达任意一个叶节点的路径，在每一个叶节点处判断，如果该路径上的节点值的和等于`targetSum`，则逐节将各个节点的值插入到结果的`list`中返回。

代码如下：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        
        # 判断二叉树是否为空
        if not root: return []
        
        # 构造一个递归函数来遍历二叉树
        # 信息向下传递（从根到叶）时，将目前累加的和叠加并向下传递
        # 到达叶节点时将判断累加和是否等于 targetSum
        # 信息向上传递（从叶到根）时，将以 list 形式返回所有经过该点的合法路径，各节点将自身的值插入其中并继续向上传递
        def _rec(root: Optional[TreeNode], current_sum) -> List[List[int]]:
            
            ### 信息向下传递过程中的预处理
            nonlocal targetSum
            # 累加路径和
            current_sum += root.val
            # 判断是否为叶节点
            if not root.left and not root.right and current_sum == targetSum:
                return [[root.val]]
            
            list_left, list_right = [], []
            
            ### 递归调用于左子树
            if root.left: list_left = _rec(root.left, current_sum)
            ### 递归调用于右子树
            if root.right: list_right = _rec(root.right, current_sum)
                
            ### 信息向上传递过程
            # 将左子树和右子树返回的可能路径合并到一个 list 中
            res_list = list_left + list_right
            # 对 list 中的每一个路径插入本节点的值
            for node_list in res_list:
                node_list.insert(0, root.val)
            return res_list
        
        return _rec(root, 0)
```

其实从这一题中可以抽象出一个二叉树遍历的整体逻辑，那就是信息的向下与向上传递的过程。

在这一题中，路径目前累加的和作为信息向下（从根到叶）传递，而对于这个信息的预处理过程则在递归调用子树之前。

当我们想把信息向下传递时，可以将该信息以递归函数的参数的形式传递。

而在叶节点判断完毕后，路径信息则以 list 的形式向上传递，而每一个节点对于信息的后处理则在递归调用子树之后。

当我们想把信息向上回传时，通常将其作为递归函数的返回值来传递。

因此整体框架如下：

```python

def _rec(root: Optional[TreeNode], *info_to_end) -> info_return:
            
            ### 向下传递信息预处理
            # ...
            # *info_to_left, *info_to_right = pre_process(root, *info_to_end ...)
            # ...
            
            ### 递归调用于左子树
            if root.left: info_left = _rec(root.left, *info_to_left)

            ### 如果左右子树的传入信息还有因果性，则可以在此处处理

            ### 递归调用于右子树
            if root.right: info_right = _rec(root.right, *info_to_right)
                
            ### 向上传递信息后处理
            # ...
            # info_return = post_process(root, info_left, info_right, ...)

            return info_return

```

框架中的 `pre_process` `post_process` 函数只是将信息的预处理和后处理过程抽象化了，代表意义就是将当前节点的信息和上游传来的信息相结合处理后，再返回给下游，不论从父节点到子节点还是从子节点到父节点，都是一样的。

# 124. Binary Tree Maximum Path Sum

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        
        sum_max = -1001
        
        def _rec(root: Optional[TreeNode]) -> int:
            
            # 判断是否为空节点
            if not root: return 0
            
            nonlocal sum_max
            
            # 递归左子树
            max_left = _rec(root.left)
            
            # 递归右子树
            max_right = _rec(root.right)
            
            # 计算只取半边时该节点处最大值
            max_branch = max(root.val, root.val + max_left, root.val + max_right)
            
            # 与全局最大值作比较时还要考虑左右两子树合并的情况，即无法继续向上扩展的情况
            val_max = max(max_branch, root.val + max_left + max_right)
            if val_max > sum_max：sum_max = val_max
            
            # 返回可衔接的情况下的最大值
            return max_branch
        
        _rec(root)
        
        return sum_max
```

# 114. Flatten Binary Tree to Linked List

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        
        if not root: return None
        
        def _rec(root: Optional[TreeNode]) -> Optional[TreeNode]:
            
            end_left = end_right = root
            
            if root.left:
                end_left = _rec(root.left)
                end_right = end_left
            
            if root.right: end_right = _rec(root.right)
            
            if root.left:
                end_left.right = root.right
                root.right = root.left
                root.left = None
            
            return end_right
        
        _rec(root)
```

# 116. Populating Next Right Pointers in Each Node

这一题可以和前面某题结合起来讨论在二叉树问题中临时存储变量的需求。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        
        node_list = []
        
        def _rec(root: 'Node', lv:int):
            
            if not root: return
            
            nonlocal node_list
            
            if len(node_list) <= lv:
                node_list.append(root)
            else:
                node_list[lv].next = root
                node_list[lv] = root
            
            _rec(root.left, lv + 1)
            _rec(root.right, lv + 1)
        
        _rec(root, 0)
        for node in node_list:
            node.next = None
        return root
```

# 129. Sum Root to Leaf Numbers

# 236. Lowest Common Ancestor of a Binary Tree

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        self.val_q = q.val
        self.val_p = p.val
        self.ans = None
        self.post_order(root)
        
        return self.ans
        
    def post_order(self, root: 'TreeNode') -> bool:

        if not root: return False

        is_cur_node = root.val == self.val_q or root.val == self.val_p
        
        left_found = self.post_order(root.left)

        right_found = self.post_order(root.right)

        if is_cur_node and (left_found or right_found):
            self.ans = root
            return False
        
        if left_found and right_found:
            self.ans = root
            return False

        return is_cur_node or left_found or right_found
```

# 297. Serialize and Deserialize Binary Tree

将节点一一编上序号，然后再得到前序遍历和中序遍历的序号 List，以及前序遍历的 val 的 List。
之后根据前序遍历和中序遍历的序号 List 还原二叉树结构，然后按照前序遍历的顺序把 val 的值一个一个填进去即可。
tip. 编号是为了防止出现因为 Node 值重复从而导致无法还原树结构的情况。

# 331. Verify Preorder Serialization of a Binary Tree

```python
class Solution(object):
    def isValidSerialization(self, preorder):
        slot = 1
        
        for c in preorder.split(','):
            slot -= 1 #each elemet consumes a slot
            if slot<0: return False
            if c!='#': slot += 2 #each non-null node also create 2 slot
            
        return slot==0 #all slots should be fill
```

# 337. House Robber III

个人目前的主要问题是对问题本身的思考不够准确和清晰，导致最终写出来的代码是符合脑子所想的解决方案，但是脑子里的解决方案本身却有问题。

如果把解题的逻辑归纳如下：

题目 -> 解决方案（脑子） -> 代码

目前从第一步到第二步的过程还需要锻炼。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        
        def _rec(root):
            
            if not root: return 0, 0
                
            max_left_theif,  max_left_no_theif = _rec(root.left)
            max_right_theif,  max_right_no_theif = _rec(root.right)
            
            max_theif = root.val + max_left_no_theif + max_right_no_theif
            max_no_theif = 0 + max(max_left_no_theif, max_left_theif) + max(max_right_no_theif, max_right_theif)
            
            return max_theif, max_no_theif
                
        return max(_rec(root))
```

# 437. Path Sum III

我在这一题用了两个递归函数来进行两重遍历，时间复杂度为`O(N^2)`。

但是实际上只需要一次遍历即可，需要牺牲一点空间复杂度来存储路径结果，但是时间复杂度只需要`O(N)`。

首先题目中需要寻找的目标路径有两种：从根开始到当前节点的完整路径和不从根开始的不完整路径，

我们需要获得每一条符合要求的路径的加和信息，才能计算出总的满足条件的路径的数量。

参考讨论里一个大神的 Back tracking 解决方案， 他给遍历路径信息提供了一个很好的思路：

首先构造一个递归函数来遍历树中的每一个节点。

当我们遍历到某一节点时，我们将它当前完整路径上所有节点的加和以出现次数的形式存储在一个`mapping`字典里，也就是加 1。

那么当我们从根遍历到该节点的时候，`mapping`里就存储了从根到这个路径上每一个节点的历史加和和对应的出现次数。

最关键的就是接下来这一想法：当前的完整路径加和与`mapping`里存储的历史加和的差值就是以该历史加和对应的中间节点为起点，到当前节点的不完整路径的加和。

于是，以当前路径为起始点，向着根方向的所有包含当前节点的路径对应的加和信息都存储在`mapping`里，包含完整路径，因为 0 也在字典中。

而函数通过遍历每一个点，从而可以不重复地判断所有的路径加和信息。

在递归调用完左子树和右子树之后，回溯到上一个节点的状态，从字典中将当前的完整路径和的出现次数减 1 即可。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        ans = [0]
        
        def dfs(node, curr_sum):
            if not node: return

            # 判断当前节点往根方向是否存在满足条件的路径和
            # 并将该路径和对应的路径数累加到结果上
            curr_sum += node.val
            if (curr_sum - sum) in mapping:
                ans[0] += mapping[curr_sum - sum]

            # 将当前完整路径的和存储在字典内
            mapping[curr_sum] += 1

            # 递归调用
            dfs(node.left, curr_sum)
            dfs(node.right, curr_sum)

            # 回溯
            mapping[curr_sum] -= 1
            
        mapping = defaultdict(int)
        mapping[0] = 1
        dfs(root, 0)
        return ans[0]
```

# 508. Most Frequent Subtree Sum

和 easy 题 501. Find Mode in Binary Search Tree 一个思路。

利用 mapping 来存储 occurrence 信息，再判断频数最高的数据。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
        
        sub_tree_sum_mapping = defaultdict(int)
        res_sum = []
        most_occurrence = 0
        
        def _rec(root):
            
            if not root: return 0
            
            nonlocal most_occurrence
            
            left_sum = _rec(root.left)
            right_sum = _rec(root.right)
            
            cur_sum = root.val + left_sum + right_sum
            sub_tree_sum_mapping[cur_sum] += 1
            if sub_tree_sum_mapping[cur_sum] > most_occurrence:
                res_sum.clear()
                res_sum.append(cur_sum)
                most_occurrence = sub_tree_sum_mapping[cur_sum]
            elif sub_tree_sum_mapping[cur_sum] == most_occurrence:
                res_sum.append(cur_sum)
            
            return cur_sum
                
        _rec(root)
        return res_sum
```

# 513. Find Bottom Left Tree Value

# 515. Find Largest Value in Each Tree Row

和上一题一样的思路，熟练掌握节点深度信息的顺向传递即可。

# 538. Convert BST to Greater Tree

# 623. Add One Row to Tree

# 652. Find Duplicate Subtrees

对子树的和进行哈希来减少判断次数，然后使用遍历来判断是否是重复子树。

# 653. Two Sum IV - Input is a BST

BFS 广度优先搜索，使用双向队列来实现队列作为辅助结构

```python
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        deque=collections.deque([root])
        s=set()
        while deque:
            node=deque.popleft()
            if k-node.val in s: return True
            s.add(node.val)
            if node.left: deque.append(node.left)
            if node.right: deque.append(node.right)
        return False
```

DFS 同样用双向队列实现栈作为辅助结构

```python
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        deque=collections.deque([root])
        s=set()
        while deque:
            node=deque.pop()
            if k-node.val in s: return True
            s.add(node.val)
            if node.right: deque.append(node.right)
            if node.left: deque.append(node.left)
        return False
```

# 654. Maximum Binary Tree

# 655. Print Binary Tree

# 662. Maximum Width of Binary Tree

# 669. Trim a Binary Search Tree

# 687. Longest Univalue Path

# 671. Second Minimum Node In a Binary Tree

简单题，但是广度优先搜索比递归要快得多。因为 DFS 更方便提前终止程序。

# 701. Insert into a Binary Search Tree

按照 BST 的性质搜索，将目标节点作为叶节点插入在最后即可。

# 865. Smallest Subtree with all the Deepest Nodes

# 814. Binary Tree Pruning

# 863. All Nodes Distance K in Binary Tree