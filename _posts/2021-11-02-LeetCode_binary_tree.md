---
title: LeetCode 刷题记录 - Binary Tree
author: Stone SHI
date: 2021-11-02 5:47:00 +0200
categories: [Blogging, LeetCode]
tags: [LeetCode, Binary Tree]
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