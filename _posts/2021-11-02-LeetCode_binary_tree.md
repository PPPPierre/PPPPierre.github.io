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

给定一个整数 n，生成一个包含所有由值为 1-n 的节点组成的 BST 的 list。

已知 BST 的性质：每一个节点的左支节点的值都小于该节点，其右支节点的值都大于等于该节点。

那么根据该性质，则可以构建一个递归函数`helper`，该函数以两个指针`i`，`j`作为输入，输出由`range(i,j+1)`这个范围的数构成的所有可能 BST 的 list。

在该递归函数内部，首先遍历`range(i,j+1)`，选择一个数`k`作为`root`，然后遍历`helper(i, k-1)`输出的元素作为左支，遍历`helper(k+1, j)`输出的元素作为右支，然后就可以构造新的节点`TreeNode(val=k, left=left_node, right=right_node)`。最后将这些新构造的节点逐个添加到 list 即可。

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