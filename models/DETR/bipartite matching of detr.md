---
title: bipartite matching of DETR
author: 逯润雨
top: false
cover: false
toc: true
mathjax: false
date: 2022-02-13 14:00:24
img:
coverImg:
keywords:
password:
summary:
tags:
 - DETR
 - bipartite matching
categories:
 - 深度学习
---

# Preface

其实以上网络结构部分过年前都已经整完了，过年后没有在这方面有什么关注，这里开始看matcher.py等内容学习，当然这里打算正儿八经记笔记了，之前的都没记笔记进行有关的学习，这里正式开始学习记录！这里首先就是关于双向匹配问题，以下将结合源码和文章详细来分析。

# Reference
* 官方仓库：https://github.com/facebookresearch/detr
* detr文章：
  * 用Transformer做object detection：DETR    https://zhuanlan.zhihu.com/p/267156624
  * 【论文笔记】从Transformer到DETR： https://zhuanlan.zhihu.com/p/146065711
  * 写得非常专心的文章：https://liumengyang.xyz/detr-detection-transformer/

* 一些双向匹配有关文章：
  * 关于DETR打比赛：https://www.kaggle.com/tanulsingh077/end-to-end-object-detection-with-transformers-detr
  * 论文+源码，简洁版：https://blog.csdn.net/weixin_36047799/article/details/106825645
  * 源码解析，较为易理解：https://zhuanlan.zhihu.com/p/361253913
  * 二分图最大匹配、完美匹配、匈牙利算法：https://www.renfei.org/blog/bipartite-matching.html

# Intro

首先我们还是先来回顾以下之前的工作，之前是官方的代码是用来定义了detr的网络结构之后网络输出是一个形状为[B, query_num, classes + 1]的预测类别和[B, query_num, 4]的预测框，对于COCO数据集来说query_num是100，也就是一个图片会预测出来100个框。显然这一百个框只有很少一部分才是真正的Ground Truth.我们接下来就要将这些预测框和相应的GT进行相联系。
一个最基本的方法是，这里设计类别的时候除了现有的类别num_classes， 还多设计了一个类别叫做背景类，假设GT有m个，而这里有100个框，那么其中100-m个框就和背景类相匹配，这样一来就有一个等容量的二分图可供我们去匹配了。
当然虽然说是这么说，目前我们还没有更加具体的措施来实现这一件事，我们需要定义一个算法完成二分图的匹配，这个算法叫做匈牙利算法，同时我们还需要定义一个新的cost来让我们能够用匈牙利算法是这个cost最小地完成我们的二分图匹配.
*注意这里的cost不是我们的损失函数，只是达到二分匹配最优情况下定义的cost*

# Some basic Exceptions

## 二分图
二分图实际上就是将图的顶点分为两部分，其中每个部分的顶点之间没有边连接，也就是该图的所有边都在两个部分之间而不在各自的部分之内。

比方说对于无向图1来说：
<a href="http://imagelol.com/image/L4JM6E"><img src="https://s6.jpg.cm/2022/02/13/L4JM6E.png" alt="L4JM6E.png" border="0" width="400px" class="centerImage"></a>
这个就不属于二分图，但是对于下面这个图：

<center><img src="https://img.renfei.org/2013/08/1.png"></center>
这个图就可以变为二分图:
<center><img src = "https://img.renfei.org/2013/08/2.png"></center>

<!-- 
<img src="http://www.cxyxiaowu.com/wp-content/uploads/2019/10/1571057653-9eea9efeb86f600.png", class="center">
 -->



## 匹配

对于匹配来说，匹配指的是一个边的集合，其中所有边都没有公共的顶点。比方说下图中红色边就是我们二分图的一个匹配：
<center><img src = "https://img.renfei.org/2013/08/3.png"></center>


## 最大匹配
最大匹配指的是一个图的所有匹配中含有边数最多的匹配方式，比方说还是刚才那个图，他的最大匹配方式就是：
<center><img src = "https://img.renfei.org/2013/08/4.png"></center>



## 完美匹配
完美匹配，顾名思义就是该图的匹配中所有顶点都在里面，这就是完美匹配，可见完美匹配一定是最大匹配但是完美匹配却不一定存在。

## 交替路径
交替路径指的是从一个未匹配点开始往后走，依次经过非匹配边、匹配边、非匹配边、匹配边.....这样形成的路径叫做交替路径。

## 增广路径
从一个未匹配点出发，走交替路径，如果过程中遇到了未匹配点那么该路径就是增广路径。

*其实交替路径走到相应的未匹配点后就结束了*

增广路径性质：
（1）P的路径长度必定为奇数，第一条边和最后一条边都不属于M，因为两个端点分属两个集合，且未匹配。
（2）P经过取反操作可以得到一个更大的匹配M’。
**（3）M为G的最大匹配当且仅当不存在相对于M的增广路径。**

## 匈牙利算法

介绍完以上后我们目前的一个问题就是如何寻找一个二分图的最大匹配问题, 这个问题的解决算法就叫做匈牙利算法。

### 基本思想
基本思想：通过寻找增广路径，把增广路径中的匹配边和非匹配边的相互交换，这样就会多出一条匹配边，直到找不到增广路径为止。


当然没有图来说明肯定是很不好理解的，这里不在详述，可以参考下面的博客来看匈牙利算法：
* https://www.cxyxiaowu.com/874.html



[TOC]



## DETR中的bipartite match

看了看好多有关的论文讲解，感觉自己终于懂了一部分，唉，真的是好难好难啊目标检测，还是非常高兴的，参考了好多博文最后结合代码看感觉确实不一样！

首先还是把参考的文章放出来，其实还是很早的文章，上面已经放过一次了，这几篇文章重点主要看的就是bipartite match部分：
* 代码解释很清楚：https://zhuanlan.zhihu.com/p/361253913
* 对论文的cost和loss解释非常清楚：https://zhuanlan.zhihu.com/p/267156624
* 这方面解释不太详细：https://zhuanlan.zhihu.com/p/146065711
* 对cost和loss解释也比较清晰：https://zhuanlan.zhihu.com/p/337649487

首先说明， DETR这里loss和cost是两个概念，刚开始看很复杂但是其实后来看DETR的loss相比于YOLO和FasterRcnn来看还是简单不少的，这里的cost是为了让我们整体的二分匹配达到最优，我们一个一个匹配完毕得到最优匹配也就是定义的cost为最小后才进行下一步的计算loss进行反向传播训练损失函数。

我们可以参见上面分享的链接来学习有关bipartite matching部分，因为本人也是拾人牙慧学得。

接下来我们可以具体看看match.py:

## `matcher.py`

我们首先有必要了解一下刚开始导入的一个库函数：
```python
from scipy.optimize import linear_sum_assignment
```
该函数是用来做匈牙利算法的，也就是这里论文作者根本没有手撸匈牙利算法，而是调用了一个python库scipy，而`scipy.optimize`这个库都是用来做线性规划的，所以这里使用了这个库。

简单来说，该库导入的是一个损失矩阵，假设我们现在有二分图的两组：(a1, a2, a3)和(b1, b2, b3)这么两组，然后我们想要寻找这两组数据的最优匹配的话，我们首先需要设计一个cost，然后我们计算出这两组元素依次之间的cost都是多少，比如cost(a1,b1)=4，cost(a1,b2)=1, cost(a1,b3)=3,cost(a2,b1)=2......

就这样我们得到一个3×3的损失矩阵，然后我们传入这个函数，函数就会输出对应的行索引和列索引。
以上例子选自下面这个链接，当然我也做了一些加工修饰，可参考这个链接：
* https://blog.csdn.net/your_answer/article/details/79160045






