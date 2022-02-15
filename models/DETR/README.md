# facebook源码阅读

还是按照和原来的源码一样的阅读顺序，先读`detr.py`

---
title: 3End-to-End-Object-Detection-with-Transformers
author: 逯润雨
top: false
cover: false
toc: true
mathjax: false
date: 2022-01-31 17:36:00
img:
coverImg:
keywords:
password:
summary:
tags:
 - detr
 - transformer
categories:
 - 深度学习
---

# Preface

目前打算阅读Facebook的detr源码，初步打算如下：

* 边看代码边搭模型，先将Transformer、MLP、以及一些主要的架构搭建好，目前backbone可以随时backbone文件中的resnet50.
  * 写代码最后核对的时候可以看第二篇文章中的极为详细的网络结构。
* 搭建完模型还要有criterion部分，这部分可能需要matcher.py搭建一个matcher, 也就是所谓的双向匹配，可以参考：
  * 用Transformer做object detection：DETR    https://zhuanlan.zhihu.com/p/267156624
  * 【论文笔记】从Transformer到DETR： https://zhuanlan.zhihu.com/p/146065711

  * 写得非常专心的文章：https://liumengyang.xyz/detr-detection-transformer/

* 这些都整完了之后可以看一下COCO数据集然后对这个学习一下完善一下仓库的data部分让它支持导入COCO。
* 全部搞完给我训练！！！将训练结果整理出来放到仓库里面。训练的时候用3张卡，同时训练可以使用pl来写相关的代码。



# Reference

* 官方仓库：https://github.com/facebookresearch/detr
* detr文章：
  * 用Transformer做object detection：DETR    https://zhuanlan.zhihu.com/p/267156624
  * 【论文笔记】从Transformer到DETR： https://zhuanlan.zhihu.com/p/146065711
  * 写得非常专心的文章：https://liumengyang.xyz/detr-detection-transformer/

* nn.Embeding:
  * 解析：https://zhuanlan.zhihu.com/p/345450458
  * 与nn.Linear区别：
    * https://discuss.pytorch.org/t/how-does-nn-embedding-work/88518
    * https://stackoverflow.com/questions/65445174/what-is-the-difference-between-an-embedding-layer-with-a-bias-immediately-afterw

* 源码理解起来还是有一些费劲，上网上找了些资料又：
  * 关于DETR打比赛：https://www.kaggle.com/tanulsingh077/end-to-end-object-detection-with-transformers-detr
  * 论文+源码，简洁版：https://blog.csdn.net/weixin_36047799/article/details/106825645
  * 源码解析，较为易理解：https://zhuanlan.zhihu.com/p/361253913
  * 二分图最大匹配、完美匹配、匈牙利算法：https://www.renfei.org/blog/bipartite-matching.html
  * IntermediateLayerGetter：https://zhuanlan.zhihu.com/p/341176618


# DETR in detr.py

## DETR
我们首先还是看总体的大类：

```python
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
```

这里的DETR传入的参数是具体的模型，比方说backbone和Transformer，都是使用的具体的模型传进去，而不是传进去模型的参数然后定义模型，我个人还是偏向于传入参数然后构建模型。这里的注释其实说的很清楚，比方说介绍`num_queries`的时候就介绍说：
>  num_queries:  number of object queries, ie detection slot. This is the maximal number of object, DETR can detect in a single image. For COCO, we recommend 100 queries.

说明num_queries实际上就是我们传入decoder的时候的object queries的个数，实际上这个就相当于一张图片最多能检测到的物体数，对于COCO数据集作者建议使用100个queries。

预测类别的时候直接使用的就是一个简单的Linear层，预测bbox时使用的是一个MLP，3层最后输出维度为4.

## nn.Embedding
对于query_embed，也就是输入Transformer的decoder的元素，我们可以注意到这里使用的是`nn.Embedding`,当然真正输入的元素我们往下看forward函数就知道其实是query_embed.weight. 论坛上有很多QA问是不是nn.Embedding就是个简单的没有bias的nn.Linear，我其实并没有对nn.Embedding有什么了解，可以参考一下以下文章：
* 解析：https://blog.csdn.net/qq_39540454/article/details/115215056
* 解析：https://www.jianshu.com/p/63e7acc5e890
* 与nn.Linear区别：
  * https://discuss.pytorch.org/t/how-does-nn-embedding-work/88518
  * https://stackoverflow.com/questions/65445174/what-is-the-difference-between-an-embedding-layer-with-a-bias-immediately-afterw

也就是，nn.Embedding就是个lookup table，相当于一个查找表，里面存的是固定大小的dictionary，我们输入的是indices， 然后我们可以获取相应的指定的indice的word embedding的结果。

上面说得可能不是非常清楚，我们可以结合一定的例子来看，看例子之前可以先看一看` [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding) `的参数：

> CLASS torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None)

官网对它的简单介绍如下, 就是我们前面说的，就是一个存word embedding的查找表，我们可以使用相应的索引也就是indices来获得相应的词嵌入向量：
> A simple lookup table that stores embeddings of a fixed dictionary and size.
>
> This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.


我们这里只看比较重要的参数，也就是前两个，第一个是`num_embeddings`, 这个是我们词典，查找表的尺寸，比方说“词典”一共有3000个单词，我们这里的num_embeddings=3000，而索引的值就是0-2999. 而第二个参数`embedding_dim`, 这个指我们表达单个单词的维数，比方说我们embedding_dim=10， 那么一个单词就是长度为10的向量。

其他的参数可以大概看看：
> padding_idx (python:int, optional) – 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
> max_norm (python:float, optional) – 最大范数，如果嵌入向量的范数超过了这个界限，就要进行再归一化。
norm_type (python:float, optional) – 指定利用什么范数计算，并用于对比max_norm，默认为2范数。
> scale_grad_by_freq (boolean, optional) – 根据单词在mini-batch中出现的频率，对梯度进行放缩。默认为False.
> sparse (bool, optional) – 若为True,则与权重矩阵相关的梯度转变为稀疏张量。
> 作者：top_小酱油
> 链接：https://www.jianshu.com/p/63e7acc5e890
或者去官网直接看英文原版解释。

所以nn.Embedding只有一个变量叫做nn.Embedding().weight:
~Embedding.weight (Tensor) – the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from $\mathcal{N}(0, 1)N(0,1)$

这个weight是可学习的，也就是我们的“词典”。

另外注意一点，前面提到过我们输入应该是indices，这就确定了我们的输入不能是分数，只能是int型的量，所以输入应该是LongTensor，虽然这里我们并不输入，但是我们注意下总是不错的。

以下是官网给的例子，可以简单看一看：
```python
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(10, 3)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
>>> embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])


>>> # example with padding_idx
>>> embedding = nn.Embedding(10, 3, padding_idx=0)
>>> input = torch.LongTensor([[0,2,0,5]])
>>> embedding(input)
tensor([[[ 0.0000,  0.0000,  0.0000],
         [ 0.1535, -2.0309,  0.9315],
         [ 0.0000,  0.0000,  0.0000],
         [-0.1655,  0.9897,  0.0635]]])

>>> # example of changing `pad` vector
>>> padding_idx = 0
>>> embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
>>> embedding.weight
Parameter containing:
tensor([[ 0.0000,  0.0000,  0.0000],
        [-0.7895, -0.7089, -0.0364],
        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
>>> with torch.no_grad():
...     embedding.weight[padding_idx] = torch.ones(3)
>>> embedding.weight
Parameter containing:
tensor([[ 1.0000,  1.0000,  1.0000],
        [-0.7895, -0.7089, -0.0364],
        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
```



源码理解起来还是有一些费劲，上网上找了些资料又：
* 关于DETR打比赛：https://www.kaggle.com/tanulsingh077/end-to-end-object-detection-with-transformers-detr
* 论文+源码，简洁版：https://blog.csdn.net/weixin_36047799/article/details/106825645
* 源码解析，较为易理解：https://zhuanlan.zhihu.com/p/361253913
* 二分图最大匹配、完美匹配、匈牙利算法：https://www.renfei.org/blog/bipartite-matching.html
* IntermediateLayerGetter：https://zhuanlan.zhihu.com/p/341176618



