# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

> 读完后感觉。。这就是将卷积搬到了Transformer之中！采用了非常大量的CNN的Tricks和技巧，几乎就像是一个没有卷积层和池化层的的Trans神经网络。

## Preface

开始进军Transformer! 本来打算看看视觉上还有没有什么SOTA的方向的，没想到一下子就到Swin Transformer了，真的是始料未及，不过中间还有一个DeiT没来得及看，由于还有后门攻击和多任务学习的工作，所以这里选择直接看Swin Transformer了。

下面是一些有关的资料：
* 官方的代码库: https://github.com/microsoft/Swin-Transformer
* 知乎非常详细的解释，包含代码解读： https://zhuanlan.zhihu.com/p/367111046
* 知乎另一个非常详细的解读，但是没有代码：https://zhuanlan.zhihu.com/p/360513527
* 同样结合源码：https://zhuanlan.zhihu.com/p/361366090
* 沐神解读：https://www.bilibili.com/video/av850677660
* 
3月传到arxiv, 然后4月放代码，接着疯狂更新各个领域的应用与迁移。
从3月开始到10月，该团队每月一篇论文的速度用该骨干网络将整个视觉任务刷了一个遍！

应用范围非常广，并且效果也非常炸裂。

## Abstract

作者表示他们提出了一个可用于视觉领域的基于Transformer的通用骨干网络，作者首先提到视觉任务和NLP任务有着非常大的尺度差异例如说图像的信息过于丰富，而语言的信息相对而言就比较简单同时图像的尺度差异比较大，比方说检测任务中鸟类和火车的尺度差异非常大，但同时文本中单词之间的差异就很小（train bird）。另外就是说图片一个像素就有挺大的信息量，同时文本信息中一句话就可能不是太多太长的序列。为了弥补这方面的差异，作者团队提出了一中Hierachical的，也就是堆叠层级式的、使用滑动窗口来计算的Transformer通用骨干网络。作者生成该结构拥有非常灵活的输入，对图片大小没有限制同时计算复杂度是线性的，随图片的大小增长。另外就是介绍了该结构在各种传统视觉任务上都刷榜，然后取得了非常好的结果等等。

shift window是这篇论文的最主要贡献，而Hierarchical实际上就是多层级的提取工作，就像卷积神经网络经常做得那样。

## 1 Introduction

首先介绍了CNN在视觉上的发展是非常好的, 也介绍了CNN的发展趋势是greater scale, more extensive connections and more sophisticated(复杂巧妙的) forms of convolution. 另一方面谈到NLP中Transformer是绝对的成功所以他考虑将Transformer用到视觉中。
这里再次提到了NLP和视觉上应用的差异，首先还是scale， NLP问题中基本元素是单词，词向量token之间差距都很小，往往都是相同大小的，但是视觉方面的模型的各种物体的大小却并不相同。另一个问题是对比于文段中的单词，图片往往都有更高分辨率的图像，一张图像的信息量非常大，对于语义分割和目标检测这些领域来说都需要精确、密集的像素水平的精确程度，比方说一般视觉上的分割任务的图像大小都是800×600这类，但是对于Transformer结构来说这样的输入有些难以接受，计算复杂度太高了。
<a href="https://imagelol.com/image/LgWfnw"><img src="https://s6.jpg.cm/2022/02/24/LgWfnw.png" alt="LgWfnw.png" border="0"></a>
可以参考这张图，为了能够让其在视觉任务上取得更好的应用，这里作者提出了这样的Hierachical的概念，在底层的时候是分为了4×4的patch, 然后随着Transformer层数的加深，逐渐patch大小也在增加，这里也和Vit做了一个对比，后来patch就变成8×8，16×16，可以说是融合了多尺度的特征，这样一来就得到了多层的feature map，也就能够使用一些很高级的方法类似于U-Net, FPN这类方法到我们的任务中。
> 但是和同期的VIT对比可以看出VIT就是每次经过一个Transformer Block输出的shape都还是16×16的，虽然有一个全局的自注意力机制，但是这样一来还是没有像SWIN T一样有一个很明确的多尺度的一个概念。
我们做进一步的观察可以发现FPN这样的结构就是明确融合了多尺度的特征来做检测，这是至关重要的，抓住不同的特征，U-Net做分割也是如此，考虑多尺度特征：
<a href="https://imagelol.com/image/Lg5DvX"><img src="https://s6.jpg.cm/2022/02/24/Lg5DvX.png" alt="Lg5DvX.png" border="0"></a>

并且这里面为了节省计算量，也选择使用了每个小patch内部进行一个局部的自注意力使用，这样计算的复杂度就大大减小了，不像VIT一样使用的是全局的自注意力，因为本身视觉任务都具有locality这样的特性，所以全局自注意力确实有些奢侈了，这样一来计算复杂度就由二次复杂度变为线性复杂度了，随着图片size大小的变化而变化。
另一个很关键的方法是shift window的提出。作者这里还着重提到了这个shift window不是简单的目标检测方法sliding window.作者指出先前也有过一些工作做的是滑动窗口在transformer上的卷积，但是那些工作都不太好，他们没法做到低延迟在硬件上，计算的速度太慢了，而且对于Transfomer来说q,k的输入都是有区别的导致问题很多 但是作者这里提出的shift window就非常不错，很好解决了这样的问题：
<a href="https://imagelol.com/image/Lg5MaD"><img src="https://s6.jpg.cm/2022/02/24/Lg5MaD.png" alt="Lg5MaD.png" border="0"></a>
灰色的框是一个基本的计算单元，而红色的框是一个中型的计算单元，小patch就像是一个token，我们每次都在红色框中做一个local的自注意力行。我们的shift便是往右下移两格（其实是一个window patch size数的一半（displacing [M/2, M/2]））。这样的话照顾到了连贯的信息特征，把这4个window相邻的四个角的特征都再次算了一遍自注意力机制，非常巧妙，相当于这4个patch算了5次local window self-attention，达到了变相的全局自注意力方法。
作者后来夸了夸自己的工作效果很好同时提出希望有一个模型的大一统(unified)，提出了这些工作在视觉上完成了通杀，在考虑能否用在NLP上。

## 2 Related Work
这方面打算跳过，因为暂时还没啥好说的。可以之后有兴趣再看看，就是些有CNN，Transformer的工作。

## 3. Method
这个值得重点说下主要分为Overall Architecture 和Shift Window的有关实现。

### 3.1 Overall Architecture

看完整体流程这个图我顿时觉得这个工作就是把卷积核变为shift window Transformer的CNN。

<a href="https://imagelol.com/image/Lg7elL"><img src="https://s6.jpg.cm/2022/02/24/Lg7elL.png" alt="Lg7elL.png" border="0"></a>

首先是一张H×W×3的图片, 然后通过和VIT类似的打成一个一个的patch的操作，这里的patch是4×4×3的patch即变为一个线性的48维向量，这样一来整张图片就变为H/4×W/4×48的一个矩阵。
然后通过stage1，Linear Embedding负责将整个图像原始值映射到一个任意维的，这里假设为C维长度的向量，然后我们通过Swin Transformer Block经过后，注意这个阶段有两次重复，整个图片的shape变为了H/4×W/4×C，接下来就是通过stage2,相比stage1，他也是重复了两次相同的模块，但是这里他将Linear Embedding 换为了patch merging，可以对比理解为CNN的池化，增大Transformer的感受野用。输出的shape变为H/8×W/8×2C

随后同样的经过了stage3, stage3相比stage2就是重复了6次相关的模块而已，输出变为H/16×W/16×4C，接着是stage4，就是把重复次数变为2次，最后的输出shape就是H/32×W/32×8C.

这里面整体的结构还是不复杂的，甚至也没有residual残差层的存在，当然这里面很重要的一点是其中Transformer Block的具体结构到底是什么。这里作者提到通过了使用基于shifted window的module取代了标准多头注意力module。而其他的部分和标准的Transformer Block没有区别，从上图图b中也能看出确实没啥区别。至于shifted window module到底是什么，我们在3.2中探究。

> As illustrated in Figure 3(b), a Swin
Transformer block consists of a shifted window based MSA
module, followed by a 2-layer MLP with GELU non-
linearity in between. A LayerNorm (LN) layer is applied
before each MSA module and each MLP, and a residual
connection is applied after each module.



### 3.2 Shifted Window based Self-Attention


作者首先还是提到了标准的transformer架构中针对每一个token都计算了和其他所有token的relationship，这样的一个全局注意力机制导致了计算复杂度是二次的，导致了在很多需要高分辨率的下游任务比方说检测和分割上有着很不友好的计算时间。

#### Self-attention in non-overlapped windows
为了更为高效的建模，作者提出了在一个local window下做一个自注意力计算，而单个的window是用来划分的图像的一个一个的部分，每个window都包含M×M个patch,一个global MSA module（也就是标准的多头注意力机制）和一个基于window的注意力机制(W-MSA)的计算复杂度对比如下（对于一张h×w个patches的feature map）：
$$\Omega_{MSA} = 4hwC^2 + 2(hw)^2C$$
$$\Omega_{W-MSA} = 4hwC^2 + 2M^2hwC$$

可以看出MSA的计算复杂度是hw的二次函数，而W-MSA在固定M的情况下（默认一般为7）是线性的复杂度，明显快了非常多，尤其是在面对一些非常大的分辨率的图像的时候。


#### Shifted Window partitioning in successive blocks

作者这里也提到W-MSA这种方法缺乏Windows之间的连接，这限制了整个模型的性能，所以为了引入这些非重叠window之间的联系，作者提出了一种shifted window的方法来保持不同Windows之间的联系与通信, 虽然之前介绍过，这里再介绍一遍，稍微详细一点，我们还是先看图2：

<a href="https://imagelol.com/image/Lg5MaD"><img src="https://s6.jpg.cm/2022/02/24/Lg5MaD.png" alt="Lg5MaD.png" border="0"></a>

首先我们左边的图片是一个8×8的feature map,经过第一个module的时候（也就是Figure 3的b图左边W-MSA那个module） 仍然是一个非常正常的windows划分，是一个2×2的window，然后每个window有4×4个patch（正常情况下一个window应该有7个patch, 但是这里简化说明了。）
而后通过第二个module的时候（也就是右边SW-MSA的那个module)，开始使用shift window来移动采样计算注意力了,每个window都往右下方想displacing ([M/2], [M/2])个单位。然后再做自注意力。这样一来就相当于将window之间的关联算了进来。
所以相应的Swin-Transformer的Block的计算方式就是这样：
<a href="https://imagelol.com/image/LgdvK6"><img src="https://s6.jpg.cm/2022/02/24/LgdvK6.png" alt="LgdvK6.png" border="0"></a>
> where $\hat{z}^l$ and $z^l$ denote the output features of the (S)W-MSA module and the MLP module for block l, respectively; W-MSA and SW-MSA denote window based multi-head self-attention using regular and shifted window partitioning configurations, respectively.


最后作者说到这种方法考虑了前一层中不重合窗口之间的联系，而且在接下来的实验中被证实在图像分类、分割、检测方面均有非常不错的效果。

#### Efficient batch computation for shifted configuration

这种计算方法说实话我没看懂，图在这里：
<a href="https://imagelol.com/image/LglyO8"><img src="https://s6.jpg.cm/2022/02/25/LglyO8.png" alt="LglyO8.png" border="0"></a>

之后再说吧。

#### Relative position bias

这里面做位置编码的时候选择加一个相对位置编码到每一个head里面：
$$Attention(Q,K,V) = SoftMax(QK^T/\sqrt{d} + B)V$$
<a href="https://imagelol.com/image/LgV7w8"><img src="https://s6.jpg.cm/2022/02/25/LgV7w8.png" alt="LgV7w8.png" border="0"></a>
另外就是作者对比了不加该位置偏置和加绝对位置偏置的对比与分析，总而言之就是没有加相对位置偏置好，而且相对位置偏置好了很多相比之下，作者在Table 4中也介绍了对比消融实验的结果：
<a href="https://imagelol.com/image/LgVFIT"><img src="https://s6.jpg.cm/2022/02/25/LgVFIT.png" alt="LgVFIT.png" border="0"></a>

最后作者说了关于这个可学习的bias，如果我们要pre-trained models来做fine-tuning的话，对于不同的window size，我们可以做一个bi-cubic interpolation，双立方插值来保证bias可用在不同的windows size大小中。



### 3.3 Architecture Variants





























