在比赛内用到的基础预训练模型：

比赛中提到的:

视频2D特征 : ResNet50, EfficientNet，inception,  Swin Transformer,  SlowFast, CLIP, Vision Transformer

视频3D特征：S3D,  I3D，TimesFormer, Video Swin Transformer

音频: Mel + VGGish

文本: BERT, Robert-wwm

Speech-To-Text Feature：extracted using the Google Cloud speech API, to extract word tokens from the audio stream, which are then encoded via pretrained word2vec embeddings

[huggingface托管](https://github.com/huggingface/transformers/blob/master/README_zh-hans.md)

常用中文预训练模型:

[ Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm):基于全词遮罩（Whole Word Masking）技术的中文预训练模型BERT-wwm，以及与此技术密切相关的模型：BERT-wwm-ext，RoBERTa-wwm-ext，RoBERTa-wwm-ext-large, RBT3, RBTL3。

- **`BERT-large模型`**：24-layer, 1024-hidden, 16-heads, 330M parameters
- **`BERT-base模型`**：12-layer, 768-hidden, 12-heads, 110M parameters

[中文MacBERT预训练模型](https://github.com/ymcui/MacBERT):**MacBERT** is an improved BERT with novel **M**LM **a**s **c**orrection pre-training task, which mitigates the discrepancy of pre-training and fine-tuning.

- **`MacBERT-large, Chinese`**: 24-layer, 1024-hidden, 16-heads, 324M parameters
- **`MacBERT-base, Chinese`**：12-layer, 768-hidden, 12-heads, 102M parameters

[中文ELECTRA预训练模型](https://github.com/ymcui/Chinese-ELECTRA):谷歌与斯坦福大学共同研发的最新预训练模型ELECTRA因其小巧的模型体积以及良好的模型性能受到了广泛关注。 为了进一步促进中文预训练模型技术的研究与发展，哈工大讯飞联合实验室基于官方ELECTRA训练代码以及大规模的中文数据训练出中文ELECTRA预训练模型供大家下载使用。

- **`ELECTRA-large, Chinese`**: 24-layer, 1024-hidden, 16-heads, 324M parameters
- **`ELECTRA-base, Chinese`**: 12-layer, 768-hidden, 12-heads, 102M parameters
- **`ELECTRA-small-ex, Chinese`**: 24-layer, 256-hidden, 4-heads, 25M parameters
- **`ELECTRA-small, Chinese`**: 12-layer, 256-hidden, 4-heads, 12M parameters

[中文XLNet预训练模型](https://github.com/ymcui/Chinese-XLNet):本项目提供了面向中文的XLNet预训练模型，旨在丰富中文自然语言处理资源，提供多元化的中文预训练模型选择。

- **`XLNet-mid`**：24-layer, 768-hidden, 12-heads, 209M parameters
- **`XLNet-base`**：12-layer, 768-hidden, 12-heads, 117M parameters

MMLAB开源实现：

[MMClassification](https://github.com/open-mmlab/mmclassification/blob/master/README_zh-CN.md)(2D特征)：

-  ResNet: 残差网络

-  ResNeXt:ResNeXt是[ResNet](https://zhuanlan.zhihu.com/p/42706477)[2]和[Inception](https://zhuanlan.zhihu.com/p/42704781)[3]的结合体，不同于[Inception v4](https://zhuanlan.zhihu.com/p/42706477)[4]的是，ResNext不需要人工设计复杂的Inception结构细节，而是每一个分支都采用相同的拓扑结构。ResNeXt的本质是[分组卷积（Group Convolution）](https://zhuanlan.zhihu.com/p/50045821)[5]，通过变量**基数（Cardinality）**来控制组的数量。组卷机是普通卷积和深度可分离卷积的一个折中方案，即每个分支产生的Feature Map的通道数为 n。

SE：Squeeze-and-Excitation的缩写，特征压缩与激发的意思。
  可以把SENet看成是channel-wise的attention，可以嵌入到含有skip-connections的模块中，ResNet,VGG,Inception等等。

- SE-ResNet: 将SE模块嵌入到ResNet中

- SE-ResNeXt: 将SE模块嵌入到ResNeXt中

- RegNet: 结合手动设计与NAS提出了一种新的网络设计范式，由NAS的设计一个单独的网络到设计一个更好的网络设计空间，来获得一族更好的网络模型，并可以从中找到网络设计的通用设计准则，这个过程就叫做网络设计空间的设计。何凯明组2020年CVPR作品

- ShuffleNet:旷世科技提出的方法，结合Group Convolution和Channel Shuffle对Resnet进行改进，可以认为是Resnet的压缩版本。Group Convolution是将输入层的不同特征图进行分组，然后采用不同的卷积核再对各个组进行卷积，这样会降低卷积的计算量。Group Convolution的主要问题是不同组之间的特征图不相互通信，降低了网络的特征提取能力。因此想MobileNet等网络在进行组卷积之后还要使用密集的1x1卷积，保证不同通道之间的信息交换。而ShuffleNet采用了Channel Shuffle方法对经过Group Convolution后的特征图进行“重组”，保证信息在不同组之间流转。如下图中(c)所示。Channel Shuffle并不是随机的，而是“均匀的打乱”。

- MobileNet：核心贡献：将标准的卷积拆分成了Depthwise + Pointwise

  对所有卷积层kernel数量统一乘以缩小因子a来进一步压缩网络

- Swin-Transformer: iccv2021 best paper,解决了计算复杂度问题的VIT

[MMAction2](https://github.com/open-mmlab/mmaction2/blob/master/README_zh-CN.md)(3D特征)：

* [C3D](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/c3d/README_zh-CN.md) (CVPR'2014)
* [TSN](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsn/README_zh-CN.md) (ECCV'2016)
* [ I3D](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/i3d/README_zh-CN.md) (CVPR'2017)
* [ I3D Non-Local](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/i3d/README_zh-CN.md) (CVPR'2018)
* [R(2+1)D](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/r2plus1d/README_zh-CN.md) (CVPR'2018)
* [TRN](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/trn/README_zh-CN.md) (ECCV'2018)
* [TSM](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README_zh-CN.md) (ICCV'2019)
* [TSM Non-Local](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README_zh-CN.md) (ICCV'2019)
* [ SlowOnly](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowonly/README_zh-CN.md) (ICCV'2019)
* [SlowFast](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowfast/README_zh-CN.md) (ICCV'2019)
* [CSN](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/csn/README_zh-CN.md) (ICCV'2019)
* [ TIN](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tin/README_zh-CN.md) (AAAI'2020)
* [TPN](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tpn/README_zh-CN.md) (CVPR'2020)
* [ X3D](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/x3d/README_zh-CN.md) (CVPR'2020)
* [ OmniSource](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/omnisource/README_zh-CN.md) (ECCV'2020)
* [MultiModality: Audio](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition_audio/resnet/README_zh-CN.md) (ArXiv'2020)
* [TANet](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tanet/README_zh-CN.md) (ArXiv'2020)
* [ TimeSformer](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/timesformer/README_zh-CN.md) (ICML'2021)
