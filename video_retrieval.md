Top Authors

Linchao Zhu, Yi Yang : University of Technology Sydney  

Keizo Oyama: [National Institute of Informatics](https://scholar.google.com/citations?view_op=view_org&hl=zh-CN&org=4044695345304176160) 

Ivan Laptev: INRIA

Cordelia Schmid: Google Research

Andrew Zisserman：VGG

相关比赛

https://arxiv.org/pdf/2008.00744.pdf

[On Semantic Similarity in Video Retrieval](https://www.semanticscholar.org/paper/c5ec16be37131398704bab98b71a154028733731)(指出实例级别的检索是不可取的)

[VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://www.semanticscholar.org/paper/f7ab6c52be9351ac3f6cf8fe6ad5efba1c1595e8)(将hard negatives引入到多模态嵌入常用的损失函数中)

[Fine-Grained Video-Text Retrieval With Hierarchical Graph Reasoning](https://www.semanticscholar.org/paper/0b78e14dfc2050878e8c817e4782c0c81ee7f5dd)(提出了一种层次图推理（HGR）模型，将视频文本匹配分解为全局到局部的层次。具体来说，该模型将文本分解为层次语义图，包括事件、动作、实体和跨层次关系的三个层次。利用基于注意的图形推理生成层次化的文本嵌入，可以指导不同层次视频表示的学习。HGR模型聚合来自不同视频文本级别的匹配，以捕获全局和本地详细信息。在文本方面：全局事件由整个句子表示，动作用动词表示，实体指名词短语。不同的层次不是独立的，它们之间的相互作用解释了它们在事件中扮演的语义角色。)

[Support-set bottlenecks for video-text representation learning](https://www.semanticscholar.org/paper/78bc767ebd02c0cc690fdb334c37bf64cfaf0115)(将数据集中的每一个样本表示为其他support样本的加权组合来缓解表征中的特例化)

模块堆叠

[A Novel Convolutional Architecture For Video-Text Retrieval](https://www.semanticscholar.org/paper/cf1362088de663d0848fdff21af09b0c0920581e)(通过对不同尺寸的卷积进行堆叠来实现对local和long-term特征的提取)

[Stacked Convolutional Deep Encoding Network For Video-Text Retrieval](https://www.semanticscholar.org/paper/f752bbe2fc42b65671cee9a7032326acf11c90f2)(希望通过堆叠的MSDC模块来学习视频文本中的long-range relations)

[Dual Encoding for Video Retrieval by Text.](https://www.semanticscholar.org/paper/3a62eef641f0cc6c248da817819582043f82a6ed)(用多层编码器从粗到细的学习模态信息，同时利用latent space的high preformace 和 concept sapace 的良好解释性)

对比学习:

[Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos](https://www.semanticscholar.org/paper/d9b1bb8053f32c6da9bbbec564d750d55b486f00)(扩展实例级别对比学习的概念，在训练过程中加入聚类的步骤，从而获取zero-shot的能力)

[TACo: Token-aware Cascade Contrastive Learning for Video-Text Alignment](https://www.semanticscholar.org/paper/e79be3f9ce409f1a9b7084ef880298665e5212d0)(提出了两种方法用于提高对比学习的性能，第一种是考虑syntactic classes of words的token-aware contrastive loss，第二种则是通过一个级联的采样方法来生成用于多模态融合层的小的负样本层)

层次化建模：

[Cross-Modal and Hierarchical Modeling of Video and Text](https://www.semanticscholar.org/paper/ea133d0067740902bc26a082c842d9e7ba48ecf6)(对视频和文本进行层次化建模，对不同层次的信息分别进行对准)

[COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning](https://www.semanticscholar.org/paper/80089ad641bae28b0e57771afef181b60011069e)(将Transformer模块引入到层次化建模中，引入attention-aware feature aggregation layer，contextual transformer两个模块，另外将cycle-consistency loss引入到对准任务中)

[HANet: Hierarchical Alignment Networks for Video-Text Retrieval](https://www.semanticscholar.org/paper/cfd2884eeb5031e3dfccd0292cdf3cabf7b901eb)(提出Hierarchical Alignment Net 来在 namely event(video and text)，action(motion and vero) and entity(appearance and noun) 三个语义层次上对准，相对于上两个引入了GCN和SeMe模块)

[HiT: Hierarchical Transformer with Momentum Contrast for Video-Text Retrieval](https://www.semanticscholar.org/paper/cdedfeeeff844392735800bc38408e7135fdf4f9)(将对比学习方法引入到层次化的Transformer中解决batch过小的问题)

[Dig into Multi-modal Cues for Video Retrieval with Hierarchical Alignment](https://www.semanticscholar.org/paper/41e9e4ddebfb5fb3562a81497bdc93735fe6d8ec)(使用multi-step attention来进行local feature的对准，holistic transformer来进行global feature的对准)

多特征融合方法：

[Learning Joint Embedding with Multimodal Cues for Cross-Modal Video-Text Retrieval](https://www.semanticscholar.org/paper/9dbca9da6a72ba3739813288b677888a6cf76272)(将音频、文字，视频、文字映射至两个不同的特征空间，通过ranking拉近空间中的距离)

[A Joint Sequence Fusion Model for Video Question Answering and Retrieval](https://www.semanticscholar.org/paper/8befcd91c24038e5c26df0238d26e2311b21719a)(通过Joint Semantic Tensor将表征对编码成3D卷积，随后用Convolutional Hierarchical Decoder解码获取其相关信息)



多expert方法:

[Use What You Have: Video retrieval using representations from collaborative experts](https://www.semanticscholar.org/paper/b16eeb1e975e8e6ea9450c78fd12da05cfd1375f)(将来自视频的多模态、极高维度的信息浓缩为单个、紧凑的视频表示，用于使用文本查询的视频检索任务，其中特异性程度是开放式的。具体地，以预训练语义嵌入的形式利用现有知识，其中包括“一般”特征，例如来自视觉内容的运动、外观和场景特征。此外，作者还探索了使用来自 ASR 和 OCR 的更“特定”线索，这些线索间歇性地可用于视频，并发现这些信号在有效用于检索方面仍然具有挑战性。 本文提出了一个协作专家模型来聚合来自这些不同的预训练专家的信息)

[TEACHTEXT: CrossModal Generalized Distillation for Text-Video Retrieval](https://www.semanticscholar.org/paper/57a4d85b92e692087f6a308148dc6f8b4debe333)(使用蒸馏方法对语言领域的多个预训练模型的信息进行提取，并将其拓展到视觉领域以证明他们提出蒸馏的方法在提取互补信息时的有效性)

[Multi-modal Transformer for Video Retrieval](https://www.semanticscholar.org/paper/6871f6c5437a747fae75a19962f418d234ce2dc1)(希望通过transformer学到多个expert中互补的信息，除此之外transformer还可以对expert中的时序信息进行处理，而不是单纯的aggregate)



Transformer based

[CLIP2Video: Mastering Video-Text Retrieval via Image CLIP](https://www.semanticscholar.org/paper/c401e01c9ee32fab7d02670d1c754f44fc1ff99e)(在CLIP的基础上引入一个TDB模块用于捕捉精细时间轴上的运动，一个TAB模块用于重新对齐视频片段和短语的token并增强多模态相关性)腾讯PCG

[CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval](https://www.semanticscholar.org/paper/281ad83e06d731d5d686acf07cd701576f1188c4)(提出了一个叫做CLIP4Clip的模型，以端到端的方式将 CLIP 模型的知识转移到视频语言检索中。)

[Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](https://www.semanticscholar.org/paper/82291bcd36ebb0a4c143b08a8bddd89c4a744586)(可以灵活的在图像-文本，视频-文本数据集上进行训练，可以是单独的方式或者是组合的方式。该论文中提出的模型是对最近的ViT和Timesformer架构的改编和扩展，由空间和时间上的 attention 组成。模型使用课程学习的训练方式（由易到难），开始时将图像视为视频的 “冻结 “快照，然后在视频数据集上训练时逐渐学会关注越来越多的时域上下文。)

[Improving Video-Text Retrieval by Multi-Stream Corpus Alignment and Dual Softmax Loss](https://www.semanticscholar.org/paper/95174adf52a7d24c05c39e1d7a68bd5506a37855)(提出了一种具有单门专家混合（CAMoE）和一个新的Dual Softmax Loss（DSL）的多流语料库对齐网络来解决文本和视频的异质性。CAMoE 使用 Mixture-of-Experts (MoE) 提取多视角视频表示，包括动作、实体、场景等，然后将它们与文本的相应部分对齐。DSL 可以避免在以前的对比方法中出现的单向最优匹配。通过引入batch中每一对的内在先验，DSL作为修正器来校正相似矩阵并实现对偶最优匹配。) **刷榜作品**

数据层面

[MDMMT: Multidomain Multimodal Transformer for Video Retrieval](https://www.semanticscholar.org/paper/3d611852a0b25dd6d8d863d7c5d5c710630543f2)(解决两个文本很相似，但是目标却需要让它们分开的问题。主要贡献：

- 将几个数据集合并，训练出了一个模型超过了当时所有单独数据集上SOTA的模型。
- 提出一种清洗数据集的方法，找到数据集训练和测试集的重合部分，将其从训练集中去除，防止过拟合。)

[Learning a Text-Video Embedding from Incomplete and Heterogeneous Data](https://www.semanticscholar.org/paper/3448af861bf5d44ce7ab6b25002504815212252e)(提出Mixture-of-Embedding-Experts (MEE) model，可以处理缺失一部分信息的“视频”，将之正常的与文本进行匹配，增加训练集大小。)

数据集

MSVD : comprises a total of 80K descriptions (in English) for 1,970 videos sourced from YouTube (with approximately 40 sentences per video). Unlike the other datasets featured in the pentathlon, the videos contained in MSVD do not possess audio streams.

DiDeMo : consists of unedited, personal videos that are collected in an open-world setting and which include diverse content such as pets, music concerts and sports games. The dataset comprises 10,464 videos which are accompanied by approximately 3-5 pairs of descriptions and distinct moments per video.

ActivityNet(+captions) : contains a total of 15K videos (sourced from the original ActivityNet dataset) accompanied by approximately 100K descriptive sentences. The videos, originally sourced from YouTube, exhibit a broad diversity of actions and content.

MSR-VTT : contains 10K videos sourced from YouTube which are accompanied by 200K descriptive captions (thus, there are 200K unique video-caption pairs in total).

YouCook2 : includes 2000 long untrimmed videos from 89 cooking recipes; on average, each distinct recipe has 22 videos. The videos are sourced from YouTube and contains content filmed from a third-person viewpoint with unfixed cameras.

