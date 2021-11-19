Top Author

Yu-Gang Jiang, Zuxuan Wu : Fudan University

Lorenzo Torresani, Wang Heng ,Feiszli, Matt: FAIR

Luc Van Gool :  Head Toyota Lab TRACE

Yi Yang : University of Technology Sydney

Kiyoharu Aizawa: University of Tokyo

Ali Diba: KU Leuven

Abdenour Hadid: Ctr Machine Vis & Signal Anal CMVS, Oulu

将Transformer用到视频分类中:

* Video Swin Transformer
* ViViT: A Video Vision Transformer
* Is Space-Time Attention All You Need for Video Understanding?

对特征及进行选择加速模型推理：

* Adaptive Focus for Efficient Video Recognition
* No frame left behind: Full Video Action Recognition(根据相似度不断对2d特征进行聚类，压缩frams)

动作识别中的时间维度建模:

* Slowfast networks for video recognition  (two branch，一个分支学习结构信息，一个分支学习时序信息)
* TSM: Temporal Shift Module for Efficient Video Understanding(让卷积核的一部分沿着时间维度移动，学到时序信息)
* Dynamic image networks for action recognition (15年，使用LSTM+光流信息进行视频分类)
* Rank Pooling for Action Recognition(16年，单独训练一个网络来学习时序信息)

* Video Action Transformer Network(将self-attention用到时序信息的提取中)
* Directional Temporal Modeling for Action Recognition  (学习3D卷积重的clip级别的时序信息)
* EFFICIENT TEMPORAL-SPATIAL FEATURE GROUPING FOR VIDEO ACTION RECOGNITION(重新设计时空解耦的卷积学习特征)
* TEA: Temporal Excitation and Aggregation for Action Recognition(设计TEA block对短时间和长时间分开建模)
* Temporal Pyramid Network for Action Recognition(时间维度的特征金字塔)

Gate-Shift Networks for Video Action Recognition: C3D S3D GST CSN TSM GSM都是对3D卷积核中时间维度的处理

将多模态融合应用到视频分类中：

* Towards Good Practices for Multi-modal Fusion in Large-scale Video Classification(将多模态双线性池化引入用以融合视频和音频的信息)

* 弹幕信息协助下的视频多标签分类  (探索了将弹幕信息融合后对视频进行分类的可能性，自建了从bilibili采集的数据集(后续会开放))

* 基于深度多模态特征融合的短视频分类（通过建立相似性损失和差异性损失，探寻短视频中不同模态之间的相似性和同一模态的差异性来辅助分类）

* Multimodal Keyless Attention Fusion for Video Classification(提出了一种keyless的attetion方法用于特征融合)
* Deep Multimodal Learning: An Effective Method for Video Classification(比较了一些比较常用的循环网络用于特征融合的性能)
* A Deep Learning Based Video Classification System Using Multimodality Correlation Approach（person correlation integration）
* Multi-Stream Multi-Class Fusion of Deep Networks for Video Classification（multi-stream multi-class fusion，通过学习类关系来提高预测性能）
* Modeling Multimodal Clues in a Hybrid Deep Learning Framework for Video Classification (the feature fusion network that produces a fused representation through modeling feature relationships outperforms a large set of alternative fusion strategies)
* Residual Attention-based Fusion for video classification(将BiLSTM和attention堆叠起来用于提取时空特征)

* Multimodal video classification with stacked contractive autoencoders(用autoencoders提取模态之间的互补信息)

[mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/README_zh-CN.md)是一款基于 PyTorch 的视频理解开源工具箱，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一。

数据集:

| 年份 | 数据集名称      | paper                                                        | 类别数 | 视频数        |
| ---- | --------------- | ------------------------------------------------------------ | ------ | ------------- |
| 2004 | KTH             | Recognizing human actions: a local svm approach              | 6      | 600           |
| 2005 | Weizmann        | Actions as space-time shapes                                 | 9      | 81            |
| 2008 |                 | Action mach: a spatio-temporal  maximum average correlation     height filter for action recognition. |        |               |
| 2011 | HMDB            | HMDB:  A large video database for human motion recognition   | 51     | 6766          |
| 2012 | UCF101          | UCF101: A     dataset of 101 human actions classes from videos in the wild | 101    | 13320         |
| 2013 |                 | Towards understanding action recognition                     |        |               |
| 2014 |                 | Jhu-isi gesture and skill  assessment working set (jigsaws): A surgical activity dataset for human  motion modeling. |        |               |
| 2014 |                 | The language     of actions: Recovering the syntax and semantics of goaldirected human  activities |        |               |
| 2015 | ActivityNet     | ActivityNet: A large-scale video benchmark for human activity  understanding | 200    | 28K           |
| 2015 |                 | THUMOS challenge:Action  recognition with a large number of classes |        |               |
| 2016 |                 | Hollywood in     homes: Crowdsourcing data collection for activity understanding. |        |               |
| 2016 |                 | Human action localization with  sparse spatial supervision   |        |               |
| 2016 |                 | Spot on: Action localization  from pointly-supervised proposals |        |               |
| 2016 |                 | Recognizing fine-grained and  composite activities using hand-centric features and script data. |        |               |
| 2017 | Kinetics        | Quo vadis, action     recognition? a new model and the kinetics dataset |        |               |
| 2017 |                 | The something something  video     database for learning and evaluating visual common sense | 174    | 108.5K/220.8K |
| 2018 |                 | Every moment counts:     Dense detailed labeling of actions in complex videos. |        |               |
| 2018 |                 | What do i annotate next? an  empirical     study of active learning for action localization. |        |               |
| 2018 |                 | Ava: A video dataset of  spatio-temporally localized atomic     visual actions |        |               |
| 2018 |                 | Scaling egocentric vision: The  epic-kitchens     dataset.   |        |               |
| 2019 | Moments in Time | Moments     in time dataset: one million videos for event understanding |        |               |
| 2019 |                 | Hacs: Human action clips and  segments dataset     for recognition and temporal localization |        |               |
| 2019 | Diving48        | Resound: Towards action  recognition without representation bias. |        |               |
| 2019 | Jester          | The Jester Dataset: A  Large-Scale Video Dataset of Human Gestures |        |               |
| 2020 | FineGYM         | FineGym: A Hierarchical Video  Dataset for Fine-grained Action Understanding |        |               |
| 2020 | OmniSource      | Omni-sourced Webly-supervised  Learning for Video Recognition |        |               |
| 2020 | HVU             | Large Scale Holistic Video  Understanding                    | 739    | 572K          |

- [ UCF101](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ucf101/README_zh-CN.md) ([主页](https://www.crcv.ucf.edu/research/data-sets/ucf101/)) (CRCV-IR-12-01) (Soomro, Roshan Zamir, and Shah 2012) is a trimmed video dataset, consisting of realistic web videos with diverse forms of camera motion and illumination. It contains 13,320 video clips with an average length of 180 frames per clip. These are labeled with 101 action classes, ranging from daily life activities to unusual sports. Each video clip is assigned just a single class label. Following the original evaluation scheme, we report the average accuracy over three training/testing splits.
- [ ActivityNet](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/activitynet/README_zh-CN.md) ([主页](http://activity-net.org/)) (CVPR'2015) (Heilbron et al. 2015) is an untrimmed video dataset. We use the ActivityNet v1.3 release, which consists of more than 648 hours of untrimmed videos from a total of around 20K videos with 1.5 annotations per video, selected from 200 classes. Videos can contain more than one activity, and, typically, large time segments of a video are not related to any activity of interest. In the official split, the distribution among training, validation, and test data is about 50%, 25%, and 25% of the total videos, respectively. Because the annotations for the testing split have not yet been published, we report experimental results on the validation split.
- [Kinetics-[400/600/700\]](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/kinetics/README_zh-CN.md) ([主页](https://deepmind.com/research/open-source/kinetics/)) (CVPR'2017) (Carreira and Zisserman 2017) is a trimmed video dataset. The dataset contains 246,535 training videos, 19,907 validation videos, and 38,685 test videos, covering 400 human action classes. Each clip lasts around 10s and is labeled with a single class. The annotations for the test split have not yet been released, so we report experimental results on the validation split.
- YouTube-8M (Abu-El-Haija et al. 2016) is massively large untrimmed video dataset. It contains over 1.9 billion video frames and 8 million videos. Each video can be annotated with multiple tags. Visual and audio features have been preextracted and are provided with the dataset for each second of the video. The visual features were obtained via a Google Inception CNN pre-trained on ImageNet (Deng et al. 2009), followed by PCA-based compression into a 1024- dimensional vector. The audio features were extracted via a pre-trained VGG-inspired (Simonyan and Zisserman 2014a) network. In the official split, the distribution among training, validation, and test data is about 70%, 20%, and 10%, respectively. As the annotations of the test split have not been released to the public and the number of videos in the validation set is overly large, we maintain 60K videos from the official validation set to validate the parameters. Other videos in the validation set are included into the training set. We report experimental res
- [Moments in Time](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/mit/README_zh-CN.md) ([主页](http://moments.csail.mit.edu/)) (TPAMI'2019) consists of 800 000, 3-second YouTube clips that capture the gist of a dynamic scene involving animals, objects, people, or natural phenomena.
- Something-Something v2 (SSv2) [26] contains 220 000 videos, with durations ranging from 2 to 6 seconds. In contrast to the other datasets, the objects and backgrounds in the videos are consistent across different action classes, and this dataset thus places more emphasis on a model’s ability to recognise fine-grained motion cues.
- Epic Kitchens-100 consists of egocentric videos capturing daily kitchen activities spanning 100 hours and 90 000 clips . We report results following the standard “action recognition” protocol. Here, each video is labelled with a “verb” and a “noun” and we therefore predict both categories using a single network with two “heads”. The topscoring verb and action pair predicted by the network form an “action”, and action accuracy is the primary metric.

相关比赛

[2021年腾讯广告算法大赛](https://algo.qq.com/)

01.视频广告秒级语义解析

02.多模态视频广告标签

两个任务均以视频，音频、文本三个模态作为输入，参赛选手利用模型对广告进行理解。

在任务一中，对于给定测试视频样本，通过算法将视频在时序上进行“幕”的分段，并且预测出每一段在呈现形式、场景、风格等三个维度上的标签，使用Mean Average Precision(MAP)进行评分。

在任务二中，对于给定的测试视频样本，通过算法预测出视频在呈现形式、场景、风格等三个维度上的标签，使用Global Average Precision(GAP)进行评分。

任务一找不到开源实现

任务二开源实现

[第十名](https://github.com/beibuwandeluori/taac2021-tagging-azx)

引入nextvald聚合数据，使用多种VIT预训练模型提取模型特征

[第六名](https://github.com/chenjiashuo123/TAAC-2021-Task2-Rank6)

使用Bi-Modal Transformer对不同模态进行融合，将融合后特征用nextvald再一次聚合，得到最终特征进行预测

[第二十七名](https://github.com/XIUXIUXIUBIUA/TAAC-2021)

[baseline](https://github.com/LJoson/TAAC_2021_baseline)

达尔文团队方案(无代码实现，但方案简单)：

将提取的多模态特征分别输入到NeXtVLAD模块中，不同模态之间的不同之处在于dropout的取值不同，通过控制 dropout 的取值来调节不同模态的贡献程度，其中视频和音频特征的 dropout 为 0.95，文本特征的 dropout 为 0.85。  

