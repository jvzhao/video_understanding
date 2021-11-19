Top Authors

Ioannis (Yiannis) Kompatsiaris, Giorgos Kordopatis-Zilos, Symeon Papadopoulos : [Information Technologies Institute](https://scholar.google.com/citations?view_op=view_org&hl=zh-CN&org=15565238550923390865), CERTH

zhili zhou :Nanjing University of Information Science and Technology

Xiushan Nie, Chaoran Cui : Shandong Jianzhu University/Shandong University of Finance and Economics



数据集：

SVD

CCWEB

UQ_VIDEO

VCDB

MUSCLE_VCD

TRECVID



相关比赛

[QQ浏览器2021AI算法大赛](https://algo.browser.qq.com/)

[赛道一：多模态视频相似度](https://docs.qq.com/doc/p/d57b07f2177d0359c2d15fb0537fa03faf1df032?dver=2.1.27147307)

信息流场景下，短视频消费引来爆发式增长，视频的语义理解对于提升用户消费效率至关重要。视频Embedding采用稠密向量能够很好的表达出视频的语义，在推荐场景下对视频去重、相似召回、排序和多样性打散等场景都有重要的作用。本赛题从视频推荐角度出发，提供真实业务的百万量级标签数据(脱敏)，以及万量级视频相似度数据(人工标注)，用于训练embedding模型，最终根据embedding计算视频之间的余弦相似度，采用Spearman’s rank correlation与人工标注相似度计算相关性，并最终排名.

数据标注含有视频分类和视频标签两种标注

开源解决方案:

[第17名](https://github.com/chenjiashuo123/AIAC-2021-Task1-Rank17)

模型分预训练和微调两个阶段，其中预训练是使用pointwise作为训练数据，任务为tag多分类；微调阶段使用pairwise作为训练集，任务为视频对相似度回归任务。

[第一名](https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st)

将BERT引入到任务中，采用多任务联合训练的方式
