

# AIChallenger2018_zsl
[AIChallenger2018_zsl](https://challenger.ai/competition/zsl2018)

### 赛题简介
本竞赛由创新工场、北京大学王亦洲教授和复旦大学付彦伟教授联合举办。

本次零样本学习（zero-shot learning）竞赛的任务是在已知类别上训练物体识别模型，要求模型能够用于识别来自未知类别的样本。
本次竞赛提供了属性，用于实现从已知类别到未知类别的知识迁移。要求参赛选手提交对测试样本的标签预测值。

### 数据说明
数据集分Test A和Test B两部分。Test A包含动物（Animals）、水果（Fruits）两个超类。Test B包含交通工具（Vehicles）、电子产品（Electronics）、发型（Hairstyles）三个超类。对于每个超类均包含训练集（80%类别）和测试集（20%类别）。训练集所有图片均标注了标签和包围框。对于部分图片（20张/类），标注了二值属性，属性值为0 或 1，表示属性"存在"或"不存在"。对于测试集中的未知类别，仅提供类别级的属性用作知识迁移。

### 零样本学习ZSL示意图
 <img src="data/zsl_demo.png" width = "610" height = "320" alt="图片名称" align=center />

### adcv 方案介绍

1. Global Semantic Consistency Network (GSC-Net)
