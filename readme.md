# （详见course.pdf）
# Learning Prior Knowledge via LSTM and the Transformer in Text Classification Task: Performance Difference and Structured Explanation

> 小想法，关于Transformer和ViT间的比对，所有带baseline版本的条目都是我提交课设作的简单版，可忽略

## Introduction

### 背景

* Transformer在文本分类任务上做为后起之秀与基于LSTM的模型取得了有竞争力的结果
* 像Bert等大模型的工作一直在刷新sota，而面向bert的蒸馏等工作也证明，预训练模型的软知识对于轻量模型有很好的提升
* 面向NLP任务的模型有很多可解释性的工作

### 挑战

* 面向Transformer 与 LSTM间可解性对比的工作并不多，其可解释的方法关注点也不对

### 现有的方法及问题

* LSTMvis 与 Context-Free 是从时间步的角度去说的？
* 这些可解释性方法的训练真的达到最高了吗？（当然，为了统一，我们只能用软标签）

### 我们的方法与实验

<img src="idea.assets/image-20221123115756615.png" alt="image-20221123115756615" style="zoom:50%;" />

* 利用统一的蒸馏软标签进行了有易到难的课程学习, 对比了lstm与Transformer在不同参数下的性能表现
* 提出了一种新的面向自回归模型的可解释方法，比以可视化的形式证明了其有效性

### 我们的贡献点

* 提出了由易到难的蒸馏学习策略
* 通过实验对比了lstm 与 Tranformer的性能差距
* 可视化了lstm 与 Transformer 的内部可解释性

## Related Works

### 文本分类模型

### 蒸馏与微调

### NLP 模型的可解释性

### LSTM 和 Transformer 间的对比

## Methodology

### LSTM 和 Transformer的计算方式

### 基于蒸馏的训练策略

<img src="idea.assets/image-20221123120113419.png" alt="image-20221123120113419" style="zoom: 50%;" />

### 可解释性的方法

## Experiment

### 数据集简介

* 简单情感分析
  * SST2
  * IMDB
* 方面级情感分析

### 实验参数配置

* 微调，这里我们参考bert微调的工作

### 微调Bert的结果



### 蒸馏训练



### 可解释性的可视化

* base版本



## 







## Conclusion







 



