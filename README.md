# <p align="center">Merlin - Algorithmic Intelligence Toolkit</p>

<p align="center">
    <a href="https://github.com/Colin-zh/merlin-alt"><img src="https://img.shields.io/badge/status-updating-brightgreen.svg"></a>
    <a href="https://github.com/python/cpython"><img src="https://img.shields.io/badge/Python-3.12-FF1493.svg"></a>
    <a href="https://opensource.org/licenses/mit-license.php"><img src="https://badges.frapsoft.com/os/mit/mit.svg"></a>
    <a href="https://github.com/Colin-zh/merlin-alt/graphs/contributors"><img src="https://img.shields.io/github/contributors/Colin-zh/merlin-alt?color=blue"></a>
    <a href="https://github.com/Colin-zh/merlin-alt/stargazers"><img src="https://img.shields.io/github/stars/Colin-zh/merlin-alt.svg?logo=github"></a>
    <a href="https://github.com/Colin-zh/merlin-alt/network/members"><img src="https://img.shields.io/github/forks/Colin-zh/merlin-alt.svg?color=blue&logo=github"></a>
    <a href="https://www.python.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" align="right" height="48" width="48" ></a>
</p>
<br/>

> 谨记：分享，是知识的最终抵达；感恩，是成长的永恒底色。<br/>
> 知识集邮是个人意义的须臾满足，唯有分享赋能他人，才能成就真正的价值实现；前行路上的每一步，尤需铭记那些“雪中送炭”的相助，其情远胜一切“锦上添花”。<br/>
> <div align="right">-- 致自己</div>

Merlin-ALT 集成了一些常用基础的算法，附带学习和示例笔记，并提供**自动化特征工程和模型调参**的能力。

项目持续更新，学习笔记均已上传，merlin工具包暂不可用，逐步完善中。。。

## Models List
很喜欢[EZ.Encoder](https://www.youtube.com/@ez.encoder.academy)老师的学习模式：知识点的积累和理解应该是Top-Down的方式，也即“**知其源，方能思其变**”。

|Model|Package|Notebook|Paper|
|:----|:-----------|:-----------|:-----------:|
|||||
|**Tabular**||||
|LightGBM|lightgbm|[LightGBM](./merlin/charms/models/lightgbm/)||
|TabNet|-|[TabNet](./merlin/charms/models/tabnet/TabNet.ipynb)||
|Tab-Transformer|-|[Tab-Transformer](./merlin/charms/models/tab_transformer/TabTransformer.ipynb)||
|FT-Transformer|-|[FT-Transformer](./merlin/charms/models/ft_transformer/FT-Transformer.ipynb)||
|||||
|**Time-Series**||||
|Prophet|prophet|[Prophet](./merlin/charms/models/prophet/prophet.ipynb)||
|InFormer|-|[InFormer](./merlin/charms/models/informer/Informer.ipynb)||
|||||
|**Recommendation**||||
|Logistic Regression|-|[LR](./merlin/charms/models/)||
|Factorization Machine|-|[FM](./merlin/charms/models/fm/FM.ipynb)||
|Field-Factorization Machine|-|[FFM](./merlin/charms/models/ffm/FFM.ipynb)||
|Factorization-supported Neural Network|-|[FNN](./merlin/charms/models/fnn/FNN.ipynb)||
|Product-based Neural Network|-|[PNN](./merlin/charms/models/pnn/PNN.ipynb)||
|Wide & Deep|-|[Wide&Deep](./merlin/charms/models/widedeep/Wide&Deep.ipynb)||
|Deep & Cross Network|-|[DCN](./merlin/charms/models/deepcross/DCN.ipynb)||
|DeepFM|-|[DeepFM](./merlin/charms/models/deepfm/DeepFM.ipynb)||
|Nerual Factorization Machine|-|[NFM](./merlin/charms/models/nfm/NFM.ipynb)||
|Attentional Factorization|-|[AFM](./merlin/charms/models/afm/AFM.ipynb)||
|Deep Interest Network|-|[DIN](./merlin/charms/models/deepinterest/DIN.ipynb)||
|xDeepFM|-|[xDeepFM](./merlin/charms/models/xdeepfm/xDeepFM.ipynb)||
|AutoInt|-|[AutoInt](./merlin/charms/models/autoint/AutoInt.ipynb)||
|Deep Interest Evolution Network|-|[DIEN](./merlin/charms/models/dien/DIEN.ipynb)||
|Behavior Sequence Transformer|-|[BST](./merlin/charms/models/bst/BST.ipynb)||
|||||
|**Multi-Task**||||
|Multi-gate Mixture-of-Experts|-|[MMOE](./merlin/charms/models/mmoe/MMOE.ipynb)||
|Entire Space Multi-Task Model|-|[ESSM](./merlin/charms/models/esmm/ESMM.ipynb)||
|Progressive Layered Extraction|-|[PLE](./merlin/charms/models/ple/PLE.ipynb)||
|||||
|**Casual**||||
|||||
|**Graph**||||
|LPA|-|[LPA](./merlin/charms/models/lpa/LPA.ipynb)||
|Louvain|-|[Louvain](./merlin/charms/models/louvain/Louvain.ipynb)||
|GCN|-|[GCN](./merlin/charms/models/gcn/GCN.ipynb)||
|Graphsage|-|[Graphsage](./merlin/charms/models/graphsage/GraphSAGE.ipynb)||
|GAT|-|[GAT](./merlin/charms/models/gat/GAT.ipynb)||
|Leiden|-|[Leiden](./merlin/charms/models/leiden/Leiden.ipynb)||
|||||
|**NLP**||||
|BERT|-|[BERT](./merlin/charms/models/)||
|GPT|-|[GPT](./merlin/charms/models/)||
|RoBERTa|-|[RoBERTa](./merlin/charms/models/)||
|LoRA|-|[LoRA](./merlin/charms/models/)||
|||||
|**RL**||||
|Q-Learning 🔥🔥|- |[Q-learning](./merlin/charms/models/q_learning/Q-Learning.ipynb)||
|DQN|- |[DQN](./merlin/charms/models/dqn/DQN.ipynb)||


## 引用或参考
[![DeepCTR-Torch](https://img.shields.io/badge/DeepCTR--Torch-shenweichen-blue?logo=github&style=for-the-badge)](https://github.com/shenweichen/DeepCTR-Torch)<br/>
[![torchkeras](https://img.shields.io/badge/torchkeras-lyhue1991-blue?logo=github&style=for-the-badge)](https://github.com/lyhue1991/torchkeras)<br/>
[![pytorch_tabular](https://img.shields.io/badge/pytorch\_tabular-pytorch--tabular-blue?logo=github&style=for-the-badge)](https://github.com/pytorch-tabular/pytorch_tabular)<br/>
[![tab-transformer-pytorch](https://img.shields.io/badge/tab--transformer--pytorch-lucidrains-blue?logo=github&style=for-the-badge)](https://github.com/lucidrains/tab-transformer-pytorch)<br/>
[![fastprogress](https://img.shields.io/badge/fastprogress-AnswerDotAI-blue?logo=github&style=for-the-badge)](https://github.com/AnswerDotAI/fastprogress)<br/>