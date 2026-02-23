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
|TabNet|-|[TabNet](./merlin/charms/models/tabnet/TabNet.ipynb)|[TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)|
|Tab-Transformer|-|[Tab-Transformer](./merlin/charms/models/tab_transformer/TabTransformer.ipynb)|[TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)|
|FT-Transformer|-|[FT-Transformer](./merlin/charms/models/ft_transformer/FT-Transformer.ipynb)|[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)|
|||||
|**Time-Series**||||
|Prophet|prophet|[Prophet](./merlin/charms/models/prophet/prophet.ipynb)|[Forecasting at Scale](https://peerj.com/preprints/3190/)|
|InFormer|-|[InFormer](./merlin/charms/models/informer/Informer.ipynb)|[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/pdf/2012.07436)|
|||||
|**Recommendation**||||
|Logistic Regression|-|[LR](./merlin/charms/models/)||
|Factorization Machine|-|[FM](./merlin/charms/models/fm/FM.ipynb)||
|Field-Factorization Machine|-|[FFM](./merlin/charms/models/ffm/FFM.ipynb)||
|Factorization-supported Neural Network|-|[FNN](./merlin/charms/models/fnn/FNN.ipynb)|[Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376)|
|Product-based Neural Network|-|[PNN](./merlin/charms/models/pnn/PNN.ipynb)|[Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144)|
|Wide & Deep|-|[Wide&Deep](./merlin/charms/models/widedeep/Wide&Deep.ipynb)|[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)|
|Deep & Cross Network|-|[DCN](./merlin/charms/models/deepcross/DCN.ipynb)|[Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)|
|DeepFM|-|[DeepFM](./merlin/charms/models/deepfm/DeepFM.ipynb)|[DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)|
|Nerual Factorization Machine|-|[NFM](./merlin/charms/models/nfm/NFM.ipynb)|[Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027)|
|Attentional Factorization|-|[AFM](./merlin/charms/models/afm/AFM.ipynb)|[Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435)|
|Deep Interest Network|-|[DIN](./merlin/charms/models/deepinterest/DIN.ipynb)|[Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)|
|xDeepFM|-|[xDeepFM](./merlin/charms/models/xdeepfm/xDeepFM.ipynb)|[xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170)|
|AutoInt|-|[AutoInt](./merlin/charms/models/autoint/AutoInt.ipynb)|[AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)|
|Deep Interest Evolution Network|-|[DIEN](./merlin/charms/models/dien/DIEN.ipynb)|[Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672)|
|Behavior Sequence Transformer|-|[BST](./merlin/charms/models/bst/BST.ipynb)|[Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874v1)|
|||||
|**Multi-Task**||||
|Multi-gate Mixture-of-Experts|-|[MMOE](./merlin/charms/models/mmoe/MMOE.ipynb)|[Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007)|
|Entire Space Multi-Task Model|-|[ESSM](./merlin/charms/models/esmm/ESMM.ipynb)|[Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)|
|Progressive Layered Extraction|-|[PLE](./merlin/charms/models/ple/PLE.ipynb)|[Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236)|
|||||
|**Casual**||||
|||||
|Uplift|-|[Uplift](./merlin/charms/models/uplift/Uplift.ipynb)|[Theoretical Impediments to Machine Learning With Seven Sparks from the Causal Revolution](https://arxiv.org/abs/1801.04016)|
|**Graph**||||
|LPA|-|[LPA](./merlin/charms/models/lpa/LPA.ipynb)|[Near linear time algorithm to detect community structures in large-scale networks](https://arxiv.org/abs/0709.2938)|
|Louvain|-|[Louvain](./merlin/charms/models/louvain/Louvain.ipynb)|[Fast unfolding of communities in large networks](https://arxiv.org/pdf/0803.0476)|
|GCN|-|[GCN](./merlin/charms/models/gcn/GCN.ipynb)|[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)|
|Graphsage|-|[Graphsage](./merlin/charms/models/graphsage/GraphSAGE.ipynb)|[Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)|
|GAT|-|[GAT](./merlin/charms/models/gat/GAT.ipynb)|[Graph Attention Networks](https://arxiv.org/abs/1710.10903)|
|Leiden|-|[Leiden](./merlin/charms/models/leiden/Leiden.ipynb)|[From Louvain to Leiden: guaranteeing well-connected communities](https://arxiv.org/abs/1810.08473)|
|||||
|**NLP**||||
|BERT|-|[BERT](./merlin/charms/models/)||
|GPT|-|[GPT](./merlin/charms/models/)||
|RoBERTa|-|[RoBERTa](./merlin/charms/models/)||
|LoRA|-|[LoRA](./merlin/charms/models/)||
|||||
|**RL**||||
|Q-Learning 🔥🔥|- |[Q-learning](./merlin/charms/models/q_learning/Q-Learning.ipynb)|[Q-learning](https://link.springer.com/article/10.1007/bf00992698)|
|DQN|- |[DQN](./merlin/charms/models/dqn/DQN.ipynb)|[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)|


## 引用或参考
[![DeepCTR-Torch](https://img.shields.io/badge/DeepCTR--Torch-shenweichen-blue?logo=github&style=for-the-badge)](https://github.com/shenweichen/DeepCTR-Torch)<br/>
[![torchkeras](https://img.shields.io/badge/torchkeras-lyhue1991-blue?logo=github&style=for-the-badge)](https://github.com/lyhue1991/torchkeras)<br/>
[![pytorch_tabular](https://img.shields.io/badge/pytorch\_tabular-pytorch--tabular-blue?logo=github&style=for-the-badge)](https://github.com/pytorch-tabular/pytorch_tabular)<br/>
[![tab-transformer-pytorch](https://img.shields.io/badge/tab--transformer--pytorch-lucidrains-blue?logo=github&style=for-the-badge)](https://github.com/lucidrains/tab-transformer-pytorch)<br/>
[![fastprogress](https://img.shields.io/badge/fastprogress-AnswerDotAI-blue?logo=github&style=for-the-badge)](https://github.com/AnswerDotAI/fastprogress)<br/>