# tabular models
from .lightgbm import lgb_model                     # LightGBM: 2017 (Ke et al., NIPS 2017)
from .tabnet import tabnet_model                    # TabNet: 2019 (Arik & Pfister, arXiv 2019)
from .tab_transformer import tab_transformer_model  # TabTransformer: 2020 (Huang et al., arXiv 2020)
from .ft_transformer import ft_transformer_model    # FT-Transformer: 2021 (Gorishniy et al., arXiv 2021)

# time series models
from .prophet import prophet_model                  # Prophet: 2017 (Taylor & Letham, arXiv 2017)
from .informer import informer_model                # Informer: 2020 (Zhou et al., AAAI 2021, arXiv 2020)

# recommendation models
from .lr import lr_model                            # 线性回归 (1960s-1970s)
from .fm import fm_model                            # FM: 2010 (Rendle, ICDM 2010)
from .ffm import ffm_model                          # FFM: 2016 (Juan et al., RecSys 2016)
from .fnn import fnn_model                          # FNN: 2016 (Zhang et al., ECIR 2016)
from .pnn import pnn_model                          # PNN: 2016 (Qu et al., ICDM 2016)
from .widedeep import wdl_model                     # Wide&Deep: 2016 (Cheng et al., DLRS 2016)
from .deepcross import dcn_model                    # DCN: 2017 (Wang et al., ADKDD 2017)
from .deepfm import deepfm_model                    # DeepFM: 2017 (Guo et al., IJCAI 2017)
from .nfm import nfm_model                          # NFM: 2017 (He & Chua, SIGIR 2017)
from .afm import afm_model                          # AFM: 2017 (Xiao et al., IJCAI 2017)
from .deepinterest import din_model                 # DIN: 2017 (Zhou et al., KDD 2018，实际arXiv 2017)
from .xdeepfm import xdeepfm_model                  # xDeepFM: 2018 (Lian et al., KDD 2018)
from .autoint import autoint_model                  # AutoInt: 2018 (Song et al., arXiv 2018)
from .dien import dien_model                        # DIEN: 2019 (Zhou et al., AAAI 2019)
from .bst import bst_model                          # BST: 2019 (Chen et al., CIKM 2019)

# multi-task learning models
from .mmoe import mmoe_model                        # MMoE: 2018 (Ma et al., KDD 2018)
from .esmm import esmm_model                        # ESMM: 2018 (Ma et al., SIGIR 2018)
from .ple import ple_model                          # PLE: 2020 (Tang et al., RecSys 2020)

# casual models
from .uplift import uplift_model                    # 

# graph models
from .lpa import lpa_model                          # LPA: 2002 (Zhu et al., NIPS 2002)
from .louvain import louvain_model                  # Louvain: 2008 (Blondel et al., J. Stat. Mech. 2008)
from .gcn import gcn_model                          # GCN: 2016 (Kipf & Welling, ICLR 2017, arXiv 2016)
from .graphsage import graphsage_model              # GraphSAGE: 2017 (Hamilton et al., NIPS 2017)
from .gat import gat_model                          # GAT: 2017 (Veličković et al., ICLR 2018, arXiv 2017)
from .leiden import leiden_model                    # Leiden: 2019 (Traag et al., Sci. Rep. 2019)

# language models
from .bert import bert_model                        # BERT: 2018 (Devlin et al., NAACL 2019, arXiv 2018)
from .gpt import gpt_model                          # GPT-1: 2018 (Radford et al., arXiv 2018)
from .roberta import roberta_model                  # RoBERTa: 2019 (Liu et al., arXiv 2019)

# reinforcement learning models
from .q_learning import q_learning_model            # Q-Learning: 1989 (Watkins, Machine Learning 1992)
from .dqn import dqn_model                          # DQN: 2013 (Mnih et al., Nature 2015, arXiv 2013)