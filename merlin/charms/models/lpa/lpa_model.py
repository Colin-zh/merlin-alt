import logging
import random
from collections import defaultdict, Counter
from matplotlib import pyplot as plt

import pandas as pd
import networkx as nx


class LPAModel:
    def __init__(self, graph, max_iter=100, random_seed=None):
        self.graph = graph
        self.max_iter = max_iter
        self.random_seed = random_seed
        # 节点初始化方式：标签为节点本身
        self.labels = {node: node for node in graph.nodes()}
        
        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)

    def run(self):
        # 迭代终止条件检查
        for iteration in range(self.max_iter):
            changed = False
            nodes = list(self.graph.nodes())
            random.shuffle(nodes)  # 随机打乱节点顺序
            
            for node in nodes:
                # 孤立节点处理：跳过度为0的节点
                if self.graph.degree(node) == 0:
                    continue
                    
                # 获取邻居的标签
                neighbor_labels = [self.labels[neighbor] 
                                 for neighbor in self.graph.neighbors(node)]
                
                if neighbor_labels:
                    # 统计邻居标签频率
                    label_counter = Counter(neighbor_labels)
                    max_count = max(label_counter.values())
                    
                    # 收集所有达到最大次数的标签
                    best_labels = [label for label, count in label_counter.items() 
                                 if count == max_count]
                    
                    # 随机选择一个标签
                    new_label = random.choice(best_labels)
                    
                    # 检查标签是否发生变化
                    if self.labels[node] != new_label:
                        self.labels[node] = new_label
                        changed = True
            
            # 终止条件：如果没有节点标签发生变化，提前结束
            if not changed:
                logging.info(f"算法在第{iteration + 1}次迭代后收敛，提前终止")
                break
        
        if changed:
            logging.info(f"算法达到最大迭代次数{self.max_iter}后终止")

    def reset_labels(self):
        """重置标签到初始状态"""
        self.labels = {node: node for node in self.graph.nodes()}

    def get_communities(self):
        communities = defaultdict(list)
        for node, label in self.labels.items():
            communities[label].append(node)
        return dict(communities)

    def analyze_communities(self, node_attr='node_type', edge_attr='edge_type'):
        """
        分析社区划分结果，返回包含详细社区信息的DataFrame
        
        Returns:
            pandas.DataFrame: 包含以下列：
                - community_id: 社区ID
                - node_count: 社区内节点总数
                - 为每种node_type动态生成以下列：
                    * {node_type}_count: 该类型节点数量
                    * {node_type}_ids: 该类型节点ID列表
                - edge_count: 社区内部边总数
                - edge_types: 社区内每种边类型及其数量的字典
                - edges: 社区内部边列表，格式[(u,v,edge_type), ...]
                - cross_community_edges: 跨社区边列表，格式[(u,v,edge_type), ...]
        """
        # 获取社区划分结果
        communities = self.get_communities()
        
        # 获取节点类型属性
        node_types = nx.get_node_attributes(self.graph, node_attr)
        
        # 动态获取所有存在的node_type值
        all_node_types = set(node_types.values()) if node_types else set()
        
        # 获取边类型属性
        edge_types = nx.get_edge_attributes(self.graph, edge_attr)
        
        # 准备结果数据
        results = []
        
        for community_id, nodes in communities.items():
            nodes_set = set(nodes)
            
            # 1. 节点统计
            node_count = len(nodes)
            
            # 动态初始化各node_type的统计字典
            type_counts = {node_type: 0 for node_type in all_node_types}
            type_ids = {node_type: [] for node_type in all_node_types}
            
            # 按node_type分类统计
            for node in nodes:
                node_type = node_types.get(node)
                if node_type in type_counts:
                    type_counts[node_type] += 1
                    type_ids[node_type].append(node)
            
            # 2. 边统计
            internal_edges = []  # 社区内部边
            edge_type_counter = defaultdict(int)
            
            # 遍历社区内所有节点对（只处理存在的边）
            for u in nodes:
                for v in self.graph.neighbors(u):
                    # 避免重复计数，只考虑u < v的情况
                    if u < v and v in nodes_set:
                        # 获取边类型
                        edge_key = (u, v) if (u, v) in edge_types else (v, u)
                        edge_type = edge_types.get(edge_key, 'UNKNOWN')
                        
                        internal_edges.append((u, v, edge_type))
                        edge_type_counter[edge_type] += 1
            
            edge_count = len(internal_edges)
            edge_types_dict = dict(edge_type_counter)
            
            # 3. 跨社区边统计
            cross_community_edges = []
            
            for u in nodes:
                for v in self.graph.neighbors(u):
                    # 如果邻居不在当前社区
                    if v not in nodes_set:
                        # 获取边类型
                        edge_key = (u, v) if (u, v) in edge_types else (v, u)
                        edge_type = edge_types.get(edge_key, 'UNKNOWN')
                        
                        cross_community_edges.append((u, v, edge_type))
            
            # 构建结果行
            result_row = {
                'community_id': community_id,
                'node_count': node_count,
                'edge_count': edge_count,
                'edge_types': edge_types_dict,
                'edges': internal_edges,
                'cross_community_edges': cross_community_edges
            }
            
            # 动态添加各node_type的统计列
            for node_type in all_node_types:
                result_row[f'{node_type}_count'] = type_counts[node_type]
                result_row[f'{node_type}_ids'] = type_ids[node_type]
            
            results.append(result_row)
        
        # 创建DataFrame
        df = pd.DataFrame(results)
        
        # 按community_id排序
        if not df.empty:
            df = df.sort_values('community_id').reset_index(drop=True)
        
        return df

    def plot_communities(self, pos=None, figsize=(16, 6), node_size=200, font_size=6,
             with_labels=True, save_path=None, node_attr='node_type'):
        """
        并排对比绘制：左侧按node_type，右侧按社区划分
        
        参数:
            pos: 节点布局，默认使用spring布局
            figsize: 图形大小
            node_size: 节点大小
            font_size: 标签字体大小
            with_labels: 是否显示节点标签
            save_path: 保存路径，如果不为None则保存图片
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 如果没有提供布局，使用spring布局
        if pos is None:
            pos = nx.spring_layout(self.graph, seed=self.random_seed)
        
        # ========== 左侧：按node_type绘制 ==========
        node_types = nx.get_node_attributes(self.graph, node_attr)
        
        if node_types:
            unique_types = set(node_types.values())
            type_colors = {}
            cmap_type = plt.get_cmap('tab10', len(unique_types))
            for i, node_type in enumerate(unique_types):
                type_colors[node_type] = cmap_type(i)
            
            for node_type in unique_types:
                nodes_with_type = [node for node in self.graph.nodes() 
                                 if node_types.get(node) == node_type]
                
                nx.draw_networkx_nodes(self.graph, pos,
                                      nodelist=nodes_with_type,
                                      node_color=[type_colors[node_type]] * len(nodes_with_type),
                                      node_size=node_size, ax=ax1,
                                      label=f'Type: {node_type}')
            
            nx.draw_networkx_edges(self.graph, pos, alpha=0.5, ax=ax1)
            
            if with_labels:
                nx.draw_networkx_labels(self.graph, pos, font_size=font_size, ax=ax1)
            
            ax1.set_title("Graph by Node Type")
            ax1.legend()
            ax1.axis('off')
        
        # ========== 右侧：按社区划分绘制 ==========
        communities = self.get_communities()
        
        community_colors = {}
        cmap_comm = plt.get_cmap('tab20', len(communities))
        for i, label in enumerate(communities.keys()):
            community_colors[label] = cmap_comm(i)
        
        for label, nodes in communities.items():
            nx.draw_networkx_nodes(self.graph, pos,
                                  nodelist=nodes,
                                  node_color=[community_colors[label]] * len(nodes),
                                  node_size=node_size, ax=ax2,
                                  label=f'Community {label}')
        
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, ax=ax2)
        
        if with_labels:
            nx.draw_networkx_labels(self.graph, pos, font_size=font_size, ax=ax2)
        
        ax2.set_title("Graph by Community Detection")
        ax2.legend()
        ax2.axis('off')
        
        plt.suptitle("Comparison: Node Type vs Community Detection", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
