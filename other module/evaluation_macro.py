import json
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from tqdm import tqdm
import logging
from collections import defaultdict

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HPONetworkDeepWalk:
    def __init__(self, json_file_path, window_size=5, walk_length=30, num_walks=200, dimensions=128):
        """
        初始化HPO网络DeepWalk处理器

        参数:
        json_file_path: HPO JSON文件路径
        window_size: Word2Vec窗口大小
        walk_length: 随机游走长度
        num_walks: 每个节点的随机游走次数
        dimensions: 嵌入维度
        """
        self.json_file_path = json_file_path
        self.window_size = window_size
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions
        self.graph = None
        self.model = None
        self.node_embeddings = {}
        self.hpo_data = None

    def load_hpo_data(self):
        """加载HPO注释数据"""
        logger.info("正在加载HPO注释数据...")

        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.hpo_data = json.load(f)

            logger.info(f"加载完成: {len(self.hpo_data)} 个HPO术语")

            # 统计基本信息
            total_proteins = set()
            for hpo_term, proteins in self.hpo_data.items():
                total_proteins.update(proteins)

            logger.info(f"总共涉及 {len(total_proteins)} 个独特蛋白质")
            logger.info(
                f"平均每个HPO术语关联 {np.mean([len(proteins) for proteins in self.hpo_data.values()]):.1f} 个蛋白质")

            return self.hpo_data

        except Exception as e:
            logger.error(f"加载HPO数据时出错: {e}")
            return None

    def create_bipartite_graph(self):
        """创建二分图：HPO术语和蛋白质"""
        logger.info("正在创建二分图...")

        if not self.hpo_data:
            logger.error("请先加载HPO数据!")
            return None

        self.graph = nx.Graph()

        # 添加节点和边
        edge_count = 0
        for hpo_term, proteins in tqdm(self.hpo_data.items(), desc="添加HPO术语"):
            # 添加HPO术语节点，并标记类型
            self.graph.add_node(hpo_term, type='hpo')

            for protein in proteins:
                # 添加蛋白质节点，并标记类型
                self.graph.add_node(protein, type='protein')
                # 添加边（HPO术语-蛋白质关联）
                self.graph.add_edge(hpo_term, protein)
                edge_count += 1

        logger.info(f"创建的网络图包含 {self.graph.number_of_nodes()} 个节点和 {edge_count} 条边")

        # 统计节点类型
        hpo_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'hpo']
        protein_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'protein']

        logger.info(f"其中: {len(hpo_nodes)} 个HPO术语节点, {len(protein_nodes)} 个蛋白质节点")

        return self.graph

    def create_hpo_cooccurrence_graph(self):
        """创建HPO共现网络（HPO术语之间的关联）"""
        logger.info("正在创建HPO共现网络...")

        if not self.hpo_data:
            logger.error("请先加载HPO数据!")
            return None

        self.graph = nx.Graph()
        cooccurrence_matrix = defaultdict(lambda: defaultdict(int))

        # 计算HPO术语之间的共现次数
        for proteins in tqdm(self.hpo_data.values(), desc="计算共现"):
            hpo_terms = list(self.hpo_data.keys())

            for i, term1 in enumerate(hpo_terms):
                for term2 in hpo_terms[i + 1:]:
                    # 计算两个HPO术语共享的蛋白质数量
                    shared_proteins = set(self.hpo_data[term1]) & set(self.hpo_data[term2])
                    if shared_proteins:
                        cooccurrence_matrix[term1][term2] += len(shared_proteins)
                        cooccurrence_matrix[term2][term1] += len(shared_proteins)

        # 添加节点和边
        for hpo_term in self.hpo_data.keys():
            self.graph.add_node(hpo_term, type='hpo')

        for term1 in tqdm(cooccurrence_matrix, desc="添加边"):
            for term2, weight in cooccurrence_matrix[term1].items():
                if term1 < term2:  # 避免重复添加
                    self.graph.add_edge(term1, term2, weight=weight)

        logger.info(
            f"创建的共现网络包含 {self.graph.number_of_nodes()} 个HPO节点和 {self.graph.number_of_edges()} 条边")
        return self.graph

    def deep_walk_random_walk(self, start_node):
        """执行深度游走随机游走"""
        walk = [start_node]

        while len(walk) < self.walk_length:
            current_node = walk[-1]
            neighbors = list(self.graph.neighbors(current_node))

            if len(neighbors) > 0:
                # 对于带权重的图，根据权重进行概率选择
                if 'weight' in self.graph[current_node][neighbors[0]]:
                    weights = [self.graph[current_node][neighbor].get('weight', 1.0) for neighbor in neighbors]
                    probabilities = np.array(weights) / sum(weights)
                    next_node = np.random.choice(neighbors, p=probabilities)
                else:
                    # 对于无权图，均匀随机选择
                    next_node = np.random.choice(neighbors)

                walk.append(next_node)
            else:
                break

        return [str(node) for node in walk]

    def generate_walks(self):
        """为所有节点生成随机游走序列"""
        logger.info("生成随机游走序列...")
        all_walks = []
        nodes = list(self.graph.nodes())

        for walk_num in tqdm(range(self.num_walks), desc="生成游走"):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self.deep_walk_random_walk(node)
                all_walks.append(walk)

        logger.info(f"生成了 {len(all_walks)} 个游走序列")
        return all_walks

    def train_deepwalk(self, graph_type='bipartite'):
        """
        训练DeepWalk模型

        参数:
        graph_type: 'bipartite' - 二分图, 'cooccurrence' - 共现网络
        """
        # 加载数据
        self.load_hpo_data()
        if not self.hpo_data:
            return None

        # 创建图
        if graph_type == 'bipartite':
            self.create_bipartite_graph()
        elif graph_type == 'cooccurrence':
            self.create_hpo_cooccurrence_graph()
        else:
            logger.error("不支持的图类型")
            return None

        if self.graph.number_of_nodes() == 0:
            logger.error("图中没有节点")
            return None

        # 生成随机游走序列
        walks = self.generate_walks()

        # 训练Word2Vec模型
        logger.info("训练Word2Vec模型...")
        self.model = Word2Vec(
            walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=1,
            sg=1,
            workers=4,
            epochs=10
        )

        # 保存节点嵌入
        for node in self.graph.nodes():
            self.node_embeddings[node] = self.model.wv[str(node)]

        logger.info("DeepWalk训练完成!")
        return self.model

    def get_hpo_embeddings(self):
        """获取HPO术语的嵌入向量"""
        hpo_embeddings = {}
        for node, attr in self.graph.nodes(data=True):
            if attr.get('type') == 'hpo' and node in self.node_embeddings:
                hpo_embeddings[node] = self.node_embeddings[node]
        return hpo_embeddings

    def get_protein_embeddings(self):
        """获取蛋白质的嵌入向量"""
        protein_embeddings = {}
        for node, attr in self.graph.nodes(data=True):
            if attr.get('type') == 'protein' and node in self.node_embeddings:
                protein_embeddings[node] = self.node_embeddings[node]
        return protein_embeddings

    def save_embeddings(self, output_file, node_type='all'):
        """保存嵌入向量到文件"""
        if node_type == 'hpo':
            embeddings = self.get_hpo_embeddings()
        elif node_type == 'protein':
            embeddings = self.get_protein_embeddings()
        else:
            embeddings = self.node_embeddings

        embeddings_df = pd.DataFrame.from_dict(embeddings, orient='index')
        embeddings_df.to_csv(output_file, index_label='node_id')
        logger.info(f"嵌入向量已保存到 {output_file}")

    def find_similar_hpo_terms(self, query_term, top_k=10):
        """查找与查询HPO术语最相似的术语"""
        if not self.model:
            logger.error("请先训练模型!")
            return

        try:
            similar_terms = self.model.wv.most_similar(str(query_term), topn=top_k)
            print(f"\n与HPO术语 {query_term} 最相似的 {top_k} 个术语:")
            for i, (term, similarity) in enumerate(similar_terms, 1):
                print(f"{i:2d}. {term}: {similarity:.4f}")
            return similar_terms
        except KeyError:
            logger.error(f"HPO术语 {query_term} 不在词汇表中")
            return None

    def visualize_embeddings(self, node_type='hpo', top_n=50, output_file=None):
        """可视化嵌入向量"""
        if node_type == 'hpo':
            embeddings_dict = self.get_hpo_embeddings()
            title = 'HPO术语嵌入可视化'
        else:
            embeddings_dict = self.get_protein_embeddings()
            title = '蛋白质嵌入可视化'

        if not embeddings_dict:
            logger.error("没有找到嵌入向量")
            return

        nodes = list(embeddings_dict.keys())[:top_n]
        embeddings = [embeddings_dict[node] for node in nodes]

        # 使用PCA降维到2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        # 绘制可视化图
        plt.figure(figsize=(15, 12))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)

        # 添加节点标签
        label_every = max(1, top_n // 20)
        for i, node in enumerate(nodes):
            if i % label_every == 0:
                plt.annotate(node, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.8)

        plt.title(f'{title}\n{top_n} {node_type}节点')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, alpha=0.3)

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图已保存到 {output_file}")

        plt.tight_layout()
        plt.show()

        return embeddings_2d


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    hpo_deepwalk = HPONetworkDeepWalk(
        json_file_path="hpo_annotation.json",
        window_size=5,
        walk_length=30,
        num_walks=100,
        dimensions=128
    )

    # 训练模型 - 二分图版本
    logger.info("训练二分图DeepWalk模型...")
    model = hpo_deepwalk.train_deepwalk(graph_type='bipartite')

    if model:
        # 保存嵌入向量
        hpo_deepwalk.save_embeddings("hpo_bipartite_embeddings.csv", node_type='all')
        hpo_deepwalk.save_embeddings("hpo_terms_embeddings.csv", node_type='hpo')
        hpo_deepwalk.save_embeddings("proteins_embeddings.csv", node_type='protein')

        # 查找相似HPO术语
        hpo_terms = list(hpo_deepwalk.get_hpo_embeddings().keys())
        if hpo_terms:
            sample_term = hpo_terms[0]
            hpo_deepwalk.find_similar_hpo_terms(sample_term)

        # 可视化
        hpo_deepwalk.visualize_embeddings(node_type='hpo', top_n=50,
                                          output_file="hpo_embeddings_visualization.png")

    # 也可以训练共现网络版本
    logger.info("\n训练共现网络DeepWalk模型...")
    hpo_deepwalk_cooccurrence = HPONetworkDeepWalk(
        json_file_path="hpo_annotation.json",
        window_size=5,
        walk_length=30,
        num_walks=100,
        dimensions=128
    )

    model_cooccurrence = hpo_deepwalk_cooccurrence.train_deepwalk(graph_type='cooccurrence')
    if model_cooccurrence:
        hpo_deepwalk_cooccurrence.save_embeddings("hpo_cooccurrence_embeddings.csv", node_type='hpo')