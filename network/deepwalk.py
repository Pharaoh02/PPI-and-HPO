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
import re

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class STRINGPPIDeepWalk:
    def __init__(self, json_file_path, window_size=5, walk_length=30, num_walks=200, dimensions=128, min_weight=0.15):
        self.json_file_path = json_file_path
        self.window_size = window_size
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions
        self.min_weight = min_weight
        self.graph = None
        self.model = None
        self.node_embeddings = {}

    def fix_json_format(self, content):
        """修复JSON格式问题"""
        # 修复数字键没有引号的问题
        content = re.sub(r'(\n\s*)(\d+)(\s*:)', r'\1"\2"\3', content)

        # 修复其他可能的格式问题
        content = re.sub(r',\s*}', '}', content)  # 移除尾随逗号
        content = re.sub(r',\s*]', ']', content)  # 移除尾随逗号

        return content

    def load_string_data(self, sample_size=None):
        """
        加载STRING JSON数据 - 处理格式问题
        """
        logger.info(f"开始加载STRING数据，最小权重阈值: {self.min_weight}")

        ppi_data = {}
        node_count = 0
        edge_count = 0

        try:
            # 首先读取整个文件内容
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            logger.info("检测到单个JSON对象格式，尝试修复格式问题...")

            # 修复JSON格式
            fixed_content = self.fix_json_format(content)

            try:
                # 尝试解析修复后的JSON
                data = json.loads(fixed_content)
                logger.info("JSON解析成功!")

                # 处理数据
                for i, (protein_id, interactions) in enumerate(tqdm(data.items(), desc="处理蛋白质数据")):
                    if sample_size and node_count >= sample_size:
                        break

                    if not isinstance(interactions, dict):
                        continue

                    filtered_interactions = {}
                    for target, weight in interactions.items():
                        try:
                            weight_val = float(weight)
                            if weight_val >= self.min_weight:
                                filtered_interactions[target] = weight_val
                                edge_count += 1
                        except (ValueError, TypeError):
                            continue

                    if filtered_interactions:
                        ppi_data[protein_id] = filtered_interactions
                        node_count += 1

            except json.JSONDecodeError as e:
                logger.error(f"JSON解析仍然失败: {e}")
                logger.info("尝试逐行解析...")

                # 如果整体解析失败，尝试逐行解析
                ppi_data = self.parse_line_by_line(content, sample_size)
                if ppi_data:
                    node_count = len(ppi_data)
                    edge_count = sum(len(targets) for targets in ppi_data.values())

        except Exception as e:
            logger.error(f"加载文件时出错: {e}")
            return None

        logger.info(f"加载完成: {node_count} 个蛋白质, {edge_count} 个相互作用")
        return ppi_data

    def parse_line_by_line(self, content, sample_size=None):
        """逐行解析JSON内容"""
        ppi_data = {}
        current_protein = None
        current_interactions = {}

        lines = content.split('\n')
        for i, line in tqdm(enumerate(lines), desc="逐行解析", total=len(lines)):
            line = line.strip()

            if sample_size and len(ppi_data) >= sample_size:
                break

            if not line:
                continue

            # 匹配蛋白质ID行: "P84085": {
            protein_match = re.match(r'"?([A-Z0-9]+)"?\s*:\s*{', line)
            if protein_match:
                # 保存前一个蛋白质的数据
                if current_protein and current_interactions:
                    ppi_data[current_protein] = current_interactions

                current_protein = protein_match.group(1)
                current_interactions = {}
                continue

            # 匹配相互作用行: "Q86X27": 0.173,
            interaction_match = re.match(r'"?([A-Z0-9]+)"?\s*:\s*([0-9.]+)', line)
            if interaction_match and current_protein:
                target = interaction_match.group(1)
                try:
                    weight = float(interaction_match.group(2))
                    if weight >= self.min_weight:
                        current_interactions[target] = weight
                except ValueError:
                    continue

            # 匹配结束括号: },
            if re.match(r'}\s*,?\s*$', line) and current_protein and current_interactions:
                ppi_data[current_protein] = current_interactions
                current_protein = None
                current_interactions = {}

        # 添加最后一个蛋白质
        if current_protein and current_interactions:
            ppi_data[current_protein] = current_interactions

        return ppi_data

    def create_graph(self, ppi_data):
        """从PPI数据创建网络图"""
        logger.info("正在创建网络图...")
        self.graph = nx.Graph()

        # 添加节点和边
        for source_node, targets in tqdm(ppi_data.items(), desc="构建图"):
            self.graph.add_node(source_node)

            for target_node, weight in targets.items():
                self.graph.add_node(target_node)
                self.graph.add_edge(source_node, target_node, weight=weight)

        logger.info(f"创建的网络图包含 {self.graph.number_of_nodes()} 个节点和 {self.graph.number_of_edges()} 条边")
        return self.graph

    def deep_walk_random_walk(self, start_node):
        """执行深度游走随机游走"""
        walk = [start_node]

        while len(walk) < self.walk_length:
            current_node = walk[-1]
            neighbors = list(self.graph.neighbors(current_node))

            if len(neighbors) > 0:
                # 根据边权重进行概率选择
                weights = []
                valid_neighbors = []

                for neighbor in neighbors:
                    if self.graph.has_edge(current_node, neighbor):
                        weight = self.graph[current_node][neighbor].get('weight', 1.0)
                        weights.append(weight)
                        valid_neighbors.append(neighbor)

                if weights:
                    probabilities = np.array(weights) / sum(weights)
                    next_node = np.random.choice(valid_neighbors, p=probabilities)
                    walk.append(next_node)
                else:
                    break
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

    def train_deepwalk(self, sample_size=1000):
        """训练DeepWalk模型"""
        # 加载数据
        ppi_data = self.load_string_data(sample_size)
        if not ppi_data:
            logger.error("无法加载PPI数据")
            return None

        # 创建图
        self.create_graph(ppi_data)

        if self.graph.number_of_nodes() == 0:
            logger.error("图中没有节点")
            return None

        # 检查图是否连通
        if not nx.is_connected(self.graph):
            logger.warning("图不是连通的，使用最大连通分量")
            largest_cc = max(nx.connected_components(self.graph), key=len)
            self.graph = self.graph.subgraph(largest_cc).copy()
            logger.info(f"使用最大连通分量: {self.graph.number_of_nodes()} 个节点")

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

    def get_all_embeddings(self):
        """获取所有节点的嵌入"""
        return self.node_embeddings

    def save_embeddings(self, output_file):
        """保存嵌入向量到文件"""
        embeddings_df = pd.DataFrame.from_dict(self.node_embeddings, orient='index')
        embeddings_df.to_csv(output_file, index_label='protein_id')
        logger.info(f"嵌入向量已保存到 {output_file}")


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    deepwalk_processor = STRINGPPIDeepWalk(
        json_file_path="STRING.v12.0.json",
        window_size=3,
        walk_length=20,
        num_walks=50,
        dimensions=64,
        min_weight=0.15
    )

    # 训练模型
    logger.info("开始训练DeepWalk模型...")
    model = deepwalk_processor.train_deepwalk(sample_size=1000)

    if model:
        # 获取所有嵌入
        embeddings = deepwalk_processor.get_all_embeddings()
        logger.info(f"生成的嵌入向量维度: {list(embeddings.values())[0].shape}")

        # 保存嵌入向量
        deepwalk_processor.save_embeddings("string_ppi_embeddings.csv")
        logger.info("处理完成!")