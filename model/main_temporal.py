#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import os
from src.method.utils import PPMI_matrix
from src.method.model import Mode
t.cuda.set_device(0)
def train(model, optimizer, train_data):
    train_index = train_data["train_index"]
    test_index = train_data["test_index"]

    num_protein, num_term = train_data["train_target"].shape
    num_pos = t.sum(train_data["train_target"], dim=0)
    num_neg = num_protein * t.ones_like(num_pos) - num_pos
    pos_weight = num_neg / (num_pos + 1e-5)
    classify_criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)

    label_index = np.where(np.any(train_data["test_target"], 0))[0]
    y_true = np.take(train_data["test_target"], label_index, axis=1)

    def train_epoch():
        model.train()
        optimizer.zero_grad()
        score = model(train_data["feature"], train_data["network"])
        loss = classify_criterion(score[train_index], train_data["train_target"])
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_epoch():
        model.eval()
        with t.no_grad():
            score = t.sigmoid(model(train_data["feature"], train_data["network"]))
            score = score.cpu().detach().numpy()
            y_score = np.take(score[test_index], label_index, axis=1)
            test_auc = roc_auc_score(y_true, y_score, average='macro')
            test_aupr = average_precision_score(y_true, y_score, average='macro')
            return test_auc, test_aupr, y_true, y_score

    best_auc = 0
    best_y_true = None
    best_y_score = None

    for epoch in range(300):
        trn_loss = train_epoch()
        if epoch % 25 == 0:
            tst_auc, tst_aupr, y_true_epoch, y_score_epoch = test_epoch()
            if tst_auc > best_auc:
                best_auc = tst_auc
                best_y_true = y_true_epoch
                best_y_score = y_score_epoch
        else:
            tst_auc, tst_aupr = 0, 0
        print("Epoch", epoch, "\t", trn_loss, "\t", tst_auc, "\t", tst_aupr)

    return best_y_true, best_y_score
def test(model, feat, net):
    model.eval()
    with t.no_grad():
        score = t.sigmoid(model(feat, net))
        score = score.cpu().detach().numpy().astype(float)
        return score
def plot_roc_pr_curves(y_true, y_score, save_path):
    """绘制ROC和PR曲线并保存"""
    plt.figure(figsize=(12, 5))

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc = roc_auc_score(y_true, y_score, average='macro')

    # 绘制ROC曲线
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    aupr = average_precision_score(y_true, y_score, average='macro')

    # 绘制PR曲线
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPR = {aupr:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return roc_auc, aupr
def calculate_metrics(y_true, y_score):
    """计算各种性能指标"""
    # 宏观平均
    macro_auc = roc_auc_score(y_true, y_score, average='macro')
    macro_aupr = average_precision_score(y_true, y_score, average='macro')

    # 微观平均
    micro_auc = roc_auc_score(y_true.ravel(), y_score.ravel())
    micro_aupr = average_precision_score(y_true.ravel(), y_score.ravel())

    # 每个类别的指标
    per_class_auc = []
    per_class_aupr = []
    valid_classes = 0

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:  # 确保有正样本
            try:
                auc = roc_auc_score(y_true[:, i], y_score[:, i])
                aupr = average_precision_score(y_true[:, i], y_score[:, i])
                per_class_auc.append(auc)
                per_class_aupr.append(aupr)
                valid_classes += 1
            except:
                continue

    metrics = {
        'macro_auc': float(macro_auc),
        'macro_aupr': float(macro_aupr),
        'micro_auc': float(micro_auc),
        'micro_aupr': float(micro_aupr),
        'mean_per_class_auc': float(np.mean(per_class_auc)) if per_class_auc else 0,
        'mean_per_class_aupr': float(np.mean(per_class_aupr)) if per_class_aupr else 0,
        'valid_classes': valid_classes,
        'total_classes': y_true.shape[1]
    }

    return metrics
def save_metrics(metrics, save_path):
    """保存性能指标到JSON文件"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
def temporal_validation_evaluation(dataset, protein_list, term_list, full_annotation, protein_features, networks,
                                   config):
    """时间验证评估"""
    print("=== 时间验证模式 ===")

    all_metrics = {}
    time_points = sorted(dataset["mask"].keys())

    for time_point in time_points:
        print(f"\n--- 时间点: {time_point} ---")

        train_mask = dataset["mask"][time_point]["train"].reindex(
            index=protein_list, columns=term_list, fill_value=0).values
        test_mask = dataset["mask"][time_point]["test"].reindex(
            index=protein_list, columns=term_list, fill_value=0).values

        train_annotation = full_annotation * train_mask
        train_protein_index = np.where(train_mask.any(1))[0]
        test_protein_index = np.where(test_mask.any(1))[0]
        train_target = full_annotation[train_protein_index]
        test_target = full_annotation[test_protein_index]

        print(f"训练蛋白质: {len(train_protein_index)}, 测试蛋白质: {len(test_protein_index)}")
        print(f"训练标注数: {np.sum(train_target)}, 测试标注数: {np.sum(test_target)}")

        train_data = {
            "train_target": t.FloatTensor(train_target).cuda(),
            "test_target": test_target,
            "feature": t.stack([t.FloatTensor(feat) for feat in protein_features]).cuda(),
            "network": t.stack([t.FloatTensor(net) for net in networks]).cuda(),
            "train_index": train_protein_index,
            "test_index": test_protein_index
        }

        model = Model(train_annotation.shape[0], train_annotation.shape[1], len(networks)).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 训练模型
        y_true, y_score = train(model, optimizer, train_data)

        if y_true is not None and y_score is not None:
            # 绘制曲线
            curve_path = f"D:/HPODNets-main/config/results/roc_pr_curves_time_{time_point}.png"
            roc_auc, aupr = plot_roc_pr_curves(y_true, y_score, curve_path)
            print(f"时间点 {time_point}: AUC = {roc_auc:.4f}, AUPR = {aupr:.4f}")

            # 计算指标
            metrics = calculate_metrics(y_true, y_score)
            metrics_path = f"D:/HPODNets-main/config/results/metrics_time_{time_point}.json"
            save_metrics(metrics, metrics_path)
            all_metrics[time_point] = metrics

            # 保存预测结果（保持原输出格式）
            pred_Y = test(model, train_data["feature"], train_data["network"])
            prediction = defaultdict(dict)
            for term_id, term in enumerate(term_list):
                prot_idx = np.where(test_mask[:, term_id] == 1)[0]
                y_pred = pred_Y[prot_idx, term_id]
                for i in range(len(y_pred)):
                    prediction[term][protein_list[prot_idx[i]]] = y_pred[i]

            result_path = config["result"].replace(".json", f"_time_{time_point}.json")
            with open(result_path, 'w') as fp:
                json.dump(prediction, fp, indent=2)

    # 保存所有时间点的汇总指标
    if all_metrics:
        summary_metrics = {
            'average_macro_auc': float(np.mean([m['macro_auc'] for m in all_metrics.values()])),
            'average_macro_aupr': float(np.mean([m['macro_aupr'] for m in all_metrics.values()])),
            'std_macro_auc': float(np.std([m['macro_auc'] for m in all_metrics.values()])),
            'std_macro_aupr': float(np.std([m['macro_aupr'] for m in all_metrics.values()])),
            'time_points': list(all_metrics.keys()),
            'detailed_metrics': all_metrics
        }
        save_metrics(summary_metrics, "D:/HPODNets-main/config/results/temporal_validation_summary.json")

        print(f"\n=== 时间验证汇总 ===")
        print(f"平均Macro AUC: {summary_metrics['average_macro_auc']:.4f} ± {summary_metrics['std_macro_auc']:.4f}")
        print(f"平均Macro AUPR: {summary_metrics['average_macro_aupr']:.4f} ± {summary_metrics['std_macro_aupr']:.4f}")


if __name__ == "__main__":
    with open("D:/HPODNets-main/config/method/main_temporal.json") as fp:
        config = json.load(fp)

    # 创建结果目录
    os.makedirs("D:/HPODNets-main/config/results", exist_ok=True)

    # 加载数据
    with open(config["dataset"], 'rb') as fp:
        dataset = pickle.load(fp)

    term_freq = dataset["annotation"].sum(axis=0)
    term_list = term_freq[term_freq > 10].index.tolist()
    full_annotation = dataset["annotation"][term_list]
    protein_list = list(full_annotation.index)
    full_annotation = full_annotation.values

    # 加载网络
    networks = []
    for path in config["network"]:
        with open(path) as fp:
            ppi = json.load(fp)
        ppi = pd.DataFrame(ppi).fillna(0).reindex(
            columns=protein_list, index=protein_list, fill_value=0).values
        diag = 1 / np.sqrt(np.sum(ppi, 1))
        diag[diag == np.inf] = 0
        neg_half_power_degree_matrix = np.diag(diag)
        normalized_ppi = np.matmul(np.matmul(neg_half_power_degree_matrix, ppi),
                                   neg_half_power_degree_matrix)
        networks.append(normalized_ppi)

    protein_features = [PPMI_matrix(net) for net in networks]

    # 根据模式选择执行路径
    if config["mode"] == "temporal":
        # 时间验证模式
        temporal_validation_evaluation(dataset, protein_list, term_list, full_annotation,
                                       protein_features, networks, config)
    elif config["mode"] == "cv":
        # 交叉验证模式（原代码）
        for fold in range(5):
            print("Fold", fold)

            train_mask = dataset["mask"][fold]["train"].reindex(
                index=protein_list, columns=term_list, fill_value=0).values
            test_mask = dataset["mask"][fold]["test"].reindex(
                index=protein_list, columns=term_list, fill_value=0).values
            train_annotation = full_annotation * train_mask

            train_protein_index = np.where(train_mask.any(1))[0]
            test_protein_index = np.where(test_mask.any(1))[0]
            train_target = full_annotation[train_protein_index]
            test_target = full_annotation[test_protein_index]

            train_data = {
                "train_target": t.FloatTensor(train_target).cuda(),
                "test_target": test_target,
                "feature": t.stack([t.FloatTensor(feat) for feat in protein_features]).cuda(),
                "network": t.stack([t.FloatTensor(net) for net in networks]).cuda(),
                "train_index": train_protein_index,
                "test_index": test_protein_index
            }

            model = Model(train_annotation.shape[0], train_annotation.shape[1], len(networks)).cuda()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            # 训练并获取预测结果
            y_true, y_score = train(model, optimizer, train_data)

            # 绘制曲线和保存指标
            if y_true is not None and y_score is not None:
                curve_path = f"D:/HPODNets-main/config/results/roc_pr_curves_fold_{fold}.png"
                roc_auc, aupr = plot_roc_pr_curves(y_true, y_score, curve_path)

                metrics = calculate_metrics(y_true, y_score)
                metrics_path = f"D:/HPODNets-main/config/results/metrics_fold_{fold}.json"
                save_metrics(metrics, metrics_path)

                print(f"Fold {fold}: AUC = {roc_auc:.4f}, AUPR = {aupr:.4f}")

            # 保存预测结果（保持原格式）
            pred_Y = test(model, train_data["feature"], train_data["network"])
            prediction = defaultdict(dict)
            for term_id, term in enumerate(term_list):
                prot_idx = np.where(test_mask[:, term_id] == 1)[0]
                y_pred = pred_Y[prot_idx, term_id]
                for i in range(len(y_pred)):
                    prediction[term][protein_list[prot_idx[i]]] = y_pred[i]

            with open(config["result"].format(fold), 'w') as fp:
                json.dump(prediction, fp, indent=2)
    else:
        # 单次训练模式
        train_mask = dataset["mask"]["train"].reindex(
            index=protein_list, columns=term_list, fill_value=0).values
        test_mask = dataset["mask"]["test"].reindex(
            index=protein_list, columns=term_list, fill_value=0).values
        train_annotation = full_annotation * train_mask

        train_protein_index = np.where(train_mask.any(1))[0]
        test_protein_index = np.where(test_mask.any(1))[0]
        train_target = full_annotation[train_protein_index]
        test_target = full_annotation[test_protein_index]

        train_data = {
            "train_target": t.FloatTensor(train_target).cuda(),
            "test_target": test_target,
            "feature": t.stack([t.FloatTensor(feat) for feat in protein_features]).cuda(),
            "network": t.stack([t.FloatTensor(net) for net in networks]).cuda(),
            "train_index": train_protein_index,
            "test_index": test_protein_index
        }

        model = Model(train_annotation.shape[0], train_annotation.shape[1], len(networks)).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 训练并获取预测结果
        y_true, y_score = train(model, optimizer, train_data)

        # 绘制曲线和保存指标
        if y_true is not None and y_score is not None:
            curve_path = "D:/HPODNets-main/config/results/roc_pr_curves.png"
            roc_auc, aupr = plot_roc_pr_curves(y_true, y_score, curve_path)

            metrics = calculate_metrics(y_true, y_score)
            metrics_path = "D:/HPODNets-main/config/results/metrics.json"
            save_metrics(metrics, metrics_path)

            print(f"Final: AUC = {roc_auc:.4f}, AUPR = {aupr:.4f}")

        # 保存预测结果（保持原格式）
        pred_Y = test(model, train_data["feature"], train_data["network"])
        prediction = defaultdict(dict)
        for term_id, term in enumerate(term_list):
            prot_idx = np.where(test_mask[:, term_id] == 1)[0]
            y_pred = pred_Y[prot_idx, term_id]
            for i in range(len(y_pred)):
                prediction[term][protein_list[prot_idx[i]]] = y_pred[i]

        with open(config["result"], 'w') as fp:
            json.dump(prediction, fp, indent=2)




'''
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import os
import traceback
from src.method.utils import PPMI_matrix
from src.method.model import Model

t.cuda.set_device(0)


def train(model, optimizer, train_data):
    train_index = train_data["train_index"]
    test_index = train_data["test_index"]

    num_protein, num_term = train_data["train_target"].shape
    num_pos = t.sum(train_data["train_target"], dim=0)
    num_neg = num_protein * t.ones_like(num_pos) - num_pos
    pos_weight = num_neg / (num_pos + 1e-5)
    classify_criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)

    label_index = np.where(np.any(train_data["test_target"], 0))[0]
    y_true = np.take(train_data["test_target"], label_index, axis=1)

    def train_epoch():
        model.train()
        optimizer.zero_grad()
        score = model(train_data["feature"], train_data["network"])
        loss = classify_criterion(score[train_index], train_data["train_target"])
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_epoch():
        model.eval()
        with t.no_grad():
            score = t.sigmoid(model(train_data["feature"], train_data["network"]))
            score = score.cpu().detach().numpy()
            y_score = np.take(score[test_index], label_index, axis=1)
            test_auc = roc_auc_score(y_true, y_score, average='macro')
            test_aupr = average_precision_score(y_true, y_score, average='macro')
            return test_auc, test_aupr, y_true, y_score

    best_auc = 0
    best_y_true = None
    best_y_score = None

    for epoch in range(300):
        trn_loss = train_epoch()
        if epoch % 25 == 0:
            tst_auc, tst_aupr, y_true_epoch, y_score_epoch = test_epoch()
            if tst_auc > best_auc:
                best_auc = tst_auc
                best_y_true = y_true_epoch
                best_y_score = y_score_epoch
        else:
            tst_auc, tst_aupr = 0, 0
        print("Epoch", epoch, "\t", trn_loss, "\t", tst_auc, "\t", tst_aupr)

    return best_y_true, best_y_score


def test(model, feat, net):
    model.eval()
    with t.no_grad():
        score = t.sigmoid(model(feat, net))
        score = score.cpu().detach().numpy().astype(float)
        return score


def plot_roc_pr_curves(y_true, y_score, save_path):
    """绘制ROC和PR曲线并保存"""
    try:
        plt.figure(figsize=(12, 5))

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
        roc_auc = roc_auc_score(y_true, y_score, average='macro')

        # 绘制ROC曲线
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
        aupr = average_precision_score(y_true, y_score, average='macro')

        # 绘制PR曲线
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPR = {aupr:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return roc_auc, aupr
    except Exception as e:
        print(f"绘制曲线时出错: {e}")
        return 0, 0


def calculate_metrics(y_true, y_score):
    """计算各种性能指标"""
    try:
        # 宏观平均
        macro_auc = roc_auc_score(y_true, y_score, average='macro')
        macro_aupr = average_precision_score(y_true, y_score, average='macro')

        # 微观平均
        micro_auc = roc_auc_score(y_true.ravel(), y_score.ravel())
        micro_aupr = average_precision_score(y_true.ravel(), y_score.ravel())

        # 每个类别的指标
        per_class_auc = []
        per_class_aupr = []
        valid_classes = 0

        for i in range(y_true.shape[1]):
            if np.sum(y_true[:, i]) > 0:  # 确保有正样本
                try:
                    auc = roc_auc_score(y_true[:, i], y_score[:, i])
                    aupr = average_precision_score(y_true[:, i], y_score[:, i])
                    per_class_auc.append(auc)
                    per_class_aupr.append(aupr)
                    valid_classes += 1
                except:
                    continue

        metrics = {
            'macro_auc': float(macro_auc),
            'macro_aupr': float(macro_aupr),
            'micro_auc': float(micro_auc),
            'micro_aupr': float(micro_aupr),
            'mean_per_class_auc': float(np.mean(per_class_auc)) if per_class_auc else 0,
            'mean_per_class_aupr': float(np.mean(per_class_aupr)) if per_class_aupr else 0,
            'valid_classes': valid_classes,
            'total_classes': y_true.shape[1]
        }

        return metrics
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return {}


def save_metrics(metrics, save_path):
    """保存性能指标到JSON文件"""
    try:
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        print(f"保存指标时出错: {e}")


def check_pickle_file(file_path):
    """检查pickle文件是否有效"""
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"错误: Pickle文件为空: {file_path}")
            return False

        # 尝试读取文件头来检查是否是有效的pickle文件
        with open(file_path, 'rb') as f:
            pickle.load(f)

        print(f"Pickle文件检查通过: {file_path} (大小: {file_size} 字节)")
        return True
    except EOFError:
        print(f"错误: Pickle文件损坏或格式不正确: {file_path}")
        return False
    except Exception as e:
        print(f"检查pickle文件时出错: {e}")
        return False


def load_dataset_safe(file_path):
    """安全地加载数据集"""
    try:
        if not check_pickle_file(file_path):
            return None

        with open(file_path, 'rb') as fp:
            dataset = pickle.load(fp)

        # 检查数据集结构
        required_keys = ['annotation', 'mask']
        for key in required_keys:
            if key not in dataset:
                print(f"错误: 数据集中缺少必需的键: {key}")
                return None

        print("数据集加载成功")
        print(f"标注数据形状: {dataset['annotation'].shape}")
        print(f"掩码键: {list(dataset['mask'].keys())}")

        return dataset
    except Exception as e:
        print(f"加载数据集时发生错误: {e}")
        print(traceback.format_exc())
        return None


def create_dummy_dataset():
    """创建虚拟数据集用于测试"""
    print("创建虚拟数据集用于测试...")

    # 创建虚拟数据
    n_proteins = 100
    n_terms = 50

    # 虚拟标注矩阵
    annotation = pd.DataFrame(
        np.random.randint(0, 2, (n_proteins, n_terms)),
        index=[f"protein_{i}" for i in range(n_proteins)],
        columns=[f"term_{i}" for i in range(n_terms)]
    )

    # 虚拟掩码
    mask = {
        "train": pd.DataFrame(
            np.random.randint(0, 2, (n_proteins, n_terms)),
            index=annotation.index,
            columns=annotation.columns
        ),
        "test": pd.DataFrame(
            np.random.randint(0, 2, (n_proteins, n_terms)),
            index=annotation.index,
            columns=annotation.columns
        )
    }

    return {'annotation': annotation, 'mask': mask}


if __name__ == "__main__":
    try:
        with open("D:/HPODNets-main/config/method/main_temporal.json") as fp:
            config = json.load(fp)

        print("配置文件加载成功")
        print(f"数据集路径: {config['dataset']}")
        print(f"模式: {config['mode']}")

        # 创建结果目录
        os.makedirs("D:/HPODNets-main/config/results", exist_ok=True)

        # 尝试加载数据集
        dataset_path = config["dataset"]
        if not os.path.exists(dataset_path):
            print(f"错误: 数据集文件不存在: {dataset_path}")
            print("请检查配置文件中的路径是否正确")
            exit(1)

        dataset = load_dataset_safe(dataset_path)

        if dataset is None:
            print("无法加载数据集，请检查文件是否损坏")
            print("是否使用虚拟数据进行测试? (y/n)")
            choice = input().strip().lower()
            if choice == 'y':
                dataset = create_dummy_dataset()
                print("使用虚拟数据进行测试")
            else:
                exit(1)

        # 处理数据
        term_freq = dataset["annotation"].sum(axis=0)
        term_list = term_freq[term_freq > 10].index.tolist()
        if not term_list:
            print("警告: 没有找到频率>10的term，使用所有term")
            term_list = dataset["annotation"].columns.tolist()

        full_annotation = dataset["annotation"][term_list]
        protein_list = list(full_annotation.index)
        full_annotation = full_annotation.values

        print(f"处理后的标注矩阵形状: {full_annotation.shape}")
        print(f"蛋白质数量: {len(protein_list)}")
        print(f"术语数量: {len(term_list)}")

        # 加载网络
        networks = []
        for i, path in enumerate(config["network"]):
            if not os.path.exists(path):
                print(f"警告: 网络文件不存在: {path}")
                continue

            try:
                with open(path) as fp:
                    ppi = json.load(fp)

                ppi_df = pd.DataFrame(ppi).fillna(0)
                ppi_reindexed = ppi_df.reindex(
                    columns=protein_list, index=protein_list, fill_value=0).values

                diag = 1 / np.sqrt(np.sum(ppi_reindexed, 1))
                diag[diag == np.inf] = 0
                neg_half_power_degree_matrix = np.diag(diag)
                normalized_ppi = np.matmul(np.matmul(neg_half_power_degree_matrix, ppi_reindexed),
                                           neg_half_power_degree_matrix)
                networks.append(normalized_ppi)
                print(f"网络 {i + 1} 加载成功: {normalized_ppi.shape}")

            except Exception as e:
                print(f"加载网络 {path} 时出错: {e}")
                continue

        if not networks:
            print("错误: 没有成功加载任何网络")
            exit(1)

        protein_features = [PPMI_matrix(net) for net in networks]

        # 根据模式选择执行路径
        if config["mode"] == "temporal":
            print("时间验证模式")
            # 这里可以添加时间验证的具体实现
            # temporal_validation_evaluation(dataset, protein_list, term_list, full_annotation,
            #                               protein_features, networks, config)

        elif config["mode"] == "cv":
            print("交叉验证模式")
            # 这里可以添加交叉验证的具体实现

        else:
            print("单次训练模式")
            # 这里可以添加单次训练的具体实现

    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        print(traceback.format_exc())'''