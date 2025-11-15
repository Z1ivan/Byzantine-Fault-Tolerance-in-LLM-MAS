#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_probe_test import SimpleProbe, preprocess_data, print_results_summary
import numpy as np
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    'gsm8k_llama3': {
        'name': 'GSM8K (LLaMA-3)',
        'base_path': 'data/hidden_states/gsm8k/llama3',
        'structure': 'split_dirs',
        'splits': {'train': 'train', 'test': 'test'},
        'label_field': 'is_correct',
        'label_file': 'data_with_labels.json',
        'states_types': ['query', 'answer', 'pooled', 'targeted'],
        'best_layer': 16,
        'recommended_pca': 256
    },
    'gsm8k_llama31': {
        'name': 'GSM8K (LLaMA-3.1)',
        'base_path': 'data/hidden_states/gsm8k/llama31',
        'structure': 'split_dirs',
        'splits': {'train': 'train', 'test': 'test'},
        'label_field': 'is_correct',
        'label_file': 'data_with_labels.json',
        'states_types': ['query', 'answer', 'pooled', 'targeted'],
        'best_layer': 12,
        'recommended_pca': 256
    },
    'safe_llama3': {
        'name': 'Safe (LLaMA-3)',
        'base_path': 'data/hidden_states/safe/llama3',
        'structure': 'split_dirs',
        'splits': {'train': 'train', 'test': 'test'},
        'label_field': 'label',
        'label_file': 'data.json',
        'states_types': ['query', 'answer', 'pooled'],
        'best_layer': 31,
        'recommended_pca': 256,
        'label_mapping': {'safe': 1, 'unsafe': 0}           
    },
    'safe_llama31': {
        'name': 'Safe (LLaMA-3.1)',
        'base_path': 'data/hidden_states/safe/llama31',
        'structure': 'split_dirs',
        'splits': {'train': 'train', 'test': 'test'},
        'label_field': 'label',
        'label_file': 'data.json',
        'states_types': ['query', 'answer', 'pooled'],
        'best_layer': 4,
        'recommended_pca': 256,
        'label_mapping': {'safe': 1, 'unsafe': 0}           
    },
    'commonsense_llama3': {
        'name': 'CommonsenseQA (LLaMA-3)',
        'base_path': 'data/hidden_states/commonsense_qa/llama3',
        'structure': 'split_dirs',
        'splits': {'train': 'train', 'test': 'validation'},
        'label_field': 'is_correct',
        'label_file': 'data_with_labels.json',
        'states_types': ['query', 'answer', 'pooled'],
        'best_layer': 16,
        'recommended_pca': 256
    },
    'commonsense_llama31': {
        'name': 'CommonsenseQA (LLaMA-3.1)',
        'base_path': 'data/hidden_states/commonsense_qa/llama31',
        'structure': 'split_dirs',
        'splits': {'train': 'train', 'test': 'validation'},
        'label_field': 'is_correct',
        'label_file': 'data_with_labels.json',
        'states_types': ['query', 'answer', 'pooled'],
        'best_layer': 12,
        'recommended_pca': 256
    }
}

def load_data_unified(dataset_name, project_root=None):

    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {dataset_name}. 可用: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    base_path = Path(project_root) / config['base_path']

    logger.info(f"加载数据集: {config['name']}")
    logger.info(f"   路径: {base_path}")

    if not base_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {base_path}")

    train_split = config['splits']['train']
    test_split = config['splits']['test']
    label_field = config['label_field']
    label_file = config['label_file']

    train_dir = base_path / train_split
    test_dir = base_path / test_split

    train_json = train_dir / f"{train_split}_{label_file}"
    test_json = test_dir / f"{test_split}_{label_file}"

    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    label_mapping = config.get('label_mapping', None)
    if label_mapping:

        train_labels = np.array([label_mapping[item[label_field]] for item in train_data])
        test_labels = np.array([label_mapping[item[label_field]] for item in test_data])
    else:

        train_labels = np.array([item[label_field] for item in train_data])
        test_labels = np.array([item[label_field] for item in test_data])

    train_query = np.load(train_dir / f"{train_split}_query_hidden_states.npy")
    train_answer = np.load(train_dir / f"{train_split}_answer_hidden_states.npy")
    train_pooled = np.load(train_dir / f"{train_split}_pooled_hidden_states.npy")

    test_query = np.load(test_dir / f"{test_split}_query_hidden_states.npy")
    test_answer = np.load(test_dir / f"{test_split}_answer_hidden_states.npy")
    test_pooled = np.load(test_dir / f"{test_split}_pooled_hidden_states.npy")

    logger.info(f"   训练: {len(train_data)} 样本, 测试: {len(test_data)} 样本")
    logger.info(f"   形状: {train_query.shape}")
    logger.info(f"   正样本率: 训练={np.mean(train_labels):.3f}, 测试={np.mean(test_labels):.3f}")

    result = {
        'train_query': train_query,
        'train_answer': train_answer,
        'train_pooled': train_pooled,
        'test_query': test_query,
        'test_answer': test_answer,
        'test_pooled': test_pooled,
        'train_labels': train_labels,
        'test_labels': test_labels
    }

    if 'targeted' in config['states_types']:
        try:
            train_targeted = np.load(train_dir / f"{train_split}_answer_targeted_states.npy")
            test_targeted = np.load(test_dir / f"{test_split}_answer_targeted_states.npy")
            result['train_targeted'] = train_targeted
            result['test_targeted'] = test_targeted
            logger.info("   加载targeted states成功")
        except:
            logger.info("   未找到targeted states")

    return result

def run_probe_experiment(dataset_name, probe_type="pooled", layer=None, pca_dim=256, 
                        epochs=100, learning_rate=0.001, weight_decay=0.01, batch_size=32, 
                        patience=10, method='logistic', mlp_hidden_dims=None, dropout=0.1, 
                        seed=1234, save_model=False):

    config = DATASET_CONFIGS[dataset_name]

    if layer is None:
        layer = config['best_layer']
        logger.info(f"使用推荐层级: {layer}")

    logger.info(f"\n实验: {config['name']}")
    logger.info(f"   类型={probe_type}, 层={layer}, PCA={pca_dim}, 方法={method}")

    data = load_data_unified(dataset_name)

    if probe_type == "query":
        X_train, X_test = data['train_query'][:, layer, :], data['test_query'][:, layer, :]
    elif probe_type == "answer":
        X_train, X_test = data['train_answer'][:, layer, :], data['test_answer'][:, layer, :]
    elif probe_type == "pooled":
        X_train, X_test = data['train_pooled'][:, layer, :], data['test_pooled'][:, layer, :]
    elif probe_type == "targeted":
        if 'train_targeted' not in data:
            raise ValueError(f"{dataset_name} 不支持targeted")
        X_train, X_test = data['train_targeted'][:, layer, :], data['test_targeted'][:, layer, :]
    else:
        raise ValueError(f"未知类型: {probe_type}")

    y_train, y_test = data['train_labels'], data['test_labels']

    X_train, X_test, pca, scaler = preprocess_data(X_train, X_test, pca_dim=pca_dim, standardize=True, seed=seed)

    probe = SimpleProbe(input_dim=X_train.shape[1], hidden_dims=mlp_hidden_dims, dropout=dropout, method=method)
    best_metrics = probe.train(X_train, y_train, X_test, y_test, epochs=epochs, 
                               batch_size=batch_size, lr=learning_rate, weight_decay=weight_decay, patience=patience)

    logger.info(f"完成: Acc={best_metrics['test_accuracy']:.4f}, AUC={best_metrics['test_auc']:.4f}")

    if save_model:

        dataset_type = dataset_name.rsplit('_', 1)[0]                            
        model_version = '3.1' if 'llama31' in dataset_name else '3'
        save_dir = f"lcd_outputs/{dataset_type}/{model_version}_{probe_type}_layer{layer}_pca{pca_dim}_{method}"

        config_data = {
            'input_dim': X_train.shape[1],
            'hidden_dims': mlp_hidden_dims,
            'dropout': dropout,
            'dataset': dataset_name,
            'probe_type': probe_type,
            'layer': layer,
            'pca_dim': pca_dim,
            'best_accuracy': best_metrics['test_accuracy'],
            'best_auc': best_metrics['test_auc']
        }
        probe.save_model(save_dir, pca=pca, scaler=scaler, config=config_data)

        metrics_file = Path(save_dir) / 'metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(best_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics已保存到: {metrics_file}")

    return {
        'dataset': dataset_name,
        'probe_type': probe_type,
        'layer': layer,
        'pca_dim': pca_dim,
        'method': method,
        'metrics': best_metrics
    }

def main():
    parser = argparse.ArgumentParser(description='统一探针训练系统')

    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()),
                       help='数据集名称')

    parser.add_argument('--probe-type', choices=['query', 'answer', 'pooled', 'targeted'], 
                       default='pooled', help='探针类型')
    parser.add_argument('--method', choices=['logistic', 'mlp'], default='logistic')
    parser.add_argument('--layer', type=int, help='层级 (默认使用推荐层)')
    parser.add_argument('--pca-dim', type=int, default=256, help='PCA维度')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--mlp-hidden-dims', nargs='+', type=int, help='MLP隐藏层维度')

    parser.add_argument('--test-layers', nargs='+', type=int, help='测试多层')
    parser.add_argument('--test-types', nargs='+', choices=['query', 'answer', 'pooled', 'targeted'])
    parser.add_argument('--test-dims', nargs='+', type=int, help='测试多PCA维度')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save-model', action='store_true', help='保存模型')

    args = parser.parse_args()

    test_layers = args.test_layers if args.test_layers else [args.layer]
    test_types = args.test_types if args.test_types else [args.probe_type]
    test_dims = args.test_dims if args.test_dims else [args.pca_dim]

    if len(test_layers) == 1 and len(test_types) == 1 and len(test_dims) == 1:
        result = run_probe_experiment(
            dataset_name=args.dataset,
            probe_type=test_types[0],
            layer=test_layers[0],
            pca_dim=test_dims[0],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            patience=args.patience,
            method=args.method,
            mlp_hidden_dims=args.mlp_hidden_dims,
            dropout=args.dropout,
            seed=args.seed,
            save_model=args.save_model
        )
        logger.info("训练完成")

    else:
        logger.info("批量实验")
        logger.info(f"   层级: {test_layers}")
        logger.info(f"   类型: {test_types}")
        logger.info(f"   维度: {test_dims}")

        results = []
        total = len(test_layers) * len(test_types) * len(test_dims)
        current = 0

        for probe_type in test_types:
            for layer in test_layers:
                for pca_dim in test_dims:
                    current += 1
                    logger.info(f"\n实验 {current}/{total}")

                    try:
                        result = run_probe_experiment(
                            dataset_name=args.dataset,
                            probe_type=probe_type,
                            layer=layer,
                            pca_dim=pca_dim,
                            epochs=args.epochs,
                            learning_rate=args.learning_rate,
                            batch_size=args.batch_size,
                            patience=args.patience,
                            method=args.method,
                            mlp_hidden_dims=args.mlp_hidden_dims,
                            dropout=args.dropout,
                            seed=args.seed,
                            save_model=args.save_model
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"实验失败: {e}")
                        results.append({
                            'dataset': args.dataset,
                            'probe_type': probe_type,
                            'layer': layer,
                            'pca_dim': pca_dim,
                            'error': str(e)
                        })

        print_results_summary(results)

if __name__ == "__main__":
    main()
