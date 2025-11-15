#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import json
import argparse
import logging
import os
import random
import joblib
from pathlib import Path

FIXED_SEED = 1234
random.seed(FIXED_SEED)
np.random.seed(FIXED_SEED)
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleProbe:

    def __init__(self, input_dim=4096, hidden_dims=None, dropout=0.1, num_classes=2, method='logistic'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method

        if method == 'logistic':

            self.classifier = LogisticRegression(
                random_state=42, 
                max_iter=2000,
                solver='liblinear',
                C=1.0,
                class_weight='balanced',
                tol=1e-4
            )
            logger.info("Using Logistic Regression")
            self.model = None
        else:

            if hidden_dims is None:

                hidden_dim = min(1024, input_dim // 2)
                hidden_dims = [hidden_dim]
            elif isinstance(hidden_dims, int):

                hidden_dims = [hidden_dims]

            layers = []
            prev_dim = input_dim

            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, num_classes))

            self.model = nn.Sequential(*layers)
            self.model.to(self.device)
            self.classifier = None

            arch_str = f"{input_dim} -> " + " -> ".join([str(dim) for dim in hidden_dims]) + f" -> {num_classes}"
            logger.info(f"Using MLP: {arch_str}")
            logger.info(f"   Number of layers: {len(hidden_dims)} hidden layers")
            logger.info(f"   Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=0.001, weight_decay=0.01, patience=10):

        if self.method == 'logistic':

            logger.info(f"Training Logistic Regression classifier...")

            best_acc = 0.0
            best_metrics = {}
            patience_counter = 0

            for epoch in range(epochs):

                self.classifier.fit(X_train, y_train)

                train_pred = self.classifier.predict(X_train)
                test_pred = self.classifier.predict(X_test)

                train_proba = self.classifier.predict_proba(X_train)[:, 1]
                test_proba = self.classifier.predict_proba(X_test)[:, 1]

                train_acc = accuracy_score(y_train, train_pred)
                test_acc = accuracy_score(y_test, test_pred)

                from sklearn.metrics import roc_auc_score
                train_auc = roc_auc_score(y_train, train_proba)
                test_auc = roc_auc_score(y_test, test_proba)

                test_precision = precision_score(y_test, test_pred, zero_division=0)
                test_recall = recall_score(y_test, test_pred, zero_division=0)
                test_f1 = f1_score(y_test, test_pred, zero_division=0)

                tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
                test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_metrics = {
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'train_auc': train_auc,
                        'test_auc': test_auc,
                        'precision': test_precision,
                        'recall': test_recall,
                        'f1_score': test_f1,
                        'specificity': test_specificity
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch:3d}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Test AUC={test_auc:.3f}")

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}, best accuracy: {best_acc:.3f}")
                    break

            return best_metrics
        else:

            train_dataset = TensorDataset(
                torch.FloatTensor(X_train), 
                torch.LongTensor(y_train)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test), 
                torch.LongTensor(y_test)
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()

            best_acc = 0.0
            best_metrics = {}
            patience_counter = 0

            for epoch in range(epochs):

                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()

                self.model.eval()
                test_correct = 0
                test_total = 0
                train_probs = []
                test_probs = []
                train_true = []
                test_true = []
                test_pred_labels = []

                with torch.no_grad():

                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        probs = torch.softmax(outputs, dim=1)[:, 1]         
                        train_probs.extend(probs.cpu().numpy())
                        train_true.extend(batch_y.cpu().numpy())

                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        probs = torch.softmax(outputs, dim=1)[:, 1]         
                        _, predicted = outputs.max(1)
                        test_total += batch_y.size(0)
                        test_correct += predicted.eq(batch_y).sum().item()
                        test_probs.extend(probs.cpu().numpy())
                        test_true.extend(batch_y.cpu().numpy())
                        test_pred_labels.extend(predicted.cpu().numpy())

                train_acc = train_correct / train_total
                test_acc = test_correct / test_total

                from sklearn.metrics import roc_auc_score
                train_auc = roc_auc_score(train_true, train_probs)
                test_auc = roc_auc_score(test_true, test_probs)

                try:
                    test_precision = precision_score(test_true, test_pred_labels, zero_division=0)
                    test_recall = recall_score(test_true, test_pred_labels, zero_division=0)
                    test_f1 = f1_score(test_true, test_pred_labels, zero_division=0)
                    cm = confusion_matrix(test_true, test_pred_labels)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    else:
                        test_specificity = 0.0
                except Exception:
                    test_precision = test_recall = test_f1 = 0.0
                    test_specificity = 0.0

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_metrics = {
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'train_auc': train_auc,
                        'test_auc': test_auc,
                        'precision': test_precision,
                        'recall': test_recall,
                        'f1_score': test_f1,
                        'specificity': test_specificity
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch:3d}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Test AUC={test_auc:.3f}")

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}, best accuracy: {best_acc:.3f}")
                    break

            return best_metrics

    def save_model(self, save_path: str, pca=None, scaler=None, config=None):

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        model_data = {
            'method': self.method,
            'config': config or {},
            'pca': pca,
            'scaler': scaler
        }

        if self.method == 'logistic':
            model_data['classifier'] = self.classifier
        else:

            torch.save(self.model.state_dict(), save_path / 'mlp_model.pth')
            model_data['model_architecture'] = str(self.model)

        joblib.dump(model_data, save_path / 'lcd_model.pkl')
        logger.info(f"Model saved to: {save_path}")

    @classmethod
    def load_model(cls, load_path: str):

        load_path = Path(load_path)
        model_data = joblib.load(load_path / 'lcd_model.pkl')

        method = model_data['method']
        config = model_data['config']

        valid_params = ['input_dim', 'hidden_dims', 'dropout', 'num_classes', 'method']
        filtered_config = {k: v for k, v in config.items() if k in valid_params}

        probe = cls(method=method, **filtered_config)

        if method == 'logistic':
            probe.classifier = model_data['classifier']
        else:

            probe.model.load_state_dict(torch.load(load_path / 'mlp_model.pth'))
            probe.model.eval()

        logger.info(f"Model loaded from: {load_path}")
        return probe, model_data['pca'], model_data['scaler']

def load_data(data_dir="../results_triple_states_clean"):

    logger.info(f"Loading data from {data_dir}...")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    is_four_states = os.path.exists(f"{data_dir}/train_answer_targeted_states.npy")

    if is_four_states:
        logger.info("Detected four-state dataset (contains targeted states)")
        required_files = [
            "train_data_with_labels.json",
            "test_data_with_labels.json", 
            "train_query_hidden_states.npy",
            "train_answer_hidden_states.npy",
            "train_answer_pooled_states.npy",
            "train_answer_targeted_states.npy",
            "test_query_hidden_states.npy",
            "test_answer_hidden_states.npy",
            "test_answer_pooled_states.npy",
            "test_answer_targeted_states.npy"
        ]
    else:
        logger.info("Detected three-state dataset")
        required_files = [
            "train_data_with_labels.json",
            "test_data_with_labels.json", 
            "train_query_hidden_states.npy",
            "train_answer_hidden_states.npy",
            "train_answer_pooled_states.npy",
            "test_query_hidden_states.npy",
            "test_answer_hidden_states.npy",
            "test_answer_pooled_states.npy"
        ]

    for file in required_files:
        if not os.path.exists(f"{data_dir}/{file}"):
            raise FileNotFoundError(f"Data file does not exist: {data_dir}/{file}")

    with open(f"{data_dir}/train_data_with_labels.json", 'r') as f:
        train_data = json.load(f)
    with open(f"{data_dir}/test_data_with_labels.json", 'r') as f:
        test_data = json.load(f)

    train_query = np.load(f"{data_dir}/train_query_hidden_states.npy")
    train_answer = np.load(f"{data_dir}/train_answer_hidden_states.npy")
    train_pooled = np.load(f"{data_dir}/train_answer_pooled_states.npy")

    test_query = np.load(f"{data_dir}/test_query_hidden_states.npy")
    test_answer = np.load(f"{data_dir}/test_answer_hidden_states.npy")
    test_pooled = np.load(f"{data_dir}/test_answer_pooled_states.npy")

    train_labels = np.array([item['is_correct'] for item in train_data])
    test_labels = np.array([item['is_correct'] for item in test_data])

    logger.info(f"Training samples: {len(train_data)}, Testing samples: {len(test_data)}")
    logger.info(f"Data shape: {train_query.shape}")

    train_correct_rate = np.mean(train_labels)
    test_correct_rate = np.mean(test_labels)
    logger.info(f"Label distribution: Training accuracy={train_correct_rate:.3f}, Testing accuracy={test_correct_rate:.3f}")

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

    if is_four_states:
        try:
            train_targeted = np.load(f"{data_dir}/train_answer_targeted_states.npy")
            test_targeted = np.load(f"{data_dir}/test_answer_targeted_states.npy")
            result['train_targeted'] = train_targeted
            result['test_targeted'] = test_targeted
            logger.info("Successfully loaded targeted states")
        except Exception as e:
            logger.warning(f"Failed to load targeted states: {e}")

    return result

def preprocess_data(X_train, X_test, pca_dim=512, standardize=True, seed=1234):

    pca = None
    scaler = None

    if pca_dim > 0 and pca_dim < X_train.shape[1]:
        logger.info(f"PCA dimensionality reduction to {pca_dim} dimensions")
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        explained_variance = np.sum(pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance ratio: {explained_variance:.3f}")

        X_train, X_test = X_train_pca, X_test_pca

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, pca, scaler

def run_probe_experiment(probe_type="answer", layer=32, pca_dim=512, 
                        epochs=100, data_dir="../results_triple_states_clean",
                        learning_rate=0.001, weight_decay=0.01, batch_size=32, patience=10,
                        method='logistic', mlp_hidden_dims=None, dropout=0.1, seed=1234, save_model=False):

    logger.info(f"Starting experiment: {probe_type} layer-{layer} PCA-{pca_dim}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Parameters: lr={learning_rate}, wd={weight_decay}, bs={batch_size}, patience={patience}, seed={seed}")

    data = load_data(data_dir)

    if probe_type == "query":
        X_train = data['train_query'][:, layer, :]
        X_test = data['test_query'][:, layer, :]
    elif probe_type == "answer":
        X_train = data['train_answer'][:, layer, :]
        X_test = data['test_answer'][:, layer, :]
    elif probe_type == "pooled":
        X_train = data['train_pooled'][:, layer, :]
        X_test = data['test_pooled'][:, layer, :]
    elif probe_type == "targeted":
        if 'train_targeted' not in data:
            raise ValueError("Current dataset does not support targeted states, please use a four-state dataset")
        X_train = data['train_targeted'][:, layer, :]
        X_test = data['test_targeted'][:, layer, :]
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    y_train = data['train_labels']
    y_test = data['test_labels']

    X_train, X_test, pca, scaler = preprocess_data(X_train, X_test, pca_dim=pca_dim, seed=seed)

    probe = SimpleProbe(input_dim=X_train.shape[1], hidden_dims=mlp_hidden_dims, dropout=dropout, method=method)

    best_metrics = probe.train(X_train, y_train, X_test, y_test, epochs=epochs, 
                          batch_size=batch_size, lr=learning_rate, weight_decay=weight_decay, patience=patience)

    logger.info(f"Experiment completed: {probe_type} layer-{layer} -> Best accuracy: {best_metrics['test_accuracy']:.3f}")

    if save_model:
        save_dir = f"models/{probe_type}_layer{layer}_pca{pca_dim}_{method}"
        config = {
            'input_dim': X_train.shape[1],
            'hidden_dims': mlp_hidden_dims,
            'dropout': dropout,
            'probe_type': probe_type,
            'layer': layer,
            'pca_dim': pca_dim,
            'best_accuracy': best_metrics['test_accuracy']
        }
        probe.save_model(save_dir, pca=pca, scaler=scaler, config=config)
        logger.info(f"Model saved to: {save_dir}")
    else:
        logger.info("Model not saved (use --save-model parameter to enable saving)")

    result = {
        'probe_type': probe_type,
        'layer': layer,
        'pca_dim': pca_dim,
        'method': method,
        'metrics': best_metrics,
        'config': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'patience': patience,
            'mlp_hidden_dims': mlp_hidden_dims,
            'dropout': dropout
        }
    }

    return result

def print_results_summary(results):

    print("\n" + "="*80)
    print("Probe Experiment Results Summary")
    print("="*80)

    valid_results = [r for r in results if 'error' not in r]

    if not valid_results:
        print("No valid experiment results")
        return

    by_type = {}
    for result in valid_results:
        probe_type = result['probe_type']
        if probe_type not in by_type:
            by_type[probe_type] = []
        by_type[probe_type].append(result)

    for probe_type in sorted(by_type.keys()):
        type_results = by_type[probe_type]
        print(f"\n{probe_type.upper()} Probe Results:")
        print("-" * 70)
        print(f"{'Layer':<8} {'PCA':<8} {'Train ACC':<12} {'Test ACC':<12} {'Test AUC':<12}")
        print("-" * 70)

        type_results.sort(key=lambda x: (x['layer'], x['pca_dim']))

        for result in type_results:
            layer = result['layer']
            pca_dim = result['pca_dim']
            train_acc = result['metrics']['train_accuracy']
            test_acc = result['metrics']['test_accuracy']
            test_auc = result['metrics']['test_auc']

            print(f"{layer:<8} {pca_dim:<8} {train_acc:<12.4f} {test_acc:<12.4f} {test_auc:<12.4f}")

    print(f"\nBest Performance Ranking (by accuracy):")
    print("-" * 70)

    sorted_by_acc = sorted(valid_results, key=lambda x: x['metrics']['test_accuracy'], reverse=True)
    print(f"{'Rank':<6} {'Configuration':<35} {'Test ACC':<12} {'Test AUC':<12}")
    print("-" * 70)

    for i, result in enumerate(sorted_by_acc[:10]):          
        config = f"{result['probe_type']}_layer{result['layer']}_pca{result['pca_dim']}"
        test_acc = result['metrics']['test_accuracy']
        test_auc = result['metrics']['test_auc']
        print(f"{i+1:<6} {config:<35} {test_acc:<12.4f} {test_auc:<12.4f}")

    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Simplified Probe Training System')

    parser.add_argument('--probe-type', choices=['answer', 'query', 'pooled', 'targeted'], 
                       default='answer', help='Probe type')
    parser.add_argument('--method', choices=['logistic', 'mlp'], default='logistic', 
                       help='Classification method: logistic=Logistic Regression, mlp=Neural Network')
    parser.add_argument('--dataset', choices=['llama3', 'llama3_1'], 
                       help='Dataset quick selection')

    parser.add_argument('--layer', type=int, default=32, help='Target layer (0-32)')
    parser.add_argument('--pca-dim', type=int, default=512, help='PCA dimension (0 means no dimensionality reduction)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (Logistic Regression and MLP)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (only for MLP)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (only for MLP)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (Logistic Regression and MLP)')

    parser.add_argument('--mlp-hidden-dims', nargs='+', type=int, 
                       help='MLP hidden layer dimensions list, e.g. --mlp-hidden-dims 1024 512 256 (multiple layers)')
    parser.add_argument('--mlp-hidden-dim', type=int, 
                       help='MLP single hidden layer dimension (backward compatible, equivalent to --mlp-hidden-dims single value)')
    parser.add_argument('--dropout', type=float, default=0.1, help='MLP dropout rate')

    parser.add_argument('--test-layers', nargs='+', type=int, 
                       help='Test multiple layers, e.g. --test-layers 0 8 16 24 32')
    parser.add_argument('--test-types', nargs='+', choices=['answer', 'query', 'pooled', 'targeted'],
                       help='Test multiple types, e.g. --test-types answer query pooled targeted')
    parser.add_argument('--test-dims', nargs='+', type=int,
                       help='Test multiple PCA dimensions, e.g. --test-dims 2 4 64 256 512')

    parser.add_argument('--data-dir', help='Data directory path')

    parser.add_argument('--seed', type=int, default=1234, help='Random seed (default 1234)')

    parser.add_argument('--save-model', action='store_true', help='Save trained model (default not saved)')

    args = parser.parse_args()

    if args.seed != FIXED_SEED:
        logger.info(f"Setting random seed: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        logger.info(f"Using default random seed: {FIXED_SEED}")

    if args.dataset:
        if args.dataset == 'llama3':
            args.data_dir = 'methods/gsm8k_decoder/results_3_four_states'
        elif args.dataset == 'llama3_1':
            args.data_dir = 'methods/gsm8k_decoder/results_four_states'

    if not args.data_dir:
        args.data_dir = 'results_3_four_states'

    test_layers = args.test_layers if args.test_layers else [args.layer]
    test_types = args.test_types if args.test_types else [args.probe_type]
    test_dims = args.test_dims if args.test_dims else [args.pca_dim]

    mlp_hidden_dims = None
    if args.mlp_hidden_dims:
        mlp_hidden_dims = args.mlp_hidden_dims
    elif args.mlp_hidden_dim:
        mlp_hidden_dims = [args.mlp_hidden_dim]

    if len(test_layers) == 1 and len(test_types) == 1 and len(test_dims) == 1:
        result = run_probe_experiment(
            probe_type=test_types[0],
            layer=test_layers[0],
            pca_dim=test_dims[0],
            epochs=args.epochs,
            data_dir=args.data_dir,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            patience=args.patience,
            method=args.method,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=args.dropout,
            seed=args.seed,
            save_model=args.save_model
        )

        logger.info("GSM8K探针训练完成!")
        logger.info(f"最终结果: 测试准确率 = {result['metrics']['test_accuracy']:.4f}, 测试AUC = {result['metrics']['test_auc']:.4f}")

    else:

        test_desc = []
        if len(test_layers) > 1:
            test_desc.append(f"层级={test_layers}")
        if len(test_types) > 1:
            test_desc.append(f"类型={test_types}")
        if len(test_dims) > 1:
            test_desc.append(f"维度={test_dims}")

        logger.info(f"开始批量GSM8K探针实验")
        logger.info(f"   测试层级: {test_layers}")
        logger.info(f"   测试类型: {test_types}")
        logger.info(f"   测试维度: {test_dims}")
        logger.info(f"   方法: {args.method}")

        results = []
        total_experiments = len(test_layers) * len(test_types) * len(test_dims)
        current_exp = 0

        for probe_type in test_types:
            for layer in test_layers:
                for pca_dim in test_dims:
                    current_exp += 1
                    logger.info(f"\n实验 {current_exp}/{total_experiments}")

                    try:
                        result = run_probe_experiment(
                            probe_type=probe_type,
                            layer=layer,
                            pca_dim=pca_dim,
                            epochs=args.epochs,
                            data_dir=args.data_dir,
                            learning_rate=args.learning_rate,
                            batch_size=args.batch_size,
                            patience=args.patience,
                            method=args.method,
                            mlp_hidden_dims=mlp_hidden_dims,
                            dropout=args.dropout,
                            seed=args.seed,
                            save_model=args.save_model
                        )
                        results.append(result)

                    except Exception as e:
                        logger.error(f"实验失败: {e}")
                        results.append({
                            'probe_type': probe_type,
                            'layer': layer,
                            'pca_dim': pca_dim,
                            'method': args.method,
                            'error': str(e)
                        })

        print_results_summary(results)

if __name__ == "__main__":
    main() 