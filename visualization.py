import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
from sklearn.manifold import TSNE

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir='visualization_results'):
    """
    绘制训练过程中的损失和准确率变化
    """
    ensure_dir(save_dir)
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir='visualization_results'):
    """
    绘制混淆矩阵
    """
    ensure_dir(save_dir)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_class_distribution(labels, class_names, save_dir='visualization_results'):
    """
    绘制类别分布
    """
    ensure_dir(save_dir)
    plt.figure(figsize=(12, 6))
    sns.countplot(x=labels)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
    plt.close()

def plot_feature_importance(model, class_names, save_dir='visualization_results'):
    """
    分析最后一层全连接层的权重，展示特征重要性
    """
    ensure_dir(save_dir)
    weights = model.fc[1].weight.data.cpu().numpy()
    # 只显示前20个特征（如果特征数大于20）
    num_features = min(20, weights.shape[1])
    plt.figure(figsize=(14, 6))
    sns.heatmap(weights[:, :num_features], cmap='coolwarm', center=0,
                xticklabels=[f'F{i}' for i in range(num_features)],
                yticklabels=class_names)
    plt.title('Feature Importance Analysis (Top 20 Features)')
    plt.xlabel('Feature')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.close()

def plot_true_analysis(y_true, probs, images, class_names, save_dir='visualization_results', samples_per_class=1):
    ensure_dir(save_dir)
    y_true = np.array(y_true)
    probs = np.array(probs)
    images = np.array(images)
    n_classes = len(class_names)
    selected_indices = []
    for class_idx in range(n_classes):
        class_corrects = np.where(y_true == class_idx)[0]
        if len(class_corrects) > 0:
            topk = np.argsort(-probs[class_corrects])[:samples_per_class]
            chosen = class_corrects[topk]
            selected_indices.extend(chosen)
        else:
            selected_indices.extend([-1]*samples_per_class)
    n_samples = n_classes * samples_per_class
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    plt.figure(figsize=(3*n_cols, 3*n_rows))
    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i+1)
        idx = selected_indices[i]
        plt.xticks([])
        plt.yticks([])
        if idx == -1:
            plt.axis('off')
            plt.title(f"无正确\n{class_names[i % n_classes]}", fontsize=10)
        else:
            img = images[idx].transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.title(f"{class_names[y_true[idx]]}\nConf: {probs[idx]:.2f}", fontsize=10)
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'true_analysis.png'))
    plt.close()

def plot_learning_rate_history(lr_history, save_dir='visualization_results'):
    """
    绘制学习率变化历史
    """
    ensure_dir(save_dir)
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history)
    plt.title('Learning Rate History')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'learning_rate_history.png'))
    plt.close()

def plot_gradient_flow(model, save_dir='visualization_results'):
    """
    分析模型各层的梯度流
    """
    ensure_dir(save_dir)
    gradients = []
    layers = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append(param.grad.norm().item())
            layers.append(name)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(gradients)), gradients)
    plt.xticks(range(len(layers)), layers, rotation=45, fontsize=8)
    plt.title('Gradient Flow Analysis')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradient_flow.png'))
    plt.close()

def plot_feature_tsne(features, labels, class_names, save_dir='visualization_results'):
    ensure_dir(save_dir)
    features = np.array(features)
    labels = np.array(labels)
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        idx = labels == i
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], s=15, label=class_name, alpha=0.7)
    plt.legend()
    plt.title('Feature Distribution (t-SNE)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_distribution_tsne.png'))
    plt.close() 