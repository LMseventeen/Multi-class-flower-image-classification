import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from datasets import get_data_loaders
from models import get_modified_resnet18
from utils import accuracy, get_param_groups
from tqdm import tqdm
from visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_feature_importance,
    plot_true_analysis,
    plot_learning_rate_history,
    plot_gradient_flow
)
import os

def evaluate_and_visualize(model, val_loader, class_names, save_dir='visualization_results'):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    all_probs = []
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu().numpy())
            all_probs.extend(probs.max(dim=1)[0].cpu().numpy())
    val_acc = val_correct / val_total
    print(f'Best Model Val Loss: {val_loss/len(val_loader.dataset):.4f}, Val Acc: {val_acc:.4f}')
    plot_confusion_matrix(all_labels, all_preds, class_names, save_dir)
    plot_feature_importance(model, class_names, save_dir)
    plot_true_analysis(all_labels, all_probs, all_images, class_names, save_dir)
    plot_gradient_flow(model, save_dir)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = './data'
    train_loader, val_loader, test_loader = get_data_loaders(data_dir)
    class_names = train_loader.dataset.dataset.classes if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.classes
    plot_class_distribution([label for _, labels in train_loader for label in labels.numpy()], class_names)
    best_model_path = 'best_model.pth'
    if os.path.exists(best_model_path):
        print('Best model already exists. Loading and generating visualizations...')
        model = get_modified_resnet18().to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        evaluate_and_visualize(model, val_loader, class_names)
        # 保存类别顺序到 class_names.txt，按数字排序
        with open('class_names.txt', 'w', encoding='utf-8') as f:
            for name in sorted(class_names, key=lambda x: int(x)):
                f.write(name + '\n')
        print('类别顺序已保存到 class_names.txt')
        return
    # 否则正常训练
    model = get_modified_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(get_param_groups(model), weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    train_losses, val_losses, train_accs, val_accs, lr_history = [], [], [], [], []
    best_acc = 0.0
    for epoch in range(15):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/15'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        lr_history.append(optimizer.param_groups[0]['lr'])
        print(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        scheduler.step()
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_learning_rate_history(lr_history)
    # 用最佳模型做可视化
    print('Training finished! Best Val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    evaluate_and_visualize(model, val_loader, class_names)
    # 保存类别顺序到 class_names.txt，按数字排序
    with open('class_names.txt', 'w', encoding='utf-8') as f:
        for name in sorted(class_names, key=lambda x: int(x)):
            f.write(name + '\n')
    print('类别顺序已保存到 class_names.txt')

if __name__ == '__main__':
    main()