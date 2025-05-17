import torch
from datasets import get_data_loaders
from models import get_modified_resnet18


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = './data'
    _, _, test_loader = get_data_loaders(data_dir)
    model = get_modified_resnet18().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 17
    class_total = [0] * 17
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i] == label:
                    class_correct[label] += 1
    print(f'Overall accuracy: {correct / total:.4f}')
    for i in range(17):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'Class {i} accuracy: {acc:.4f}')

if __name__ == '__main__':
    evaluate() 