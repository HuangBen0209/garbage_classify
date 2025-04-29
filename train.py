import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split
from torchvision import models, transforms
from torch.utils.data import DataLoader
from data_loader import ImageFolderWithTxt
from models import ResNet18,ResNet34, ResNet50,CNN_MultiScale
import matplotlib.pyplot as plt
import os

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / len(dataloader)
    acc = 100 * correct / total
    return loss, acc


def train_model(model, model_name, train_loader, val_loader, test_loader, num_classes, device, num_epochs=100, learning_rate=0.001):
    log_file = f"{model_name}_train_log.txt"
    model_save_path = f"{model_name}_best_model.pth"
    curve_save_path = f"{model_name}_train_curve.png"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0

    early_stopping_patience = 5  # 如果验证集精度超过 patience 个 epoch 没提升就停止
    patience_counter = 0


    with open(log_file, "w") as f:
        f.write("Epoch\tTrainLoss\tTrainAcc\tValLoss\tValAcc\n")

    print(f"开始在 {device} 上训练：")
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        with tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix(loss=running_loss / (total // train_loader.batch_size + 1), accuracy=100 * correct / total)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        with open(log_file, "a") as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_acc:.2f}\t{val_loss:.4f}\t{val_acc:.2f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved with val acc {best_val_acc:.2f}%")
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            print(
                f"⚠️  No improvement in val acc. Early stopping counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print("⏹️ Early stopping triggered.")
                break

        scheduler.step()

    # 测试阶段
    model.load_state_dict(torch.load(model_save_path))  # 加载最优模型
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[{model_name}] Test Accuracy: {test_acc:.2f}%")
    with open(log_file, "a") as f:
        f.write(f"最终测试准确率: {test_acc:.2f}%\n")

    # 可视化训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Val Loss", color='red')
    plt.plot(train_accuracies, label="Train Acc", color='green')
    plt.plot(val_accuracies, label="Val Acc", color='orange')
    plt.title(f"{model_name} - Training & Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(curve_save_path)
    print(f"[{model_name}] Training curve saved to {curve_save_path}")
    print(f"[{model_name}] Training complete!\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数设置
    data_folder = "./train_data_v2"
    batch_size = 64
    num_epochs = 25
    learning_rate = 0.0001

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.56719673, 0.5293289, 0.48351972],
                             std=[0.20874391, 0.21455203, 0.22451781]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.56719673, 0.5293289, 0.48351972],
                             std=[0.20874391, 0.21455203, 0.22451781]),
    ])

    # 加载数据
    full_dataset = ImageFolderWithTxt(data_folder, transform=None)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = full_dataset.num_classes


    # # #  第一次次训练：自定义的垃圾模型，别浪费时间跑了
    # model_cnn_multiscale = CNN_MultiScale(num_classes=num_classes).to(device)
    # train_model(model_cnn_multiscale, "model_cnn_multiscale", train_loader, val_loader, test_loader, num_classes, device, num_epochs,learning_rate)
    #
    # # #  第一次训练：ResNet18  效果还不错
    # model_resnet18 = ResNet34(num_classes=num_classes).to(device)
    # train_model(model_resnet18, "ResNet18", train_loader, val_loader, test_loader, num_classes, device, num_epochs,learning_rate)
    #
    # # # 第三次训练：ResNet50
    # model_resnet50 = ResNet50(num_classes=num_classes).to(device)
    # train_model(model_resnet50, "ResNet50", train_loader, val_loader, test_loader, num_classes, device, num_epochs,learning_rate)

    # 第四次训练：ResNet34

    model_resnet34 = ResNet34(num_classes=num_classes).to(device)
    train_model(model_resnet34, "ResNet34", train_loader, val_loader, test_loader, num_classes, device, num_epochs, learning_rate)


if __name__ == "__main__":
    main()
