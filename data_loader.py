import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class ImageFolderWithTxt(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        :param folder_path: 数据集所在的文件夹路径
        :param transform: 对图像进行的预处理操作（例如：Resize，Normalize等）
        """
        self.folder_path = folder_path
        self.transform = transform
        self.samples = []

        # 遍历文件夹，读取所有txt文件，找到对应的图片和标签
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                txt_path = os.path.join(folder_path, filename)
                with open(txt_path, "r") as f:
                    line = f.readline().strip()
                    image_name, label = line.split(",")  # 假设每行数据格式为 img_1.jpg, 0
                    image_path = os.path.join(folder_path, image_name.strip())
                    self.samples.append((image_path, int(label)))  # 保存图片路径和标签

        # 获取分类的数量
        self.num_classes = len(set(label for _, label in self.samples))  # 根据标签获取分类数量

    def __len__(self):
        """
        返回数据集中的样本数
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的图片和标签
        """
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")  # 确保是RGB格式

        if self.transform:
            image = self.transform(image)  # 应用预处理操作

        return image, label

# 用法示例
if __name__ == "__main__":
    data_folder = r"D:\Desktop\train_data_v2\train_data_v2"  # 这里需要替换为你的数据路径

    # 定义训练集的 transform：Resize到256 -> 随机裁剪224
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.56719673, 0.5293289, 0.48351972],
                             std = [0.20874391, 0.21455203, 0.22451781]),
    ])

    # 测试集的 transform：Resize到256 -> 中心裁剪224
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.56719673, 0.5293289, 0.48351972],
                             std=[0.20874391, 0.21455203, 0.22451781]),
    ])

    # 加载整个数据集
    full_dataset = ImageFolderWithTxt(data_folder, transform=None)

    # 划分训练集和测试集：80%训练，20%测试
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 给训练集和测试集分别应用 transform
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"分类数量: {full_dataset.num_classes}")
