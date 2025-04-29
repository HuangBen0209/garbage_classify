import sys
import torch
import torchvision.transforms as transforms
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QFileDialog, QMessageBox, QHBoxLayout, QStatusBar)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt
from PIL import Image
from models import ResNet34  # 你的模型类

# 读取类别映射
def load_class_labels(file_path="mapping.txt"):
    class_labels = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label_id, label_name = line.strip().split(": ")
            class_labels[int(label_id)] = label_name
    return class_labels

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.56719673, 0.5293289, 0.48351972],
                         std=[0.20874391, 0.21455203, 0.22451781]),
])
# 加载模型
def load_model(model_path="ResNet34_best_model.pth", num_classes=40):
    model = ResNet34(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"),weights_only=True))
    model.eval()
    return model

# 预测函数
def predict_image(image_path, model, class_labels):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()
    return class_labels.get(predicted_class, "未知类别")

# 主窗口类
class ClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("垃圾图片分类器")
        self.setMinimumSize(500, 600)  # 增大最小窗口尺寸
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)

        self.model = load_model()  # 加载模型
        self.class_labels = load_class_labels()  # 加载类别映射

        # 图像显示区域
        self.image_label = QLabel("请选择一张图片")
        self.image_label.setStyleSheet("border: 2px solid #3f3f3f; border-radius: 10px; background-color: #f4f4f4;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)

        # 结果显示区域
        self.result_label = QLabel("预测结果：")
        self.result_label.setStyleSheet("font-size: 18px; color: #2E8B57; font-weight: bold;")

        # 上传按钮
        self.upload_button = QPushButton("上传图片")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                padding: 12px 30px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.upload_button.clicked.connect(self.upload_image)

        # 状态栏显示
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("准备上传图片", 3000)

        # 布局优化
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_button)
        button_layout.setAlignment(Qt.AlignCenter)

        # 整体布局
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.addWidget(self.image_label, stretch=1)  # 图像区域可自动伸缩
        layout.addWidget(self.result_label)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_bar)

        self.setLayout(layout)

    # 图片上传和预测
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            try:
                # 更新状态
                self.status_bar.showMessage("正在处理图片...", 2000)

                # 显示图像
                pixmap = QPixmap(file_name).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)

                # 模型预测
                result = predict_image(file_name, self.model, self.class_labels)
                self.result_label.setText(f"预测结果：{result}")

                # 更新状态
                self.status_bar.showMessage("预测完成", 2000)
            except Exception as e:
                self.status_bar.showMessage("出现错误", 2000)
                QMessageBox.critical(self, "错误", f"处理图片出错：\n{str(e)}")

# 启动程序
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClassifierApp()
    window.show()  # 窗口将默认可最大化
    sys.exit(app.exec_())
