import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout,
    QProgressBar, QMessageBox, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import torch
import torchvision.transforms as transforms
from models import get_modified_resnet18
import os

# 1. 读取类别顺序
def load_class_names():
    with open('class_names.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

# 2. 编号到中文名的映射（顺序必须和官方一致）
ID2CN = [
    '水仙花',   # 0
    '雪花莲',   # 1
    '雏菊',     # 2
    '款冬',     # 3
    '蒲公英',   # 4
    '黄花九轮草',# 5
    '毛茛',     # 6
    '银莲花',   # 7
    '三色堇',    # 8
    '铃兰',     # 9
    '风铃草',   # 10
    '番红花',   # 11
    '鸢尾花',   # 12
    '虎皮百合', # 13
    '郁金香',   # 14
    '棋盘花',   # 15
    '向日葵'   # 16
]

# 3. 得到类别顺序
CLASS_NAMES = load_class_names()  # 例如 ['0', '1', ..., '16']
# 4. 得到中文名顺序
CLASS_NAMES_CN = [ID2CN[int(i)] for i in CLASS_NAMES]

def predict(img, model, device, class_names_cn):
    # 推理时的transform与验证/测试一致（去掉Normalize）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    try:
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            print('Raw logits:', output.cpu().numpy())
            prob = torch.softmax(output, dim=1)
            print('Softmax:', prob.cpu().numpy())
            pred = torch.argmax(prob, dim=1).item()
            print('Predicted index:', pred)
            print('Predicted class name:', class_names_cn[pred])
            confidence = prob[0, pred].item()
        return class_names_cn[pred], confidence
    except Exception as e:
        return None, str(e)

class FlowerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("花卉图片识别系统")
        self.resize(900, 600)
        # 主布局：左右分栏
        main_layout = QHBoxLayout()
        # 左侧：图片展示区
        left_frame = QFrame()
        left_layout = QVBoxLayout()
        self.image_label = QLabel("请上传图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 2px dashed #bbb; background: #fafafa;")
        left_layout.addWidget(self.image_label)
        left_frame.setLayout(left_layout)
        # 右侧：按钮和结果区
        right_frame = QFrame()
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        # 按钮区
        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("选择图片")
        self.btn_select.setStyleSheet("padding: 8px 24px; font-size: 15px;")
        self.btn_select.clicked.connect(self.open_image)
        self.btn_recognize = QPushButton("识别")
        self.btn_recognize.setStyleSheet("padding: 8px 24px; font-size: 15px;")
        self.btn_recognize.clicked.connect(self.recognize_image)
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_recognize)
        # 结果区
        self.result_label = QLabel("识别结果：")
        self.result_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.conf_label = QLabel("")
        self.conf_label.setFont(QFont("微软雅黑", 13))
        self.conf_label.setAlignment(Qt.AlignCenter)
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        # 组装右侧
        right_layout.addStretch(1)
        right_layout.addLayout(btn_layout)
        right_layout.addSpacing(30)
        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.conf_label)
        right_layout.addWidget(self.progress_bar)
        right_layout.addStretch(2)
        right_frame.setLayout(right_layout)
        # 主布局组装
        main_layout.addWidget(left_frame, 2)
        main_layout.addWidget(right_frame, 1)
        self.setLayout(main_layout)
        # 变量
        self.current_img = None
        # 加载模型
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = get_modified_resnet18().to(self.device)
            self.model.load_state_dict(torch.load('best_model.pth', map_location=self.device))
            self.model.eval()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            sys.exit(1)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            img = Image.open(file_path).convert('RGB')
            self.current_img = img
            # 显示图片
            qimg = QImage(file_path)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            self.result_label.setText("识别结果：")
            self.conf_label.setText("")

    def recognize_image(self):
        if self.current_img is None:
            QMessageBox.warning(self, "提示", "请先选择图片！")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        pred_class, confidence = predict(self.current_img, self.model, self.device, CLASS_NAMES_CN)
        self.progress_bar.setValue(80)
        if pred_class is None:
            self.result_label.setText(f"识别失败")
            self.conf_label.setText("")
        else:
            self.result_label.setText(f"识别结果：{pred_class}")
            self.conf_label.setText(f"置信度：{confidence:.2%}")
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FlowerApp()
    win.show()
    sys.exit(app.exec_())