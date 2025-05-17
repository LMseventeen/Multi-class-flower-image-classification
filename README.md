## 项目简介
本项目基于PyTorch，使用迁移学习方法（ResNet18）对Oxford Flowers17数据集进行多类别花卉分类，并实现Grad-CAM注意力可视化。

## 项目需求

本项目旨在基于PyTorch框架，利用迁移学习方法，完成对**Oxford Flowers17**数据集的多类别花卉图像分类任务。项目的主要目标和步骤如下：

### 1. 实践迁移学习与微调
- 使用**预训练的ResNet18模型**作为特征提取器。
- 冻结除最后一个block（layer4）外的所有卷积层，仅微调最后一个block和全连接层，以适应新任务。

### 2. 数据准备与增强
- 下载并解压Flowers17数据集。
- 按6:2:2比例划分为训练集、验证集和测试集。
- 对训练集进行数据增强（如随机裁剪、水平翻转），提升模型泛化能力。

### 3. 模型结构修改
- 替换ResNet18的最后全连接层，使其输出类别数为17。
- 在全局平均池化层后添加Dropout层，防止过拟合。

### 4. 训练与优化
- 使用交叉熵损失函数。
- 对不同部分的参数（如新加的全连接层和微调的block）设置不同的学习率。
- 实现训练循环，并在每个epoch后在验证集上评估性能，使用学习率调度器动态调整学习率。

### 5. 测试与评估
- 在测试集上评估最终模型，计算整体准确率和每个类别的准确率。

### 6. 可解释性与可视化
- 实现**注意力可视化**（如Grad-CAM），展示模型在分类时关注的图像区域，提升模型的可解释性。
- 提供丰富的训练过程可视化，帮助理解模型行为。

# Flowers17 多类别花卉分类（迁移学习+可解释性）

## 主要功能
- 数据集下载与增强
- 迁移学习与微调
- 多类别分类训练与评估
- 注意力可视化（Grad-CAM）
- 训练过程可视化与分析

## 可视化功能
项目提供了丰富的可视化功能，帮助理解模型训练过程和性能：

1. **训练过程可视化**
   - 训练和验证损失曲线
   - 训练和验证准确率曲线
   - 学习率变化历史

2. **模型性能分析图表**
   - 混淆矩阵（confusion_matrix.png）：展示模型在各个类别上的预测情况
   - 特征重要性分析（feature_importance.png）：展示最后一层全连接层的权重
   - 梯度流分析（gradient_flow.png）：展示模型各层的梯度范数

3. **数据分布与特征分布可视化**
   - 类别分布图（class_distribution.png）：展示数据集中各类别的样本数量
   - 预测分布柱状图（feature_distribution.png）：展示每个类别被预测为该类的样本数量分布
   - 特征分布t-SNE可视化（feature_distribution_tsne.png）：用t-SNE将模型高维特征降到2维，每个点代表一张图片，颜色代表类别，直观展示特征空间聚类情况

4. **预测结果分析图表**
   - 正确预测分析（true_analysis.png）：每个类别展示1个置信度最高的正确预测样本（如有），每行5个小图，自动换行，标题只显示类别名称和置信度，无坐标轴

所有可视化图片均保存在`visualization_results`目录下。
![eb70b2b8a88c9a785f069f994e5f634](https://github.com/user-attachments/assets/97545e8b-8324-4b84-9115-8895cdec4a66)
![3ae27dbf4b028c62f3724024cd3442a](https://github.com/user-attachments/assets/88972918-cc56-478d-945e-eeb6c02bbcf5)
![2f048586cbe415ccf21df9503bbd678](https://github.com/user-attachments/assets/e3191405-3788-429b-abbd-0a7a7d6b97b5)
![f2faf63a3220d231fa94de250499a6f](https://github.com/user-attachments/assets/4221415f-ba61-4613-b1c9-b4c9cb91532b)
![38e147425e40f38cbc21713c24c7b87](https://github.com/user-attachments/assets/75f85c93-e8ef-49ef-b53c-698c0e94d04d)

<img width="696" alt="259d24063edb29a88e9e11cac6ab065" src="https://github.com/user-attachments/assets/5c7f7449-fd0b-4af3-a5d0-b3217a6773d2" />




如需自定义每类展示的样本数，可在`visualization.py`中调整`samples_per_class`参数。


## 运行步骤
1. 安装依赖：`pip install -r requirements.txt`
2. 数据准备：运行`datasets.py`下载和划分数据
3. 训练模型：`python train.py`
4. 测试评估：`python evaluate.py`
5. 可视化注意力：`python gradcam.py`

## 桌面应用：PyQt5 图形界面（flows_app.py）

本项目支持通过 PyQt5 桌面应用进行花卉图片识别，界面友好，操作简单。

<img width="450" alt="28b2dd44acd8f8f03b280d62235733b" src="https://github.com/user-attachments/assets/7dd1bb43-0bd5-4fe5-bdfd-3cfdafd2a3e2" />




### 使用方法

1. 安装依赖：
   ```bash
   pip install pyqt5 pillow torch torchvision
   ```
2. 确保 `best_model.pth` 在项目根目录。
3. 运行应用：
   ```bash
   python flower_app.py
   ```
4. 在弹出的窗口中点击"选择图片"，选择一张花卉图片，界面会显示图片和预测结果。

### 常见问题简化说明
- 若模型未找到，请确保 `best_model.pth` 在项目根目录。
- 若图片无法识别，请检查图片格式是否为 jpg、jpeg 或 png。
- 若遇到 PyQt5 相关报错，请确认依赖已正确安装。


## GPU 支持说明

本项目支持 GPU 训练，可以显著提升训练速度。要启用 GPU 支持，请确保：

1. 系统要求：
   - NVIDIA GPU
   - 已安装 NVIDIA 显卡驱动
   - 已安装 CUDA Toolkit（建议 CUDA 11.8 或更新版本）

2. PyTorch 安装：
   - 如果当前安装的是 CPU 版本，需要重新安装支持 CUDA 的版本：
   ```bash
   # 1. 首先卸载现有版本
   pip uninstall torch torchvision

   # 2. 清理 pip 缓存
   pip cache purge

   # 3. 使用清华镜像源安装支持 CUDA 的版本（推荐）
   pip3 install torch torchvision --user -i https://pypi.tuna.tsinghua.edu.cn/simple --index-url https://download.pytorch.org/whl/cu118

   # 或者使用阿里云镜像源
   pip3 install torch torchvision --user -i https://mirrors.aliyun.com/pypi/simple/ --index-url https://download.pytorch.org/whl/cu118

   # 或者使用中国科技大学镜像源
   pip3 install torch torchvision --user -i https://pypi.mirrors.ustc.edu.cn/simple/ --index-url https://download.pytorch.org/whl/cu118
   ```

   安装完成后，可以通过以下代码验证 GPU 支持：
   ```python
   import torch
   print('CUDA是否可用:', torch.cuda.is_available())
   print('GPU数量:', torch.cuda.device_count())
   if torch.cuda.is_available():
       print('GPU名称:', torch.cuda.get_device_name(0))
   ```

   如果遇到安装问题，可以尝试以下解决方案：
   - 下载超时：
     ```bash
     # 使用国内镜像源
     pip3 install torch torchvision --user -i https://pypi.tuna.tsinghua.edu.cn/simple --index-url https://download.pytorch.org/whl/cu118
     ```
   - 文件被占用：
     ```bash
     # 1. 关闭所有 Python 相关进程
     # 2. 删除临时文件
     # PowerShell:
     Remove-Item -Path "$env:TEMP\pip-unpack-*" -Force
     # 或者 CMD:
     # del /f /q "%TEMP%\pip-unpack-*"
     # 3. 使用 --user 选项重新安装
     pip3 install torch torchvision --user -i https://pypi.tuna.tsinghua.edu.cn/simple --index-url https://download.pytorch.org/whl/cu118
     ```
   - 权限问题：
     ```bash
     # 以管理员身份运行命令提示符
     # 或使用 --user 选项
     pip3 install torch torchvision --user -i https://pypi.tuna.tsinghua.edu.cn/simple --index-url https://download.pytorch.org/whl/cu118
     ```

3. 验证 GPU 是否可用：
   ```python
   import torch
   print('CUDA是否可用:', torch.cuda.is_available())
   print('GPU数量:', torch.cuda.device_count())
   ```

4. 常见问题：
   - 如果 `torch.cuda.is_available()` 返回 `False`：
     - 检查是否安装了 NVIDIA 显卡驱动
     - 检查是否安装了 CUDA Toolkit
     - 确认 PyTorch 安装的是 CUDA 版本
   - 如果出现 CUDA 相关错误：
     - 确保 CUDA 版本与 PyTorch 版本兼容
     - 检查 GPU 显存是否足够
     - 检查系统环境变量是否正确设置
   - 如果安装过程中遇到问题：
     - 确保关闭所有 Python 相关进程
     - 尝试使用管理员权限运行命令
     - 如果使用虚拟环境，确保激活了正确的环境
     - 可以尝试先卸载再重新安装：
       ```bash
       pip uninstall torch torchvision
       pip cache purge
       pip3 install torch torchvision --user --index-url https://download.pytorch.org/whl/cu118
       ```


