import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import hdf5storage
from sklearn.metrics import accuracy_score
import torch.optim as optim
from metrics import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import GRUModel, CNNWithCAFM, CNN, CNNWithCAFM256, CNNWithCAFMdropto, CNNWithCAFMMoreBlock


# ======================
# 测试配置参数
# ======================
DATA_DIR = "D:\workspace\Graduation-Project/Test/train_filter_150"  # 训练集.mat文件目录
LABEL_PATH = "D:\workspace\Graduation-Project/Test/train_label_150.mat"  # 训练标签.mat文件路径
TEST_MAT_DIR = "D:\workspace\Graduation-Project\Test/test_filter_100"  # 测试集.mat文件目录
LABEL_MAT_PATH = "D:\workspace\Graduation-Project\Test/test_label_100.mat"  # 测试标签文件路径
DEV_MAT_DIR = "D:\workspace\Graduation-Project\Test\dev"  # 验证集路径
DEV_LABEL_MAT_PATH = "D:\workspace\Graduation-Project\Test/develop_label.mat"  # 验证集标签
MODEL_PATH = "D:\workspace\Graduation-Project\Test/trained_model.pth"  # 训练好的模型路径
BATCH_SIZE = 8  # 与训练时保持一致
INPUT_VAR_NAME = 'resultimage'  # 自带的滤波器
# INPUT_VAR_NAME = 'b_filter'  # 滤波器b
# INPUT_VAR_NAME = 'no_filter'  # 无滤波
# INPUT_VAR_NAME = 'rnn_data_b'  # 滤波器b
# INPUT_VAR_NAME = 'rnn_data'  # 自带的滤波器

LABEL_VAR_NAME = 'data'  # 标签变量名
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 5
EPOCH = 50


# 早停法类
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=PATIENCE, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): 验证损失没有改善时的容忍次数. Default: 5
            delta (float): 验证损失的最小变化，用于判断是否改善. Default: 0
            path (str): 模型权重保存路径. Default: 'checkpoint.pt'
        """
        self.verbose = None
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存当前模型当验证损失下降时"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 1. 数据准备模块 --------------------------------------------------
class MatDataset(Dataset):
    def __init__(self, data_dir, label_path,
                 input_var=INPUT_VAR_NAME, label_var=LABEL_VAR_NAME):
        """
        data_dir: 包含输入.mat文件的目录
        label_path: 标签.mat文件路径
        input_var: 输入数据的变量名（'amp_map'）
        label_var: 标签数据的变量名（'data'）
        """
        # 加载输入文件列表
        self.data_files = [os.path.join(data_dir, f)
                           for f in os.listdir(data_dir)
                           if f.endswith('.mat')]
        self.data_files.sort()  # 确保顺序固定

        # 加载标签数据
        label_data = loadmat(label_path)
        self.labels = torch.FloatTensor(label_data[label_var])

        # 验证数据一致性
        assert len(self.data_files) == self.labels.shape[0], \
            f"数据量不匹配: {len(self.data_files)}个输入 vs {self.labels.shape[0]}个标签"

        self.input_var = input_var

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # 加载输入数据
        mat_data = hdf5storage.loadmat(self.data_files[idx])
        x = mat_data[self.input_var].astype(np.float32)
        # 转换为Tensor并添加通道维度
        x_tensor = torch.from_numpy(x)  # 形状 [C, T]

        # 数据标准化（通道级）
        x_tensor = (x_tensor - x_tensor.mean(dim=1, keepdim=True)) / \
                   (x_tensor.std(dim=1, keepdim=True) + 1e-8)
        # 获取对应标签
        y_tensor = self.labels[idx]

        return x_tensor, y_tensor


# 2. 测试 -----------------------------------------------------
# 测试流程函数
# ======================
def test_model(model, test_loader):
    running_loss = 0.0
    criterion = nn.L1Loss()
    model.eval()
    all_preds = []
    all_labels = []
    mae_test_list = []
    rmse_test_list = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            # 前向传播
            outputs = model(inputs)

            # 保存结果
            mae, rmse = metric(np.floor(outputs.cpu().numpy()), labels.numpy())
            mae_test_list.append(mae)
            rmse_test_list.append(rmse)
            loss = criterion(outputs.cpu(), labels)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)
    mae = np.mean(np.array(mae_test_list))
    rmse = np.mean(np.array(rmse_test_list))

    return mae, rmse, epoch_loss


def dev_valid(model, dev_loader):
    # 设置为评估模式
    model.eval()
    #  定义评估指标
    criterion = nn.MSELoss()  # 假设回归任务，使用均方误差
    #  验证过程
    val_loss = 0.0
    all_predictions = []
    all_targets = []
    mae_list = []
    rmse_list = []
    with torch.no_grad():
        for inputs, targets in dev_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # 前向传播
            outputs = model(inputs)

            mae, rmse = metric(outputs.cpu().numpy(), targets.cpu().numpy())
            mae_list.append(mae)
            rmse_list.append(rmse)
            # 计算损失
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            # 保存预测结果
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 计算平均损失
    val_loss = val_loss / len(dev_loader.dataset)

    # results_df = pd.DataFrame({
    #     "Predictions": np.array(all_predictions).squeeze(),
    #     "Targets": np.array(all_targets).squeeze()
    # })

    return mae_list, rmse_list


def test_acc(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)

            # 前向传播
            outputs = model(inputs)

            # 保存结果
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    # 合并结果
    preds = np.floor(np.concatenate(all_preds, axis=0))
    labels = np.concatenate(all_labels, axis=0)

    return preds.squeeze(), labels.squeeze()


# 3. 训练循环 -----------------------------------------------------
def train_full(model, test_loader, data_dir, label_path,
               epochs, batch_size=BATCH_SIZE, lr=1e-3):
    # 初始化数据集和数据加载器
    dataset = MatDataset(data_dir, label_path)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 每隔十轮学习率降到原来的%
    step_schedule = optim.lr_scheduler.StepLR(step_size=30, gamma=0.6, optimizer=optimizer)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 初始化早停法
    early_stopping = EarlyStopping(patience=PATIENCE * 2, delta=0.001, path='best_model.pt')

    log_test_mae = []
    log_test_rmse = []
    test_loss_list = []

    mae_list = []
    rmse_list = []
    loss_list = []
    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        mae_inner_list = []
        rmse_inner_list = []
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # 保持维度一致

            optimizer.zero_grad()
            outputs = model(inputs)

            mae, rmse = metric(np.floor(outputs.detach().cpu().numpy()), labels.detach().cpu().numpy())
            mae_inner_list.append(mae)
            rmse_inner_list.append(rmse)

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # step_schedule.step()

            running_loss += loss.item() * inputs.size(0)

        mae = np.mean(np.array(mae_inner_list))
        rmse = np.mean(np.array(rmse_inner_list))
        mae_list.append(mae)
        rmse_list.append(rmse)
        val_mae, val_rmse, val_loss = test_model(model, test_loader)
        log_test_mae.append(val_mae)
        log_test_rmse.append(val_rmse)

        # 早停法检查
        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping triggered")
        #     break

        epoch_loss = running_loss / len(dataset)
        loss_list.append(epoch_loss)
        test_loss_list.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} | train_Loss: {epoch_loss:.4f} | test_loss: {val_loss:.4f}")
        print(f"train:MAE:{mae:.2f} | RMSE:{rmse:.2f} test:MAE:{val_mae:.2f}|RMSE:{val_rmse:.2f}")

        scheduler.step(val_loss)

    return mae_list, rmse_list, loss_list, log_test_mae, log_test_rmse, test_loss_list


# 4. 主程序 ------------------------------------------------------
if __name__ == "__main__":
    # 初始化模型
    model = CNNWithCAFMMoreBlock(input_channels=29)  # 根据实际数据通道数设置
    # model = GRUModel(s
    #     input_size=29,
    #     hidden_size=128,
    #     num_layers=2,
    #     bidirectional=True,
    #     dropout=0.3,
    #     num_classes=1
    # )
    # 1. 加载测试数据
    test_dataset = MatDataset(TEST_MAT_DIR, LABEL_MAT_PATH)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    # 2. 加载验证数据
    dev_dataset = MatDataset(DEV_MAT_DIR, DEV_LABEL_MAT_PATH)
    dev_loader = DataLoader(dev_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # 开始训练
    mae_list, rmse_list, loss_list, log_test_mae, log_test_rmse, log_test_loss = train_full(model,
                                                                                            test_loader,
                                                                                            data_dir=DATA_DIR,
                                                                                            label_path=LABEL_PATH,
                                                                                            epochs=EPOCH,
                                                                                            batch_size=BATCH_SIZE, )
    # 保存模型
    torch.save(model.state_dict(), "trained_model.pth")
    torch.save(model, "trained_model.pth")

    # 2. 加载训练好的模型
    model = torch.load(MODEL_PATH, weights_only=False).to(DEVICE)

    # val_mae_list, val_rmse_list = dev_valid(model, dev_loader)

    # 执行测试
    # predictions, ground_truth = test_acc(model, test_loader)
    #
    # acc = accuracy_score(predictions, ground_truth)
    # print(acc)

    # # 创建一个包含三个子图的图表，排列成一行三列
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1行3列
    #
    # 第一个子图：训练MAE和RMSE
    axs[0].plot(mae_list, label='Training MAE', marker='o')
    axs[0].plot(rmse_list, label='Training RMSE', marker='s')
    axs[0].set_title('Training MAE and RMSE')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Error')
    axs[0].legend()

    # 第二个子图：测试MAE和RMSE
    axs[1].plot(log_test_mae, label='Test MAE', marker='o')
    axs[1].plot(log_test_rmse, label='Test RMSE', marker='s')
    axs[1].set_title('Test MAE and RMSE')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Error')
    axs[1].legend()

    # # 第三个子图：验证MAE和RMSE
    # axs[2].plot(val_mae_list, label='val MAE', marker='o')
    # axs[2].plot(val_rmse_list, label='val RMSE', marker='s')
    # axs[2].set_title('Valid MAE and RMSE')
    # axs[2].set_xlabel('Epoch')
    # axs[2].set_ylabel('Error')
    # axs[2].legend()
    # #
    # # 第四个子图：训练损失
    # axs[3].plot(loss_list, label='Training Loss', marker='o')
    # axs[3].plot(log_test_loss, label='Test Loss', marker='s')
    # axs[3].set_title('Loss')
    # axs[3].set_xlabel('Epoch')
    # axs[3].set_ylabel('Loss')
    # axs[3].legend()

    axs[2].plot(loss_list, label='Training Loss', marker='o')
    axs[2].plot(log_test_loss, label='Test Loss', marker='s')
    axs[2].set_title('Loss')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].legend()

    # 调整子图间距
    plt.tight_layout()

    # 显示图表
    plt.show()
