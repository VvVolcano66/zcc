import numpy as np
from sympy.printing.pytorch import torch
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from predicate.data_pipeline import SpatioTemporalDataset
from predicate.model.XGBoostPredictor import XGBoostPredictor


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return mae, rmse

# 1. 加载与划分数据
# 获取项目根目录 (当前脚本的上级目录)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data', 'task')

dataset = SpatioTemporalDataset(data_dir=data_dir)
demand_tensor, time_slots = dataset.load_and_gridify()
X_train, Y_train, X_test, Y_test = dataset.create_seq_data(demand_tensor, seq_len=5, pre_len=1)

print(f"数据准备完毕. 测试集形状: {X_test.shape}")


# --- 测试 XGBoost 模型 ---
xgb_model = XGBoostPredictor()
xgb_model.fit(X_train, Y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_mae, xgb_rmse = calculate_metrics(Y_test, xgb_preds)
print(f"[XGBoost Model] MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}")

# --- 测试 PyTorch 深度模型 ---
# 转化为 Tensor
X_tr_t = torch.FloatTensor(X_train)
Y_tr_t = torch.FloatTensor(Y_train)
X_te_t = torch.FloatTensor(X_test)

# 动态导入 CNN-LSTM 模型 (因为文件名包含连字符)
import importlib.util
spec = importlib.util.spec_from_file_location("CNN_LSTM", os.path.join(project_root, "predicate", "model", "CNN_LSTM.py"))
cnn_lstm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn_lstm_module)
SpatioTemporalNet = cnn_lstm_module.SpatioTemporalNet

dl_model = SpatioTemporalNet(seq_len=5)
optimizer = torch.optim.Adam(dl_model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

print("训练 PyTorch 模型中...")
for epoch in range(50):
    optimizer.zero_grad()
    outputs = dl_model(X_tr_t)
    loss = criterion(outputs, Y_tr_t)
    loss.backward()
    optimizer.step()

# DL 评估
dl_model.eval()
with torch.no_grad():
    dl_preds = dl_model(X_te_t).numpy()
dl_mae, dl_rmse = calculate_metrics(Y_test, dl_preds)
print(f"[Deep Model]    MAE: {dl_mae:.4f}, RMSE: {dl_rmse:.4f}")