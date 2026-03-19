import numpy as np
from sympy.printing.pytorch import torch
import os
import sys
import pandas as pd
from datetime import datetime
import importlib.util

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from predicate.data_pipeline import SpatioTemporalDataset
from predicate.model.XGBoostPredictor import XGBoostPredictor


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return mae, rmse


def load_specific_date_files(data_dir, date_list):
    """加载指定日期范围的文件"""
    df_list = []
    for date_str in date_list:
        file_path = os.path.join(data_dir, f'tasks_{date_str}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df_list.append(df)
            print(f"✅ 已加载：tasks_{date_str}.csv")
        else:
            print(f"⚠️ 文件不存在：{file_path}")

    if not df_list:
        raise FileNotFoundError("未找到任何有效的数据文件")

    all_data = pd.concat(df_list, ignore_index=True)
    print(f"\n📊 共加载 {len(date_list)} 天的数据，总计 {len(all_data)} 条任务记录")
    return all_data


def filter_by_time_range(df, start_hour=None, end_hour=None):
    """按小时范围过滤数据"""
    if start_hour is None and end_hour is None:
        return df

    df_copy = df.copy()
    df_copy['first_time'] = pd.to_datetime(df_copy['first_time'])
    df_copy['hour'] = df_copy['first_time'].dt.hour

    if start_hour is not None and end_hour is not None:
        mask = (df_copy['hour'] >= start_hour) & (df_copy['hour'] < end_hour)
        filtered = df_copy[mask].reset_index(drop=True)
        print(f"⏰ 时间范围：{start_hour:02d}:00 - {end_hour:02d}:00")
    elif start_hour is not None:
        mask = df_copy['hour'] >= start_hour
        filtered = df_copy[mask].reset_index(drop=True)
        print(f"⏰ 时间范围：{start_hour:02d}:00 - 23:59")
    elif end_hour is not None:
        mask = df_copy['hour'] < end_hour
        filtered = df_copy[mask].reset_index(drop=True)
        print(f"⏰ 时间范围：00:00 - {end_hour:02d}:00")

    print(f"📋 过滤后剩余 {len(filtered)} 条任务记录")
    return filtered


# ==================== 配置区域 ====================

TRAIN_DATES = [
    '2016-10-08', '2016-10-09', '2016-10-10',
    '2016-10-11', '2016-10-12', '2016-10-13',
    '2016-10-14', '2016-10-15', '2016-10-16',
    '2016-10-17', '2016-10-18', '2016-10-19',
    '2016-10-20', '2016-10-21'
]

TEST_DATES = [
    '2016-10-22'
]

START_HOUR = 6   # 从 6:00 开始（包含早高峰前的时段）
END_HOUR = 10    # 到 10:00 结束（覆盖整个早高峰）

SEQ_LEN = 5
PRE_LEN = 1
TRAIN_RATIO = 0.8
# ================================================

if __name__ == "__main__":
    print("=" * 70)
    print("时空需求预测模型测试（多模型对比实验）")
    print("=" * 70)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'task')

    print(f"\n📁 数据目录：{data_dir}")
    print(f"📅 训练日期：{len(TRAIN_DATES)} 天")
    print(f"📅 测试日期：{len(TEST_DATES)} 天")
    if START_HOUR is not None or END_HOUR is not None:
        print(f"⏰ 测试时段：{START_HOUR if START_HOUR else '00'}:00 - {END_HOUR if END_HOUR else '24'}:00")
    else:
        print("⏰ 测试时段：全天 24 小时")
    print("=" * 70)

    print("\n【步骤 1】加载训练数据...")
    train_data = load_specific_date_files(data_dir, TRAIN_DATES)
    train_data = filter_by_time_range(train_data, START_HOUR, END_HOUR)

    print("\n【步骤 2】加载测试数据...")
    test_data = load_specific_date_files(data_dir, TEST_DATES)
    test_data = filter_by_time_range(test_data, START_HOUR, END_HOUR)

    print("\n【步骤 3】构建时空网格...")
    dataset = SpatioTemporalDataset(data_dir=data_dir, time_interval=30)

    print("\n【步骤 4】生成训练张量...")
    demand_tensor_train, time_slots_train = dataset.load_and_gridify_from_dataframe(train_data)

    print("\n【步骤 5】生成测试张量...")
    demand_tensor_test, time_slots_test = dataset.load_and_gridify_from_dataframe(test_data)

    print("\n【步骤 6】创建滑动窗口数据集...")
    X_train, Y_train = dataset.create_seq_data_single_tensor(
        demand_tensor_train, seq_len=SEQ_LEN, pre_len=PRE_LEN
    )
    X_test, Y_test = dataset.create_seq_data_single_tensor(
        demand_tensor_test, seq_len=SEQ_LEN, pre_len=PRE_LEN
    )

    print(f"✅ 数据集准备完毕")
    print(f"   训练集：X={X_train.shape}, Y={Y_train.shape}")
    print(f"   测试集：X={X_test.shape}, Y={Y_test.shape}")

    # ================= 准备深度学习 Tensor =================
    X_tr_t = torch.FloatTensor(X_train)
    Y_tr_t = torch.FloatTensor(Y_train)
    X_te_t = torch.FloatTensor(X_test)
    Y_te_t = torch.FloatTensor(Y_test)

    # ================= 模型测试 1：XGBoost =================
    print("\n" + "=" * 70)
    print("【模型测试 1】XGBoost 基线模型")
    print("=" * 70)
    xgb_model = XGBoostPredictor()
    xgb_model.fit(X_train, Y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_mae, xgb_rmse = calculate_metrics(Y_test, xgb_preds)
    print(f"[XGBoost Model] MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}")

    # ================= 模型测试 2：CNN-LSTM =================
    print("\n" + "=" * 70)
    print("【模型测试 2】CNN-LSTM 深度学习基线")
    print("=" * 70)

    spec_cnn = importlib.util.spec_from_file_location("CNN_LSTM",
                                                      os.path.join(project_root, "predicate", "model", "CNN_LSTM.py"))
    cnn_lstm_module = importlib.util.module_from_spec(spec_cnn)
    spec_cnn.loader.exec_module(cnn_lstm_module)
    SpatioTemporalNet = cnn_lstm_module.SpatioTemporalNet

    dl_model = SpatioTemporalNet(seq_len=SEQ_LEN)
    optimizer_dl = torch.optim.Adam(dl_model.parameters(), lr=0.01)
    criterion_dl = torch.nn.MSELoss()

    print(f"训练 {50} 个 epoch...")
    for epoch in range(50):
        optimizer_dl.zero_grad()
        outputs = dl_model(X_tr_t)
        loss = criterion_dl(outputs, Y_tr_t)
        loss.backward()
        optimizer_dl.step()

    dl_model.eval()
    with torch.no_grad():
        dl_preds = dl_model(X_te_t).numpy()
    dl_mae, dl_rmse = calculate_metrics(Y_test, dl_preds)
    print(f"[CNN-LSTM]      MAE: {dl_mae:.4f}, RMSE: {dl_rmse:.4f}")

    print("\n" + "=" * 70)
    print("【模型测试 3】ST-Transformer 时空注意力网络 (论文创新)")
    print("=" * 70)

    spec_st = importlib.util.spec_from_file_location("ST_Transformer", os.path.join(project_root, "predicate", "model",
                                                                                    "ST_Transformer.py"))
    st_module = importlib.util.module_from_spec(spec_st)
    spec_st.loader.exec_module(st_module)
    ST_Transformer = st_module.ST_Transformer

    # 动态获取网格大小 (H, W)
    grid_size = (X_train.shape[2], X_train.shape[3])

    # 初始化你的创新模型
    st_model = ST_Transformer(seq_len=SEQ_LEN, grid_size=grid_size, d_model=64, nhead=4, num_layers=2)
    optimizer_st = torch.optim.Adam(st_model.parameters(), lr=0.005)
    criterion_st = torch.nn.MSELoss()

    print(f"训练 {50} 个 epoch...")
    for epoch in range(50):
        optimizer_st.zero_grad()
        outputs_st = st_model(X_tr_t)

        # 适配维度：由于模型输出的是 (Batch, H, W)，而标签是 (Batch, 1, H, W)
        target_st = Y_tr_t.squeeze(1) if len(Y_tr_t.shape) == 4 else Y_tr_t

        loss_st = criterion_st(outputs_st, target_st)
        loss_st.backward()
        optimizer_st.step()

        # 打印部分日志
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/50], Loss: {loss_st.item():.6f}")

    st_model.eval()
    with torch.no_grad():
        st_preds_raw = st_model(X_te_t).numpy()
        # 恢复预测维度以匹配真实标签 (Batch, 1, H, W)
        st_preds = np.expand_dims(st_preds_raw, axis=1) if len(st_preds_raw.shape) == 3 else st_preds_raw

    st_mae, st_rmse = calculate_metrics(Y_test, st_preds)
    print(f"[ST-Transformer] MAE: {st_mae:.4f}, RMSE: {st_rmse:.4f}")

    # ================= 模型测试 4：ST-GCN 图卷积网络 =================
    print("\n" + "=" * 70)
    print("【模型测试 4】ST-GCN 时空图卷积网络")
    print("=" * 70)

    spec_gcn = importlib.util.spec_from_file_location("ST_GCN",
                                                      os.path.join(project_root, "predicate", "model", "ST_GCN.py"))
    gcn_module = importlib.util.module_from_spec(spec_gcn)
    spec_gcn.loader.exec_module(gcn_module)
    ST_GCN = gcn_module.ST_GCN

    # 初始化 ST-GCN 模型
    gcn_model = ST_GCN(seq_len=SEQ_LEN, grid_size=grid_size, hidden_dim=64)
    optimizer_gcn = torch.optim.Adam(gcn_model.parameters(), lr=0.001)
    criterion_gcn = torch.nn.MSELoss()

    print(f"训练 {50} 个 epoch...")
    for epoch in range(50):
        optimizer_gcn.zero_grad()
        outputs_gcn = gcn_model(X_tr_t)

        # 适配维度
        target_gcn = Y_tr_t.squeeze(1) if len(Y_tr_t.shape) == 4 else Y_tr_t

        loss_gcn = criterion_gcn(outputs_gcn, target_gcn)
        loss_gcn.backward()
        optimizer_gcn.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/50], Loss: {loss_gcn.item():.6f}")

    gcn_model.eval()
    with torch.no_grad():
        gcn_preds_raw = gcn_model(X_te_t).numpy()
        gcn_preds = np.expand_dims(gcn_preds_raw, axis=1) if len(gcn_preds_raw.shape) == 3 else gcn_preds_raw

    gcn_mae, gcn_rmse = calculate_metrics(Y_test, gcn_preds)
    print(f"[ST-GCN]        MAE: {gcn_mae:.4f}, RMSE: {gcn_rmse:.4f}")

    # ================= 7. 输出对比结果 =================
    print("\n" + "=" * 70)
    print("📊 毕业论文实验：模型性能横向对比")
    print("=" * 70)
    print(f"{'模型':<20} {'MAE (平均绝对误差)':<20} {'RMSE (均方根误差)':<20}")
    print("-" * 65)
    print(f"{'XGBoost (ML 基线)':<22} {xgb_mae:<20.4f} {xgb_rmse:<20.4f}")
    print(f"{'CNN-LSTM (DL 基线)':<22} {dl_mae:<20.4f} {dl_rmse:<20.4f}")
    print(f"{'ST-Transformer (创新)':<21} {st_mae:<20.4f} {st_rmse:<20.4f}")
    print(f"{'ST-GCN (图卷积)':<21} {gcn_mae:<20.4f} {gcn_rmse:<20.4f}")

    # 自动找出 MAE 最小的模型
    models_perf = {
        "XGBoost": xgb_mae,
        "CNN-LSTM": dl_mae,
        "ST-Transformer": st_mae,
        "ST-GCN": gcn_mae
    }
    better_model = min(models_perf, key=models_perf.get)
    print(f"\n🏆 综合表现最优的模型：{better_model}")