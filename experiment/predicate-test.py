import numpy as np
import torch
import os
import sys
import pandas as pd
import copy
from datetime import datetime
import importlib.util

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import config
from predicate.data_pipeline import SpatioTemporalDataset
from predicate.model.XGBoostPredictor import XGBoostPredictor


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return mae, rmse


def clone_model_state(model):
    return copy.deepcopy(model.state_dict())


def create_plateau_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=50,
        min_lr=1e-5
    )


def current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def transform_for_training(array, use_log1p=True):
    arr = array.astype(np.float32)
    return np.log1p(arr) if use_log1p else arr


def invert_from_training_space(array, use_log1p=True):
    arr = np.asarray(array)
    if use_log1p:
        arr = np.expm1(arr)
    return np.clip(arr, 0.0, None)


def weighted_mse_loss(pred, target, raw_target, alpha=2.0):
    mean_target = torch.mean(raw_target)
    weights = 1.0 + alpha * raw_target / (mean_target + 1e-6)
    return torch.mean(weights * (pred - target) ** 2)


def persistence_predict(X):
    """
    Last-value baseline: use the final observed frame as the next prediction.
    """
    return X[:, -1:, :, :]


def discover_available_task_dates(data_dir):
    date_list = []
    for file_name in sorted(os.listdir(data_dir)):
        if not (file_name.startswith('tasks_') and file_name.endswith('.csv')):
            continue
        date_str = file_name[len('tasks_'):-len('.csv')]
        try:
            pd.Timestamp(date_str)
            date_list.append(date_str)
        except Exception:
            continue
    if not date_list:
        raise FileNotFoundError(f"No task CSV files found in {data_dir}")
    return sorted(date_list)


def build_date_split(date_list, val_days=3, test_days=3):
    if len(date_list) < val_days + test_days + 3:
        raise ValueError("Not enough daily task files to build train/val/test split.")

    train_dates = date_list[:-(val_days + test_days)]
    val_dates = date_list[-(val_days + test_days):-test_days]
    test_dates = date_list[-test_days:]
    return train_dates, val_dates, test_dates


def build_periodic_features(demand_tensor, all_slots, seq_len, pre_len, start_hour, time_interval):
    slot_to_idx = {pd.Timestamp(ts): idx for idx, ts in enumerate(all_slots)}
    periodic_inputs = []
    target_slot_ids = []

    for i in range(len(demand_tensor) - seq_len - pre_len + 1):
        periodic_frames = []
        for step in range(pre_len):
            target_ts = pd.Timestamp(all_slots[i + seq_len + step])
            prev_day_ts = target_ts - pd.Timedelta(days=1)
            if prev_day_ts in slot_to_idx:
                periodic_frames.append(demand_tensor[slot_to_idx[prev_day_ts]])
            else:
                periodic_frames.append(np.zeros_like(demand_tensor[0]))

        first_target_ts = pd.Timestamp(all_slots[i + seq_len])
        slot_id = ((first_target_ts.hour * 60 + first_target_ts.minute) - start_hour * 60) // time_interval
        target_slot_ids.append(max(0, int(slot_id)))
        periodic_inputs.append(np.stack(periodic_frames, axis=0))

    return np.array(periodic_inputs), np.array(target_slot_ids, dtype=np.int64)

def load_specific_date_files(data_dir, date_list):
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
VAL_DAYS = 3
TEST_DAYS = 3
START_HOUR = 6
END_HOUR = 10
SEQ_LEN = 5
PRE_LEN = 1
max_epochs = 3000
patience = 200
USE_LOG1P_TRANSFORM = True
# ================================================

if __name__ == "__main__":
    print("=" * 70)
    print("时空需求预测模型测试（多模型对比实验 - 严格 Train/Val/Test 划分）")
    print("=" * 70)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'task')
    all_dates = discover_available_task_dates(data_dir)
    TRAIN_DATES, VAL_DATES, TEST_DATES = build_date_split(
        all_dates, val_days=VAL_DAYS, test_days=TEST_DAYS
    )

    print("使用自动日期切分：")
    print(f"  Train: {TRAIN_DATES[0]} -> {TRAIN_DATES[-1]} ({len(TRAIN_DATES)} days)")
    print(f"  Val:   {VAL_DATES[0]} -> {VAL_DATES[-1]} ({len(VAL_DATES)} days)")
    print(f"  Test:  {TEST_DATES[0]} -> {TEST_DATES[-1]} ({len(TEST_DATES)} days)")

    print("\n【步骤 1】加载训练集 (用于更新模型权重)...")
    train_data = load_specific_date_files(data_dir, TRAIN_DATES)
    train_data = filter_by_time_range(train_data, START_HOUR, END_HOUR)

    print("\n【步骤 2】加载验证集 (用于触发早停 Early Stopping)...")
    val_data = load_specific_date_files(data_dir, VAL_DATES)
    val_data = filter_by_time_range(val_data, START_HOUR, END_HOUR)

    print("\n【步骤 3】加载测试集 (绝对盲盒，仅用于最终评价)...")
    test_data = load_specific_date_files(data_dir, TEST_DATES)
    test_data = filter_by_time_range(test_data, START_HOUR, END_HOUR)

    print("\n【步骤 4】构建时空网格并生成张量...")
    dataset = SpatioTemporalDataset(data_dir=data_dir, time_interval=30)
    demand_tensor_train, slots_train = dataset.load_and_gridify_from_dataframe(
        train_data, start_hour=START_HOUR, end_hour=END_HOUR
    )
    demand_tensor_val, slots_val = dataset.load_and_gridify_from_dataframe(
        val_data, start_hour=START_HOUR, end_hour=END_HOUR
    )
    demand_tensor_test, slots_test = dataset.load_and_gridify_from_dataframe(
        test_data, start_hour=START_HOUR, end_hour=END_HOUR
    )

    print("\n【步骤 5】创建滑动窗口序列...")
    X_train, Y_train = dataset.create_seq_data_single_tensor(demand_tensor_train, seq_len=SEQ_LEN, pre_len=PRE_LEN)
    X_val, Y_val = dataset.create_seq_data_single_tensor(demand_tensor_val, seq_len=SEQ_LEN, pre_len=PRE_LEN)
    X_test, Y_test = dataset.create_seq_data_single_tensor(demand_tensor_test, seq_len=SEQ_LEN, pre_len=PRE_LEN)
    X_train_periodic, train_slot_ids = build_periodic_features(
        demand_tensor_train, slots_train, seq_len=SEQ_LEN, pre_len=PRE_LEN,
        start_hour=START_HOUR, time_interval=dataset.time_interval
    )
    X_val_periodic, val_slot_ids = build_periodic_features(
        demand_tensor_val, slots_val, seq_len=SEQ_LEN, pre_len=PRE_LEN,
        start_hour=START_HOUR, time_interval=dataset.time_interval
    )
    X_test_periodic, test_slot_ids = build_periodic_features(
        demand_tensor_test, slots_test, seq_len=SEQ_LEN, pre_len=PRE_LEN,
        start_hour=START_HOUR, time_interval=dataset.time_interval
    )

    Y_test_raw = Y_test.copy()
    X_train_model = transform_for_training(X_train, use_log1p=USE_LOG1P_TRANSFORM)
    Y_train_model = transform_for_training(Y_train, use_log1p=USE_LOG1P_TRANSFORM)
    X_val_model = transform_for_training(X_val, use_log1p=USE_LOG1P_TRANSFORM)
    Y_val_model = transform_for_training(Y_val, use_log1p=USE_LOG1P_TRANSFORM)
    X_test_model = transform_for_training(X_test, use_log1p=USE_LOG1P_TRANSFORM)
    Y_test_model = transform_for_training(Y_test, use_log1p=USE_LOG1P_TRANSFORM)
    X_train_periodic_model = transform_for_training(X_train_periodic, use_log1p=USE_LOG1P_TRANSFORM)
    X_val_periodic_model = transform_for_training(X_val_periodic, use_log1p=USE_LOG1P_TRANSFORM)
    X_test_periodic_model = transform_for_training(X_test_periodic, use_log1p=USE_LOG1P_TRANSFORM)

    print(f"✅ 数据集准备完毕")
    print(f"   [Train] 训练集：X={X_train.shape}")
    print(f"   [Val]   验证集：X={X_val.shape} (用于早停)")
    print(f"   [Test]  测试集：X={X_test.shape} (用于最终对比)")
    print(f"   [Target Transform] {'log1p' if USE_LOG1P_TRANSFORM else 'none'}")

    grid_size = (X_train.shape[2], X_train.shape[3])
    num_nodes = grid_size[0] * grid_size[1]

    # === 【核心修改】检测并使用显卡 GPU ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f"🚀 深度学习计算设备已切换至：{device.type.upper()}")
    print("=" * 70)

    # === 生成深度学习所需的 Tensor 并搬运到 GPU ===
    X_tr_t = torch.FloatTensor(X_train_model).to(device)
    Y_tr_t = torch.FloatTensor(Y_train_model).to(device)
    X_val_t = torch.FloatTensor(X_val_model).to(device)
    Y_val_t = torch.FloatTensor(Y_val_model).to(device)
    X_te_t = torch.FloatTensor(X_test_model).to(device)
    Y_te_t = torch.FloatTensor(Y_test_model).to(device)
    Y_tr_raw_t = torch.FloatTensor(Y_train).to(device)
    Y_val_raw_t = torch.FloatTensor(Y_val).to(device)
    X_tr_periodic_t = torch.FloatTensor(X_train_periodic_model).to(device)
    X_val_periodic_t = torch.FloatTensor(X_val_periodic_model).to(device)
    X_te_periodic_t = torch.FloatTensor(X_test_periodic_model).to(device)
    train_slot_ids_t = torch.LongTensor(train_slot_ids).to(device)
    val_slot_ids_t = torch.LongTensor(val_slot_ids).to(device)
    test_slot_ids_t = torch.LongTensor(test_slot_ids).to(device)

    # ================= 模型测试 1：XGBoost =================
    print("\n" + "=" * 70)
    print("【模型测试 1】XGBoost 基线模型")
    print("=" * 70)

    persistence_preds = persistence_predict(X_test)
    persistence_mae, persistence_rmse = calculate_metrics(Y_test_raw, persistence_preds)
    print(f"[Persistence]    MAE: {persistence_mae:.4f}, RMSE: {persistence_rmse:.4f}")

    xgb_model = XGBoostPredictor()
    xgb_model.fit(X_train_model, Y_train_model)
    xgb_preds = invert_from_training_space(
        xgb_model.predict(X_test_model), use_log1p=USE_LOG1P_TRANSFORM
    )
    xgb_mae, xgb_rmse = calculate_metrics(Y_test_raw, xgb_preds)
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

    dl_model = SpatioTemporalNet(seq_len=SEQ_LEN).to(device) # <--- 搬到 GPU
    optimizer_dl = torch.optim.Adam(dl_model.parameters(), lr=0.001)
    scheduler_dl = create_plateau_scheduler(optimizer_dl)
    criterion_dl = torch.nn.MSELoss()

    best_val_loss_dl = float('inf')
    patience_counter_dl = 0
    best_state_dl = clone_model_state(dl_model)
    best_epoch_dl = 0

    for epoch in range(max_epochs):
        dl_model.train()
        optimizer_dl.zero_grad()
        outputs_dl = dl_model(X_tr_t)
        loss_dl = criterion_dl(outputs_dl, Y_tr_t)
        loss_dl.backward()
        optimizer_dl.step()

        dl_model.eval()
        with torch.no_grad():
            preds_val_dl = dl_model(X_val_t)
            val_loss_dl = criterion_dl(preds_val_dl, Y_val_t).item()

        scheduler_dl.step(val_loss_dl)

        if val_loss_dl < best_val_loss_dl:
            best_val_loss_dl = val_loss_dl
            patience_counter_dl = 0
            best_state_dl = clone_model_state(dl_model)
            best_epoch_dl = epoch + 1
        else:
            patience_counter_dl += 1

        if patience_counter_dl >= patience:
            print(f"  Early stop at epoch {epoch + 1}, best epoch = {best_epoch_dl}, best val loss = {best_val_loss_dl:.4f}")
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch + 1:04d}/{max_epochs}], Train Loss: {loss_dl.item():.4f} | "
                f"Val Loss: {val_loss_dl:.4f} | LR: {current_lr(optimizer_dl):.6f}"
            )

    dl_model.load_state_dict(best_state_dl)
    dl_model.eval()
    with torch.no_grad():
        dl_preds = invert_from_training_space(
            dl_model(X_te_t).cpu().numpy(), use_log1p=USE_LOG1P_TRANSFORM
        )
    dl_mae, dl_rmse = calculate_metrics(Y_test_raw, dl_preds)
    print(f"  Best Epoch: {best_epoch_dl}, Best Val Loss: {best_val_loss_dl:.4f}")
    print(f"[CNN-LSTM]      MAE: {dl_mae:.4f}, RMSE: {dl_rmse:.4f}")

    # ================= 模型测试 3：ST-Transformer =================
    print("\n" + "=" * 70)
    print("【模型测试 3】ST-Transformer 时空注意力网络")
    print("=" * 70)

    spec_st = importlib.util.spec_from_file_location("ST_Transformer", os.path.join(project_root, "predicate", "model",
                                                                                    "ST_Transformer.py"))
    st_module = importlib.util.module_from_spec(spec_st)
    spec_st.loader.exec_module(st_module)
    ST_Transformer = st_module.ST_Transformer

    st_model = ST_Transformer(seq_len=SEQ_LEN, grid_size=grid_size, d_model=64, nhead=4, num_layers=2).to(device)
    optimizer_st = torch.optim.Adam(st_model.parameters(), lr=0.0005)
    scheduler_st = create_plateau_scheduler(optimizer_st)
    criterion_st = torch.nn.MSELoss()

    best_val_loss_st = float('inf')
    patience_counter_st = 0
    best_state_st = clone_model_state(st_model)
    best_epoch_st = 0

    for epoch in range(max_epochs):
        st_model.train()
        optimizer_st.zero_grad()
        outputs_st = st_model(X_tr_t)
        target_st = Y_tr_t.squeeze(1) if len(Y_tr_t.shape) == 4 else Y_tr_t
        loss_st = criterion_st(outputs_st, target_st)
        loss_st.backward()
        optimizer_st.step()

        st_model.eval()
        with torch.no_grad():
            preds_val_st = st_model(X_val_t)
            target_val_st = Y_val_t.squeeze(1) if len(Y_val_t.shape) == 4 else Y_val_t
            val_loss_st = criterion_st(preds_val_st, target_val_st).item()

        scheduler_st.step(val_loss_st)

        if val_loss_st < best_val_loss_st:
            best_val_loss_st = val_loss_st
            patience_counter_st = 0
            best_state_st = clone_model_state(st_model)
            best_epoch_st = epoch + 1
        else:
            patience_counter_st += 1

        if patience_counter_st >= patience:
            print(f"  Early stop at epoch {epoch + 1}, best epoch = {best_epoch_st}, best val loss = {best_val_loss_st:.4f}")
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch + 1:04d}/{max_epochs}], Train Loss: {loss_st.item():.4f} | "
                f"Val Loss: {val_loss_st:.4f} | LR: {current_lr(optimizer_st):.6f}"
            )

    st_model.load_state_dict(best_state_st)
    st_model.eval()
    with torch.no_grad():
        st_preds_raw = st_model(X_te_t).cpu().numpy()
        st_preds = np.expand_dims(st_preds_raw, axis=1) if len(st_preds_raw.shape) == 3 else st_preds_raw
        st_preds = invert_from_training_space(st_preds, use_log1p=USE_LOG1P_TRANSFORM)
    st_mae, st_rmse = calculate_metrics(Y_test_raw, st_preds)
    print(f"  Best Epoch: {best_epoch_st}, Best Val Loss: {best_val_loss_st:.4f}")
    print(f"[ST-Transformer] MAE: {st_mae:.4f}, RMSE: {st_rmse:.4f}")

    # ================= 模型测试 4：ST-GCN =================
    print("\n" + "=" * 70)
    print("【模型测试 4】ST-GCN 时空图卷积网络")
    print("=" * 70)

    spec_gcn = importlib.util.spec_from_file_location("ST_GCN",
                                                      os.path.join(project_root, "predicate", "model", "ST_GCN.py"))
    gcn_module = importlib.util.module_from_spec(spec_gcn)
    spec_gcn.loader.exec_module(gcn_module)
    ST_GCN = gcn_module.ST_GCN

    gcn_model = ST_GCN(num_nodes=num_nodes, seq_len=SEQ_LEN, hidden_dim=64).to(device)
    optimizer_gcn = torch.optim.Adam(gcn_model.parameters(), lr=0.001)
    scheduler_gcn = create_plateau_scheduler(optimizer_gcn)
    criterion_gcn = torch.nn.MSELoss()

    adj = torch.zeros((num_nodes, num_nodes))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            idx = i * grid_size[1] + j
            adj[idx, idx] = 1.0
            if i > 0: adj[idx, (i - 1) * grid_size[1] + j] = 1.0
            if i < grid_size[0] - 1: adj[idx, (i + 1) * grid_size[1] + j] = 1.0
            if j > 0: adj[idx, i * grid_size[1] + (j - 1)] = 1.0
            if j < grid_size[1] - 1: adj[idx, i * grid_size[1] + (j + 1)] = 1.0

    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_normalized = adj.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt).to(device) # <--- 搬到 GPU

    best_val_loss_gcn = float('inf')
    patience_counter_gcn = 0
    best_state_gcn = clone_model_state(gcn_model)
    best_epoch_gcn = 0

    for epoch in range(max_epochs):
        gcn_model.train()
        optimizer_gcn.zero_grad()
        X_tr_gcn = X_tr_t.view(X_tr_t.shape[0], X_tr_t.shape[1], num_nodes)
        outputs_gcn_raw = gcn_model(X_tr_gcn, adj_normalized)
        outputs_gcn = outputs_gcn_raw.view(-1, grid_size[0], grid_size[1])
        target_gcn = Y_tr_t.squeeze(1) if len(Y_tr_t.shape) == 4 else Y_tr_t
        loss_gcn = criterion_gcn(outputs_gcn, target_gcn)
        loss_gcn.backward()
        optimizer_gcn.step()

        gcn_model.eval()
        with torch.no_grad():
            X_val_gcn = X_val_t.view(X_val_t.shape[0], X_val_t.shape[1], num_nodes)
            preds_val_gcn_raw = gcn_model(X_val_gcn, adj_normalized)
            preds_val_gcn = preds_val_gcn_raw.view(-1, grid_size[0], grid_size[1])
            target_val_gcn = Y_val_t.squeeze(1) if len(Y_val_t.shape) == 4 else Y_val_t
            val_loss_gcn = criterion_gcn(preds_val_gcn, target_val_gcn).item()

        scheduler_gcn.step(val_loss_gcn)

        if val_loss_gcn < best_val_loss_gcn:
            best_val_loss_gcn = val_loss_gcn
            patience_counter_gcn = 0
            best_state_gcn = clone_model_state(gcn_model)
            best_epoch_gcn = epoch + 1
        else:
            patience_counter_gcn += 1

        if patience_counter_gcn >= patience:
            print(f"  Early stop at epoch {epoch + 1}, best epoch = {best_epoch_gcn}, best val loss = {best_val_loss_gcn:.4f}")
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch + 1:04d}/{max_epochs}], Train Loss: {loss_gcn.item():.4f} | "
                f"Val Loss: {val_loss_gcn:.4f} | LR: {current_lr(optimizer_gcn):.6f}"
            )

    gcn_model.load_state_dict(best_state_gcn)
    gcn_model.eval()
    with torch.no_grad():
        X_te_gcn = X_te_t.view(X_te_t.shape[0], X_te_t.shape[1], num_nodes)
        gcn_preds_raw = gcn_model(X_te_gcn, adj_normalized)
        gcn_preds_reshaped = gcn_preds_raw.view(-1, grid_size[0], grid_size[1]).cpu().numpy()
        gcn_preds = np.expand_dims(gcn_preds_reshaped, axis=1) if len(
            gcn_preds_reshaped.shape) == 3 else gcn_preds_reshaped

    gcn_preds = invert_from_training_space(gcn_preds, use_log1p=USE_LOG1P_TRANSFORM)
    gcn_mae, gcn_rmse = calculate_metrics(Y_test_raw, gcn_preds)
    print(f"  Best Epoch: {best_epoch_gcn}, Best Val Loss: {best_val_loss_gcn:.4f}")
    print(f"[ST-GCN]        MAE: {gcn_mae:.4f}, RMSE: {gcn_rmse:.4f}")

    # ================= 模型测试 5：BSTGCNet (论文复现) =================
    print("\n" + "=" * 70)
    print("【模型测试 5】BSTGCNet (IEEE TMC 论文复现)")
    print("=" * 70)

    spec_bst = importlib.util.spec_from_file_location("BSTGCNet",
                                                      os.path.join(project_root, "predicate", "model", "BSTGCNet.py"))
    bst_module = importlib.util.module_from_spec(spec_bst)
    spec_bst.loader.exec_module(bst_module)
    BSTGCNet = bst_module.BSTGCNet

    print("构建语义相似度图 Gs, 邻居图 Gn, 距离图 Gd...")

    historical_demand = X_train.reshape(-1, num_nodes).T
    hist_tensor = torch.FloatTensor(historical_demand)
    norm = torch.norm(hist_tensor, p=2, dim=1, keepdim=True)
    norm_tensor = hist_tensor / (norm + 1e-8)
    G_s = torch.mm(norm_tensor, norm_tensor.t()).to(device) # <--- 搬到 GPU

    G_n = adj.clone().to(device) # <--- 搬到 GPU

    G_d = torch.zeros((num_nodes, num_nodes))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            idx1 = i * grid_size[1] + j
            for x in range(grid_size[0]):
                for y in range(grid_size[1]):
                    idx2 = x * grid_size[1] + y
                    dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                    G_d[idx1, idx2] = 1.0 / (dist + 1.0)
    G_d = G_d.to(device) # <--- 搬到 GPU

    bst_model = BSTGCNet(num_nodes=num_nodes, in_features=1, hidden_dim=64, seq_len=SEQ_LEN, pre_len=PRE_LEN).to(device)
    optimizer_bst = torch.optim.Adam(bst_model.parameters(), lr=0.001)
    scheduler_bst = create_plateau_scheduler(optimizer_bst)
    criterion_bst = torch.nn.MSELoss()

    best_val_loss_bst = float('inf')
    patience_counter_bst = 0
    best_state_bst = clone_model_state(bst_model)
    best_epoch_bst = 0

    print(f"训练最大 {max_epochs} 个 epoch，已启用早停机制 (Patience={patience})...")
    for epoch in range(max_epochs):
        bst_model.train()
        optimizer_bst.zero_grad()

        X_tr_bst = X_tr_t.view(X_tr_t.shape[0], X_tr_t.shape[1], num_nodes, 1)
        outputs_bst = bst_model(X_tr_bst, G_s, G_n, G_d)

        target_bst = Y_tr_t.view(Y_tr_t.shape[0], PRE_LEN, num_nodes)
        loss_bst = criterion_bst(outputs_bst, target_bst)
        loss_bst.backward()
        optimizer_bst.step()

        bst_model.eval()
        with torch.no_grad():
            X_val_bst = X_val_t.view(X_val_t.shape[0], X_val_t.shape[1], num_nodes, 1)
            preds_val_bst = bst_model(X_val_bst, G_s, G_n, G_d)
            target_val_bst = Y_val_t.view(Y_val_t.shape[0], PRE_LEN, num_nodes)
            val_loss_bst = criterion_bst(preds_val_bst, target_val_bst).item()

        scheduler_bst.step(val_loss_bst)

        if val_loss_bst < best_val_loss_bst:
            best_val_loss_bst = val_loss_bst
            patience_counter_bst = 0
            best_state_bst = clone_model_state(bst_model)
            best_epoch_bst = epoch + 1
        else:
            patience_counter_bst += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch + 1:04d}/{max_epochs}], Train Loss: {loss_bst.item():.4f} | "
                f"Val Loss: {val_loss_bst:.4f} | LR: {current_lr(optimizer_bst):.6f}")

        if patience_counter_bst >= patience:
            print(
                f"⚠️ BSTGCNet 触发早停！结束于 Epoch: {epoch + 1}, "
                f"best epoch = {best_epoch_bst}, best val loss = {best_val_loss_bst:.4f}"
            )
            break

    bst_model.load_state_dict(best_state_bst)
    bst_model.eval()
    with torch.no_grad():
        X_te_bst = X_te_t.view(X_te_t.shape[0], X_te_t.shape[1], num_nodes, 1)
        bst_preds_raw = bst_model(X_te_bst, G_s, G_n, G_d)
        bst_preds = bst_preds_raw.view(-1, 1, grid_size[0], grid_size[1]).cpu().numpy() # <--- 转回 CPU

    bst_preds = invert_from_training_space(bst_preds, use_log1p=USE_LOG1P_TRANSFORM)
    bst_mae, bst_rmse = calculate_metrics(Y_test_raw, bst_preds)
    print(f"  Best Epoch: {best_epoch_bst}, Best Val Loss: {best_val_loss_bst:.4f}")
    print(f"[BSTGCNet]      MAE: {bst_mae:.4f}, RMSE: {bst_rmse:.4f}")

    # ================= 模型测试 6：MCTGNet (面向多中心调度的创新模型) =================
    print("\n" + "=" * 70)
    print("【模型测试 6】MCTGNet (Multi-Center Trend-Gated Network)")
    print("=" * 70)

    spec_mctg = importlib.util.spec_from_file_location(
        "MCTGNet",
        os.path.join(project_root, "predicate", "model", "MCTGNet.py")
    )
    mctg_module = importlib.util.module_from_spec(spec_mctg)
    spec_mctg.loader.exec_module(mctg_module)
    MCTGNet = mctg_module.MCTGNet

    mctg_model = MCTGNet(
        seq_len=SEQ_LEN,
        grid_size=grid_size,
        hidden_dim=64,
        num_centers=getattr(config, 'NUM_ZONES', 5),
        num_time_slots=((END_HOUR - START_HOUR) * 60) // dataset.time_interval
    ).to(device)
    optimizer_mctg = torch.optim.Adam(mctg_model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler_mctg = create_plateau_scheduler(optimizer_mctg)

    best_val_loss_mctg = float('inf')
    patience_counter_mctg = 0
    best_state_mctg = clone_model_state(mctg_model)
    best_epoch_mctg = 0

    for epoch in range(max_epochs):
        mctg_model.train()
        optimizer_mctg.zero_grad()
        outputs_mctg = mctg_model(X_tr_t, X_tr_periodic_t, train_slot_ids_t)
        target_mctg = Y_tr_t.squeeze(1) if len(Y_tr_t.shape) == 4 else Y_tr_t
        raw_target_mctg = Y_tr_raw_t.squeeze(1) if len(Y_tr_raw_t.shape) == 4 else Y_tr_raw_t
        loss_mctg = weighted_mse_loss(outputs_mctg, target_mctg, raw_target_mctg, alpha=2.5)
        loss_mctg.backward()
        optimizer_mctg.step()

        mctg_model.eval()
        with torch.no_grad():
            preds_val_mctg = mctg_model(X_val_t, X_val_periodic_t, val_slot_ids_t)
            target_val_mctg = Y_val_t.squeeze(1) if len(Y_val_t.shape) == 4 else Y_val_t
            raw_target_val_mctg = Y_val_raw_t.squeeze(1) if len(Y_val_raw_t.shape) == 4 else Y_val_raw_t
            val_loss_mctg = weighted_mse_loss(
                preds_val_mctg, target_val_mctg, raw_target_val_mctg, alpha=2.5
            ).item()

        scheduler_mctg.step(val_loss_mctg)

        if val_loss_mctg < best_val_loss_mctg:
            best_val_loss_mctg = val_loss_mctg
            patience_counter_mctg = 0
            best_state_mctg = clone_model_state(mctg_model)
            best_epoch_mctg = epoch + 1
        else:
            patience_counter_mctg += 1

        if patience_counter_mctg >= patience:
            print(
                f"  Early stop at epoch {epoch + 1}, best epoch = {best_epoch_mctg}, "
                f"best val loss = {best_val_loss_mctg:.4f}"
            )
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch [{epoch + 1:04d}/{max_epochs}], Train Loss: {loss_mctg.item():.4f} | "
                f"Val Loss: {val_loss_mctg:.4f} | LR: {current_lr(optimizer_mctg):.6f}"
            )

    mctg_model.load_state_dict(best_state_mctg)
    mctg_model.eval()
    with torch.no_grad():
        mctg_preds_raw = mctg_model(X_te_t, X_te_periodic_t, test_slot_ids_t).cpu().numpy()
        mctg_preds = np.expand_dims(mctg_preds_raw, axis=1) if len(mctg_preds_raw.shape) == 3 else mctg_preds_raw
        mctg_preds = invert_from_training_space(mctg_preds, use_log1p=USE_LOG1P_TRANSFORM)
    mctg_mae, mctg_rmse = calculate_metrics(Y_test_raw, mctg_preds)
    print(f"  Best Epoch: {best_epoch_mctg}, Best Val Loss: {best_val_loss_mctg:.4f}")
    print(f"[MCTGNet]       MAE: {mctg_mae:.4f}, RMSE: {mctg_rmse:.4f}")

    # ================= 7. 输出对比结果 =================
    print("\n" + "=" * 70)
    print("📊 毕业论文实验：模型性能横向对比 (最终版)")
    print("=" * 70)
    print(f"{'模型':<25} {'MAE (平均绝对误差)':<20} {'RMSE (均方根误差)':<20}")
    print("-" * 70)
    print(f"{'Persistence (Last Value)':<27} {persistence_mae:<20.4f} {persistence_rmse:<20.4f}")
    print(f"{'XGBoost (ML 基线)':<27} {xgb_mae:<20.4f} {xgb_rmse:<20.4f}")
    print(f"{'CNN-LSTM (DL 基线)':<27} {dl_mae:<20.4f} {dl_rmse:<20.4f}")
    print(f"{'ST-Transformer (对比模型)':<24} {st_mae:<20.4f} {st_rmse:<20.4f}")
    print(f"{'ST-GCN (图卷积基线)':<24} {gcn_mae:<20.4f} {gcn_rmse:<20.4f}")
    print(f"{'BSTGCNet (论文复现)':<25} {bst_mae:<20.4f} {bst_rmse:<20.4f}")
    print(f"{'MCTGNet (创新模型)':<25} {mctg_mae:<20.4f} {mctg_rmse:<20.4f}")

    # 自动找出 MAE 最小的模型
    models_perf = {
        "Persistence": persistence_mae,
        "XGBoost": xgb_mae,
        "CNN-LSTM": dl_mae,
        "ST-Transformer": st_mae,
        "ST-GCN": gcn_mae,
        "BSTGCNet": bst_mae,
        "MCTGNet": mctg_mae
    }
    better_model = min(models_perf, key=models_perf.get)
    print(f"\n🏆 综合表现最优的模型：{better_model}")
