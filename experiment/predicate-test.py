import numpy as np
from sympy.printing.pytorch import torch
import os
import sys
import pandas as pd
from datetime import datetime

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
    """
    加载指定日期范围的文件

    Args:
        data_dir: 数据目录路径
        date_list: 日期列表，格式如 ['2016-10-08', '2016-10-09', ...]

    Returns:
        筛选后的 DataFrame
    """
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
    """
    按小时范围过滤数据

    Args:
        df: 原始 DataFrame
        start_hour: 起始小时（包含），如 7 表示 7:00
        end_hour: 结束小时（不包含），如 9 表示 9:00

    Returns:
        过滤后的 DataFrame
    """
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

# 1. 选择用于训练和测试的日期（学习哪些文件）
# 示例：选择前 10 天作为训练数据
TRAIN_DATES = [
    '2016-10-08', '2016-10-09', '2016-10-10',
    '2016-10-11', '2016-10-12', '2016-10-13',
    '2016-10-14', '2016-10-15', '2016-10-16',
    '2016-10-17'
]

# 示例：选择后 5 天作为测试数据
TEST_DATES = [
    '2016-10-18'
]

# 2. 选择时间段（测试哪个时间段）
# 设置为 None 表示使用全天数据
START_HOUR = 7  # 例如：7 表示从 7:00 开始
END_HOUR = 9  # 例如：9 表示到 9:00 结束（不包含 9:00）

# 3. 数据集参数
SEQ_LEN = 5  # 序列长度（用几个历史时间步）
PRE_LEN = 1  # 预测长度（预测未来几个时间步）
TRAIN_RATIO = 0.8  # 训练集比例

# ================================================

if __name__ == "__main__":
    print("=" * 70)
    print("时空需求预测模型测试（可配置版）")
    print("=" * 70)

    # 获取项目根目录
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

    # 1. 加载指定日期的数据
    print("\n【步骤 1】加载训练数据...")
    train_data = load_specific_date_files(data_dir, TRAIN_DATES)
    train_data = filter_by_time_range(train_data, START_HOUR, END_HOUR)

    print("\n【步骤 2】加载测试数据...")
    test_data = load_specific_date_files(data_dir, TEST_DATES)
    test_data = filter_by_time_range(test_data, START_HOUR, END_HOUR)

    # 2. 创建数据集对象
    print("\n【步骤 3】构建时空网格...")
    dataset = SpatioTemporalDataset(data_dir=data_dir, time_interval=30)

    # 3. 分别处理训练集和测试集
    print("\n【步骤 4】生成训练张量...")
    demand_tensor_train, time_slots_train = dataset.load_and_gridify_from_dataframe(train_data)
    print(f"训练张量形状：{demand_tensor_train.shape}")
    print(f"时间片数量：{len(time_slots_train)}")

    print("\n【步骤 5】生成测试张量...")
    demand_tensor_test, time_slots_test = dataset.load_and_gridify_from_dataframe(test_data)
    print(f"测试张量形状：{demand_tensor_test.shape}")
    print(f"时间片数量：{len(time_slots_test)}")

    # 4. 创建序列数据集
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

    # 5. 测试 XGBoost 模型
    print("\n" + "=" * 70)
    print("【模型测试 1】XGBoost")
    print("=" * 70)
    xgb_model = XGBoostPredictor()
    print("训练中...")
    xgb_model.fit(X_train, Y_train)
    print("预测中...")
    xgb_preds = xgb_model.predict(X_test)
    xgb_mae, xgb_rmse = calculate_metrics(Y_test, xgb_preds)
    print(f"[XGBoost Model] MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}")

    # 6. 测试 PyTorch 深度模型
    print("\n" + "=" * 70)
    print("【模型测试 2】CNN-LSTM 深度学习模型")
    print("=" * 70)

    # 转化为 Tensor
    X_tr_t = torch.FloatTensor(X_train)
    Y_tr_t = torch.FloatTensor(Y_train)
    X_te_t = torch.FloatTensor(X_test)
    Y_te_t = torch.FloatTensor(Y_test)

    # 动态导入 CNN-LSTM 模型
    import importlib.util

    spec = importlib.util.spec_from_file_location("CNN_LSTM",
                                                  os.path.join(project_root, "predicate", "model", "CNN_LSTM.py"))
    cnn_lstm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cnn_lstm_module)
    SpatioTemporalNet = cnn_lstm_module.SpatioTemporalNet

    dl_model = SpatioTemporalNet(seq_len=SEQ_LEN)
    optimizer = torch.optim.Adam(dl_model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    print(f"训练 {50} 个 epoch...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = dl_model(X_tr_t)
        loss = criterion(outputs, Y_tr_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch + 1}/50], Loss: {loss.item():.6f}")

    # DL 评估
    dl_model.eval()
    with torch.no_grad():
        dl_preds = dl_model(X_te_t).numpy()
    dl_mae, dl_rmse = calculate_metrics(Y_test, dl_preds)
    print(f"\n[Deep Model]    MAE: {dl_mae:.4f}, RMSE: {dl_rmse:.4f}")

    # 7. 输出对比结果
    print("\n" + "=" * 70)
    print("📊 模型性能对比")
    print("=" * 70)
    print(f"{'模型':<20} {'MAE':<15} {'RMSE':<15}")
    print("-" * 50)
    print(f"{'XGBoost':<20} {xgb_mae:<15.4f} {xgb_rmse:<15.4f}")
    print(f"{'CNN-LSTM':<20} {dl_mae:<15.4f} {dl_rmse:<15.4f}")

    better_model = "XGBoost" if xgb_mae < dl_mae else "CNN-LSTM"
    print(f"\n🏆 表现更好的模型：{better_model}")