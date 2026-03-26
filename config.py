# config.py

# --- 模拟与算法参数 ---
NUM_ZONES = 5        # 分区数量 (论文中的 k_n)
CITY_NAME = "Chengdu"

# --- 地理位置参数 ---
# 成都市中心坐标 (用于 OSMnx 下载)
# 对应论文图 3 的大致中心位置 (纬度, 经度)
CHENGDU_CENTER = (30.67, 104.06)

# 下载半径 (米)
# 之前的 2000 米可能略大，如果为了测试速度可以改小，比如 1000
DOWNLOAD_DIST = 5000

# --- 绘图配置 ---
# 颜色映射风格
COLOR_MAP = 'tab10'
# --- 配送经济学与物理学参数设定 ---

# --- 预测模块参数 ---
# 将需求预测网格与多中心分区数量解耦，避免修改 NUM_ZONES 时静默影响预测实验。
PRED_GRID_SIZE = (5, 5)

# 1. 物理参数
WORKER_SPEED_KMH = 18.0               # 骑手平均行驶速度 (18 km/h，考虑了红绿灯和市区路况)
WORKER_SPEED_MS = WORKER_SPEED_KMH / 3.6 # 换算为米/秒 (约 5.0 m/s)

# 2. 经济参数 (单位: 元 RMB)
TASK_BASE_REWARD = 8.0                # 每成功送达一单的固定奖励 (8元/单)
TRAVEL_COST_PER_KM = 0.5              # 骑手行驶成本 (0.5元/公里，包含电费、车辆折旧、体力成本折算)
TRAVEL_COST_PER_METER = TRAVEL_COST_PER_KM / 1000.0 # 换算为每米成本 (0.0005元/米)

# 3. 惩罚参数 (用于高级算法中的成本核算)
DELAY_PENALTY = 5.0                   # 订单超时/无法分配的系统声誉损失惩罚 (5元/单)
TASK_EXPIRE_MINUTES = 10

# 4. 在线模拟实验配置
EXPERIMENT_TEST_DATE = "2016-10-31"   # 默认实验日期
EXPERIMENT_START_HOUR = 7             # 默认实验开始时间（小时）
EXPERIMENT_END_HOUR = 9               # 默认实验结束时间（小时）
EXPERIMENT_TIME_SLOT_MINUTES = 15     # 默认时间槽长度（分钟）
WORKER_INIT_PREP_MINUTES = 5          # 初始化工人位置时回看历史轨迹的分钟数

# 5. 预测驱动预调度配置
DISPATCH_PRED_SEQ_LEN = 8
DISPATCH_PRED_PRE_LEN = 1
DISPATCH_PRED_VAL_DAYS = 2
BSTGCNET_DISPATCH_MAX_EPOCHS = 300
BSTGCNET_DISPATCH_PATIENCE = 50
MCTGNET_DISPATCH_MAX_EPOCHS = 3000
MCTGNET_DISPATCH_PATIENCE = 150
MCTGNET_DISPATCH_LR = 0.0005
MCTGNET_DISPATCH_LOG_INTERVAL = 20
MAX_TASKS_PER_WORKER = 4
PREDISPATCH_BACKLOG_WEIGHT = 1.0
PREDISPATCH_MIN_BUFFER_WORKERS = 3
PREDISPATCH_RESERVE_RATIO = 0.15
PREDISPATCH_MAX_SHARE_PER_DONOR = 0.35
PREDISPATCH_MAX_DISTANCE_KM = 8.0
GAME_DISPATCH_FAIRNESS_WEIGHT = 0.5
GAME_DISPATCH_DISTANCE_PENALTY = 0.015
GAME_DISPATCH_DONOR_MAX_UTILITY_DROP = 0.04
GAME_DISPATCH_RECEIVER_MIN_GAIN = 0.01
GAME_DISPATCH_MAX_ITERATIONS = 120
