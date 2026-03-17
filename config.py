# config.py

# --- 模拟与算法参数 ---
NUM_ZONES = 10        # 分区数量 (论文中的 k_n)
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

# 1. 物理参数
WORKER_SPEED_KMH = 18.0               # 骑手平均行驶速度 (18 km/h，考虑了红绿灯和市区路况)
WORKER_SPEED_MS = WORKER_SPEED_KMH / 3.6 # 换算为米/秒 (约 5.0 m/s)

# 2. 经济参数 (单位: 元 RMB)
TASK_BASE_REWARD = 8.0                # 每成功送达一单的固定奖励 (8元/单)
TRAVEL_COST_PER_KM = 0.5              # 骑手行驶成本 (0.5元/公里，包含电费、车辆折旧、体力成本折算)
TRAVEL_COST_PER_METER = TRAVEL_COST_PER_KM / 1000.0 # 换算为每米成本 (0.0005元/米)

# 3. 惩罚参数 (用于高级算法中的成本核算)
DELAY_PENALTY = 5.0                   # 订单超时/无法分配的系统声誉损失惩罚 (5元/单)