import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import time

# 导入你写的模块
from data_loader import get_real_road_network
from map_algorithms import run_kmeans_baseline, run_rcc_algorithm, find_region_centers
import config  # <--- 引入你的配置文件


def draw_real_map(G, nodes, coords, partition, centers=None, title="Real Map Visualization"):
    """专门针对真实经纬度路网的可视化函数 (已修复绘图 Bug)"""
    plt.figure(figsize=(20, 20), dpi=200)
    ax = plt.gca()

    # 构建包含全图所有节点坐标的完整字典
    pos = {node: (coords[i][0], coords[i][1]) for i, node in enumerate(nodes)}

    # 1. 绘制道路 (Edges)
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='#AAAAAA', ax=ax, width=0.5)

    # [修复警告] 使用 plt.get_cmap 替代废弃的 cm.get_cmap
    try:
        cmap = plt.get_cmap(config.COLOR_MAP, config.NUM_ZONES)
    except Exception:
        cmap = plt.get_cmap('tab20', config.NUM_ZONES)

    # 2. 绘制节点
    for rid in set(partition.values()):
        region_nodes = [n for n in partition if partition[n] == rid]
        color = cmap(rid % config.NUM_ZONES)

        # [核心修复] 传入完整的 pos，并使用 nodelist 指定只画当前区域的节点
        nx.draw_networkx_nodes(G, pos, nodelist=region_nodes, node_size=10,
                               node_color=[color], ax=ax)

    # 3. 绘制发货中心 (Centers)
    if centers:
        for rid, center_node in centers.items():
            if center_node in pos:
                # [核心修复] 同样使用 nodelist 来绘制中心点
                nx.draw_networkx_nodes(G, pos, nodelist=[center_node], node_size=600,
                                       node_color='red', node_shape='*',
                                       edgecolors='white', linewidths=1.5, ax=ax)
                ax.text(pos[center_node][0], pos[center_node][1],
                        s=f" Hub-{rid}", color='black', fontweight='bold',
                        fontsize=14, ha='left', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    plt.title(title, fontsize=24, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"🚀 正在初始化 {config.CITY_NAME} 的物流网络规划任务...")
    print(f"   - 中心坐标: {config.CHENGDU_CENTER}")
    print(f"   - 覆盖半径: {config.DOWNLOAD_DIST} 米")
    print(f"   - 规划片区: {config.NUM_ZONES} 个")
    print("=" * 50)

    total_start_time = time.time()

    # ==========================================
    # 1. 加载数据
    # ==========================================
    t0 = time.time()
    G, coords, nodes = get_real_road_network(config.CHENGDU_CENTER, dist=config.DOWNLOAD_DIST)
    print(f"[耗时] 数据加载完成，用时: {time.time() - t0:.2f} 秒")

    # ==========================================
    # 💡 新增：算法结果缓存逻辑
    # ==========================================
    # 动态生成缓存文件名，把关键参数带上，防止改了参数读错缓存
    algo_cache_dir = r"D:\biyelunwen\tool\cache"
    os.makedirs(algo_cache_dir, exist_ok=True)
    algo_cache_file = os.path.join(
        algo_cache_dir,
        f"algo_results_{config.CITY_NAME}_k{config.NUM_ZONES}_dist{config.DOWNLOAD_DIST}.pkl"
    )

    if os.path.exists(algo_cache_file):
        print(f"\n✅ [Cache] 发现算法结果缓存: {algo_cache_file}")
        print(">> 正在直接加载分区和中心点，跳过计算...")
        with open(algo_cache_file, 'rb') as f:
            rcc_partition, centers = pickle.load(f)
    else:
        print("\n⚠️ 未找到当前参数配置的算法缓存，开始执行计算...")

        # ==========================================
        # 2. 运行 Baseline (K-means)
        # ==========================================
        t1 = time.time()
        kmeans_partition = run_kmeans_baseline(coords, nodes, k=config.NUM_ZONES)
        print(f"[耗时] K-means 聚类完成，用时: {time.time() - t1:.2f} 秒")

        # ==========================================
        # 3. 运行 RCC 连通性修复
        # ==========================================
        t2 = time.time()
        rcc_partition = run_rcc_algorithm(G, kmeans_partition, k=config.NUM_ZONES)
        print(f"[耗时] RCC 连通性修复完成，用时: {time.time() - t2:.2f} 秒")

        # ==========================================
        # 4. 寻找最优发货中心
        # ==========================================
        t3 = time.time()
        centers = find_region_centers(G, rcc_partition, weight='length')
        print(f"[耗时] 中心点选址完成，用时: {time.time() - t3:.2f} 秒")

        # 将计算结果保存到缓存
        print(f">> 正在将算法结果写入缓存...")
        with open(algo_cache_file, 'wb') as f:
            pickle.dump((rcc_partition, centers), f)
        print("✅ 算法结果缓存保存成功！")

    print("=" * 50)
    print(f"🎉 全部准备就绪！总用时: {time.time() - total_start_time:.2f} 秒")

    # 5. 可视化最终结果
    # 注意：确保这里调用的是刚才修复了报错的 draw_real_map
    draw_real_map(G, nodes, coords, rcc_partition, centers=centers,
                  title=f"{config.CITY_NAME} Logistics Hubs ({config.NUM_ZONES} Zones)")