import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from collections import defaultdict


def run_kmeans_baseline(coords, nodes, k):
    """
    [Baseline Algorithm] K-means++
    只根据经纬度坐标进行聚类，不考虑路网连通性。
    缺点：容易产生"飞地" (Disconneted Components)。
    """
    print(f">> [Baseline] Running K-means++ (k={k})...")

    # 1. 运行 K-means
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)

    # 2. 转为字典格式 {node_id: region_id}
    partition = {}
    for i, node in enumerate(nodes):
        partition[node] = labels[i]

    return partition


def run_rcc_algorithm(G, base_partition, k):
    """
    [Proposed Algorithm] RCC (Region Clustering based on Connectivity)
    复现论文核心逻辑：修复 K-means 产生的非连通区域。
    """
    print(f">> [Proposed] Running RCC (Connectivity Repair)...")

    # 复制一份初始分区
    final_partition = base_partition.copy()

    # 获取所有的 Region ID
    region_ids = set(base_partition.values())

    # 遍历每个区域，检查它是不是断开的
    for rid in region_ids:
        # 1. 找出该区域包含的所有节点
        region_nodes = [n for n in final_partition if final_partition[n] == rid]

        # 2. 在原图中构建该区域的子图
        subgraph = G.subgraph(region_nodes)

        # 3. 检查连通分量 (Connected Components)
        # 如果一个区域被分成了好几块，components 会大于 1
        if not nx.is_connected(subgraph):
            # 获取所有分块，按大小排序 (最大的那块是主大陆，小的都是岛屿)
            components = list(nx.connected_components(subgraph))
            components.sort(key=len, reverse=True)

            # 主大陆保留，剩下的“孤岛”需要重新分配
            main_land = components[0]
            islands = components[1:]

            # 处理每一个孤岛
            for island in islands:
                for node in island:
                    # 策略：找这个孤岛节点在原图中最近的邻居
                    # 如果邻居属于别的连通区域，就把它归顺过去
                    neighbors = list(G.neighbors(node))
                    for nbr in neighbors:
                        nbr_rid = final_partition[nbr]
                        # 如果邻居不在这个破碎的区域里，就合并过去
                        if nbr_rid != rid:
                            final_partition[node] = nbr_rid
                            break

    return final_partition


def find_region_centers(G, partition, weight='length'):
    """
    在每个分区中寻找最优发货中心（Hub / Depot）
    策略：寻找网络“重心”（Barycenter），即最小化该点到区域内所有其他节点的路径总和。

    参数:
    G: networkx 图对象 (你的路网)
    partition: dict, 格式为 {node_id: region_id}
    weight: str, 边权重的字段名，比如道路的真实长度 'length' 或通行时间 'travel_time'。
            如果图没有权重，可以设为 None，此时将计算跳数(hops)。
    """
    print(">> Finding optimal logistics centers for each region...")

    region_centers = {}

    # 1. 按 region_id 将节点归类
    regions = {}
    for node, rid in partition.items():
        if rid not in regions:
            regions[rid] = []
        regions[rid].append(node)

    # 2. 遍历每个区域，寻找最优中心点
    for rid, nodes in regions.items():
        # 提取该区域的局部路网子图
        subgraph = G.subgraph(nodes)

        # 安全检查：确保子图是连通的（经过你之前的 RCC 处理，这里通常已经是连通的）
        # 如果由于某种极端原因存在断点，我们只在最大的连通块里找中心
        if not nx.is_connected(subgraph):
            print(f"  [Warning] Region {rid} has disconnected parts. Searching center in the largest component.")
            largest_cc = max(nx.connected_components(subgraph), key=len)
            subgraph = G.subgraph(largest_cc)

        try:
            # nx.barycenter 返回距离总和最小的节点列表（可能存在多个并列第一的节点）
            barycenters = nx.barycenter(subgraph, weight=weight)
            # 我们只需要选取其中一个作为发货中心即可
            best_center_node = barycenters[0]
            region_centers[rid] = best_center_node

            # 如果你是做应急调度，可以把上面两行替换为下面这行：
            # best_center_node = nx.center(subgraph, weight=weight)[0]

        except nx.NetworkXException as e:
            print(f"  [Error] Could not find center for Region {rid}: {e}")
            region_centers[rid] = nodes[0]  # 兜底策略：随便选一个点

    return region_centers
