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
    加入了 KDTree 空间查询，彻底解决“纯飞地”（完全与主路网断开的孤岛）导致算法失效的问题。
    """
    print(f">> [Proposed] Running RCC (Connectivity Repair)...")

    # 复制一份初始分区
    final_partition = base_partition.copy()
    region_ids = set(base_partition.values())

    # 🚀 性能优化与飞地处理准备：预先构建全图的 KDTree
    node_list = list(G.nodes())
    node_coords = []
    for n in node_list:
        data = G.nodes[n]
        # 兼容处理 x/y 和 lon/lat
        x = data.get('x', data.get('lon'))
        y = data.get('y', data.get('lat'))
        node_coords.append([x, y])

    tree = KDTree(node_coords)

    # 遍历每个区域，检查它是不是断开的
    for rid in region_ids:
        # 1. 找出该区域包含的所有节点
        region_nodes = [n for n in final_partition if final_partition[n] == rid]

        # 2. 在原图中构建该区域的子图
        subgraph = G.subgraph(region_nodes)

        # 3. 检查连通分量 (Connected Components)
        if not nx.is_connected(subgraph):
            # 获取所有分块，按大小排序 (最大的那块是主大陆，小的都是岛屿)
            components = list(nx.connected_components(subgraph))
            components.sort(key=len, reverse=True)

            main_land = components[0]
            islands = components[1:]

            # 处理每一个孤岛
            for island in islands:
                target_rid = None

                # ==== 策略 A：优先寻找拓扑邻居 (路网连通性) ====
                for node in island:
                    neighbors = list(G.neighbors(node))
                    for nbr in neighbors:
                        nbr_rid = final_partition[nbr]
                        # 找到邻居属于别的区域，准备合并过去
                        if nbr_rid != rid:
                            target_rid = nbr_rid
                            break
                    if target_rid is not None:
                        break

                # ==== 策略 B：如果拓扑断开 (纯飞地)，使用 KDTree 寻找空间最近的异区节点 ====
                if target_rid is None:
                    # 提取孤岛节点的坐标
                    island_coords = []
                    island_nodes_list = list(island)
                    for node in island_nodes_list:
                        data = G.nodes[node]
                        island_coords.append([
                            data.get('x', data.get('lon')),
                            data.get('y', data.get('lat'))
                        ])

                    # 查询距离最近的k个节点 (设为20以确保能跳出孤岛本身找到外界节点)
                    k_neighbors = min(20, len(node_list))
                    distances, indices = tree.query(island_coords, k=k_neighbors)

                    found_external = False
                    # 遍历查询结果，寻找最近的非当前区域的节点
                    for i in range(len(island_nodes_list)):
                        # 处理单节点的维度兼容性
                        idx_array = np.atleast_1d(indices[i])
                        for j in range(len(idx_array)):
                            candidate_node = node_list[idx_array[j]]
                            candidate_rid = final_partition[candidate_node]

                            # 找到了离飞地空间直线距离最近，且不属于自身区域的节点
                            if candidate_rid != rid:
                                target_rid = candidate_rid
                                found_external = True
                                break
                        if found_external:
                            break

                # 统一执行合并：将整个孤岛划入找到的 target_rid
                if target_rid is not None:
                    for node in island:
                        final_partition[node] = target_rid

    return final_partition


def find_region_centers(G, partition, weight='length'):
    """
    在每个分区中寻找最优发货中心（Hub / Depot）
    策略：寻找网络“重心”（Barycenter），即最小化该点到区域内所有其他节点的路径总和。
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
        if not nx.is_connected(subgraph):
            print(f"  [Warning] Region {rid} has disconnected parts. Searching center in the largest component.")
            largest_cc = max(nx.connected_components(subgraph), key=len)
            subgraph = G.subgraph(largest_cc)

        try:
            # nx.barycenter 返回距离总和最小的节点列表
            barycenters = nx.barycenter(subgraph, weight=weight)
            # 我们只需要选取其中一个作为发货中心即可
            best_center_node = barycenters[0]
            region_centers[rid] = best_center_node

        except nx.NetworkXException as e:
            print(f"  [Error] Could not find center for Region {rid}: {e}")
            region_centers[rid] = nodes[0]  # 兜底策略：随便选一个点

    return region_centers