from typing import Any

import networkx as nx


def get_distance(G: nx.Graph, node1: Any, node2: Any) -> float:
    """
    获取两个节点之间的最短路径距离（米）

    Args:
        G: 路网图
        node1: 起始节点
        node2: 目标节点

    Returns:
        距离（米），如果无法到达则返回无穷大
    """
    try:
        return nx.shortest_path_length(G, node1, node2, weight='length')
    except nx.NetworkXNoPath:
        return float('inf')