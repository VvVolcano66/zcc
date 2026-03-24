import osmnx as ox
import networkx as nx
import numpy as np
import warnings
import os
import pickle

CACHE_DIR = r"D:\biyelunwen\tool\cache"


def _build_cache_file(center_point, dist):
    lat, lon = center_point
    safe_lat = str(lat).replace('.', '_')
    safe_lon = str(lon).replace('.', '_')
    return os.path.join(
        CACHE_DIR,
        f"road_network_v2_{safe_lat}_{safe_lon}_dist{dist}.pkl"
    )


def get_real_road_network(center_point, dist=2000):
    """
    带缓存的路网加载器
    """
    cache_file = _build_cache_file(center_point, dist)

    # --- 1. 尝试从本地加载 ---
    if os.path.exists(cache_file):
        print(f"✅ [Cache] 发现本地缓存: {cache_file}")
        print(">> 正在直接加载路网，跳过下载...")
        try:
            with open(cache_file, 'rb') as f:
                G_simple, coords, nodes = pickle.load(f)
            print(f"   - 加载完成: {len(nodes)} 节点, {len(G_simple.edges)} 边")
            return G_simple, coords, nodes
        except Exception as e:
            print(f"⚠️ 缓存文件损坏 ({e})，准备重新下载...")

    # --- 2. 如果没有缓存，执行下载 ---
    print(f"正在从 OSM 下载坐标 {center_point} 附近 {dist}米 的路网...")

    ox.settings.use_cache = True
    ox.settings.log_console = True
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        # 下载
        G_raw = ox.graph_from_point(center_point, dist=dist, network_type='drive')
        # 转无向图
        G_simple = nx.Graph(G_raw)

        print(f"路网处理完成: {len(G_simple.nodes)} 节点")

        # 提取坐标
        nodes = list(G_simple.nodes())
        coords = []
        for node in nodes:
            data = G_simple.nodes[node]
            if 'x' in data and 'y' in data:
                coords.append([data['x'], data['y']])
            else:
                coords.append([data['lon'], data['lat']])

        coords = np.array(coords)

        # --- 3. 保存到本地缓存 ---
        print(f">> 正在保存路网到缓存文件: {cache_file} ...")
        # 确保目录存在
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        with open(cache_file, 'wb') as f:
            # 打包保存 G, coords, nodes
            pickle.dump((G_simple, coords, nodes), f)
        print("✅ 保存成功！下次运行将秒开。")

        return G_simple, coords, nodes

    except Exception as e:
        print("\n!!! 下载或处理出错 !!!")
        print(e)
        raise e
