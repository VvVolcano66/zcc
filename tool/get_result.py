from typing import List, Dict


def generate_assignment_report(assignment_details: List[Dict]) -> None:
    """
    生成分配报告

    Args:
        assignment_details: 分配详情列表
    """
    if not assignment_details:
        print("无分配记录")
        return

    print("\n" + "=" * 70)
    print("任务分配详细报告")
    print("=" * 70)

    total_reward = 0
    total_cost = 0
    total_profit = 0

    for detail in assignment_details:
        print(f"\n区域 {detail['region_id']}:")
        print(f"  工人 ID: {detail['wid']}")
        print(f"  任务 ID: {detail['task_id']}")
        print(f"  任务奖励：{detail['reward']:.2f} 元")
        print(f"  行驶距离：{detail['distance']:.2f} 米")
        print(f"  行驶成本：{detail['cost']:.2f} 元")
        print(f"  净利润：{detail['profit']:.2f} 元")

        total_reward += detail['reward']
        total_cost += detail['cost']
        total_profit += detail['profit']

    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    print(f"总分配任务数：{len(assignment_details)}")
    print(f"总奖励收入：{total_reward:.2f} 元")
    print(f"总行驶成本：{total_cost:.2f} 元")
    print(f"总利润：{total_profit:.2f} 元")
    if total_reward > 0:
        print(f"利润率：{(total_profit / total_reward) * 100:.2f}%")
    print("=" * 70)