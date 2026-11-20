import re
from collections import defaultdict


def _format_metric_row(label, metrics):
    return (f"{label:<20} {metrics['num']:<10} {metrics['acc']:<10.4f} "
            f"{metrics['osr']:<10.4f} {metrics['ne']:<10.4f} "
            f"{metrics['spl']:<10.4f} {metrics['spl_num']:<10.4f}")


def print_results(results):
    """Print aggregate, seen/unseen, and object-size metrics."""
    total = results['total_results']
    separator_width = 80

    print(f"{'Category':<20} {'Num':<10} {'Acc':<10} {'OSR':<10} {'NE':<10} "
          f"{'SPL':<10} {'SPL NUM':<10}")
    print("-" * separator_width)

    print(_format_metric_row('Total', total))

    print("\n" + "=" * separator_width)
    print("Seen Metric:")
    print("-" * separator_width)
    for key, metrics in results['seen_metric'].items():
        print(_format_metric_row(f"  {key}", metrics))

    print("\n" + "=" * separator_width)
    print("Size Metric:")
    print("-" * separator_width)
    for key, metrics in results['size_metric'].items():
        print(_format_metric_row(f"  {key}", metrics))

def process_results(env_result_items_dict):
    default_seen = defaultdict(list)
    default_size = defaultdict(list)
    
    total_items = [item  for env,items in env_result_items_dict.items() for item in items]
    def cal_metric(total_rows):
        spl = 0
        spl_num = 0
        if len(total_rows) > 0:
            for row in total_rows:
                p_i = len(row['gen_pose_list'])
                if row.get('action',None) is None:
                    continue
                l_i = len(row['action'])
                success = row['acc']
                spl_num+=1
                if success:
                    spl += l_i / max(p_i, l_i)
        spl = spl / len(total_rows) if len(total_rows) > 0 else 0     
        metrics = {
            'num' : len(total_rows),
            'acc' : sum([cur['acc'] for cur in total_rows]) / len(total_rows) if len(total_rows) > 0 else 0,
            'osr' : sum([cur['osr'] for cur in total_rows]) / len(total_rows) if len(total_rows) > 0 else 0,
            'ne'  : sum([cur['ne'] for cur in total_rows]) / len(total_rows) if len(total_rows) > 0 else 0,
            'spl': spl,
            'spl_num':spl_num,
        }
        return metrics
    total_results = cal_metric(total_items)
    for row in total_items:
        size_str = row['size']
        size = re.search(r'^\s*([a-zA-Z]+)', size_str.strip()).group(1)
        is_seen = row["used-in-train"]
        default_seen[is_seen].append(row)
        default_size[size].append(row)

    seen_metric = {
        k:cal_metric(item_list) for k,item_list in default_seen.items()
    }
    size_metric = {
        k:cal_metric(item_list) for k,item_list in default_size.items()
    }
    return dict(
        total_results = total_results,
        seen_metric = seen_metric,
        size_metric = size_metric
    )
