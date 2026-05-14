import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

zone_start = 1
zone_end = 672
for zone_id in range(zone_start, zone_end + 1):
    # -------------------------- 1. 数据读取与预处理（完全保留原逻辑） --------------------------
    df = pd.read_csv(f'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\march_pred_result\march_pred_zone_{zone_id:03d}.csv')
    df_processed = df.copy()

    # 转换日期格式为标准格式
    df_processed['predict_date'] = pd.to_datetime(df_processed['predict_date'].astype(str), format='%Y%m%d')
    # 计算星期几（0=周一, 1=周二, ..., 6=周日）、周起始/结束（保留，不影响汇总）
    df_processed['weekday'] = df_processed['predict_date'].dt.dayofweek
    df_processed['week_start_monday'] = df_processed['predict_date'] - pd.to_timedelta(df_processed['weekday'], unit='D')
    df_processed['week_end_sunday'] = df_processed['week_start_monday'] + pd.to_timedelta(6, unit='D')

    # -------------------------- 2. 生成核心文件：grid_max_crowd_weekly_series.csv（完全保留原逻辑） --------------------------
    # 分组取最大ucnt并匹配对应usum，保证一一对应
    group_keys1 = ['final_grid_id', 'week_end_sunday', 'weekday', 'predict_hour', 'age', 'gender']
    max_ucnt_idx = df_processed.groupby(group_keys1)['ucnt_pred'].idxmax()
    max_ucnt_groups = df_processed.loc[max_ucnt_idx, group_keys1 + ['ucnt_pred', 'usum_pred']].reset_index(drop=True)

    # 二次分组取所有周的最大ucnt+对应usum
    group_keys2 = ['final_grid_id', 'weekday', 'predict_hour', 'age', 'gender']
    final_max_idx = max_ucnt_groups.groupby(group_keys2)['ucnt_pred'].idxmax()
    grid_max_weekly = max_ucnt_groups.loc[final_max_idx, :].reset_index(drop=True)

    # 整理排序，添加星期中文名称
    weekday_map = {0: '星期一', 1: '星期二', 2: '星期三', 3: '星期四', 4: '星期五', 5: '星期六', 6: '星期日'}
    grid_max_weekly['weekday_name'] = grid_max_weekly['weekday'].map(weekday_map)
    # 固定列顺序，保证可读性
    column_order = ['final_grid_id', 'weekday', 'weekday_name', 'predict_hour', 'age', 'gender', 'ucnt_pred', 'usum_pred']
    grid_max_result = grid_max_weekly[column_order].sort_values(
        by=['final_grid_id', 'weekday', 'predict_hour', 'age', 'gender']
    ).reset_index(drop=True)

    # -------------------------- 3. 生成汇总文件：grid_max_crowd_weekly_series_summary.csv（核心修改：分组维度） --------------------------
    # 汇总逻辑：每个网格 + 星期几（周一至周日） + 每个小时 → 统计ucnt/usum总和
    summary_group_keys = ['final_grid_id', 'weekday', 'predict_hour']  # 关键修改：替换为weekday（星期每一天）
    grid_summary = df_processed.groupby(summary_group_keys).agg(
        total_ucnt=('ucnt_pred', 'sum'),  # 该网格-该星期-该小时的ucnt总和（所有年龄/性别）
        total_usum=('usum_pred', 'sum')   # 对应维度下的usum总和（所有年龄/性别）
    ).reset_index()

    # 添加星期中文名称，便于阅读（和核心文件保持一致）
    grid_summary['weekday_name'] = grid_summary['weekday'].map(weekday_map)
    # 调整汇总文件列顺序，优先展示易读字段
    summary_col_order = ['final_grid_id', 'weekday', 'weekday_name', 'predict_hour', 'total_ucnt', 'total_usum']
    grid_summary = grid_summary[summary_col_order].sort_values(
        by=['final_grid_id', 'weekday', 'predict_hour']
    ).reset_index(drop=True)

    # -------------------------- 4. 结果保存（文件名按要求，编码utf-8-sig兼容Excel） --------------------------
    # 核心文件：含最大ucnt+对应usum（原逻辑不变）
    grid_max_result.to_csv(f'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\march_pred_result\processed\grid_max_crowd_weekly_series_{zone_id:03d}.csv', index=False, encoding='utf-8-sig')
    # 汇总文件：网格+星期每一天+小时的ucnt/usum总和（按要求命名）
    grid_summary.to_csv(f'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\march_pred_result\processed\grid_max_crowd_weekly_series_summary_{zone_id:03d}.csv', index=False, encoding='utf-8-sig')

    # -------------------------- 5. 结果验证统计（输出关键信息，确认数据正确性） --------------------------
    print("=== 核心文件：grid_max_crowd_weekly_series.csv 统计 ===")
    print(f"总记录数：{len(grid_max_result)}")
    print(f"网格数：{grid_max_result['final_grid_id'].nunique()}")
    print(f"列信息：{grid_max_result.columns.tolist()}")
    print("前5行示例：")
    print(grid_max_result.head(), '\n')

    print("=== 汇总文件：grid_max_crowd_weekly_series_summary.csv 统计 ===")
    print(f"总记录数：{len(grid_summary)}")
    print(f"网格数：{grid_summary['final_grid_id'].nunique()}")
    print(f"星期覆盖：{sorted(grid_summary['weekday'].unique())} → {[weekday_map[i] for i in sorted(grid_summary['weekday'].unique())]}")
    print(f"小时覆盖：{sorted(grid_summary['predict_hour'].unique())}（0-23小时）")
    print(f"列信息：{grid_summary.columns.tolist()}")
    print("前5行示例：")
    print(grid_summary.head(), '\n')

    print("=== 所有文件输出完成 ===")
    print(f"1. 核心文件：/mnt/grid_max_crowd_weekly_series_{zone_id:03d}.csv（最大ucnt+对应usum）")
    print(f"2. 汇总文件：/mnt/grid_max_crowd_weekly_series_summary_{zone_id:03d}.csv（网格+星期+小时的ucnt/usum总和）")
