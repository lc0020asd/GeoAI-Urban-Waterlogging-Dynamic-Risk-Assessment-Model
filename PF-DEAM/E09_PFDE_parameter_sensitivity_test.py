import pandas as pd
import numpy as np
import os
from typing import Dict
from tqdm import tqdm  # 进度条，方便查看批量读取/计算进度
import csv

# -------------------------- 1. 核心配置参数（无需修改，所有降雨等级自动遍历） --------------------------
# 降雨等级与β值映射表（严格匹配表15，按小雨→特大暴雨顺序）
BETA_MAP: Dict[str, float] = {
    "小雨": 1.00,
    "中雨": 0.80,
    "大雨": 0.80,
    "暴雨": 0.65,
    "大暴雨": 0.65,
    "特大暴雨": 0.50
}
# 权重系数α（人群数量更重要，取0.75）
ALPHA = 0.75
# summary文件序号范围（1到648）
ZONE_START = 1
ZONE_END = 648

# 数据文件基础路径（匹配你的实际路径，无需修改）
SUMMARY_BASE_PATH = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\march_pred_result\processed'
SUMMARY_FILE_PATTERN = 'grid_max_crowd_weekly_series_summary_{:03d}.csv'  # 03d补零适配1→001
BUILDING_V_PATH = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\buildingdata\building_v_sum.csv'
# 结果保存路径（在summary目录生成，自动带降雨等级标识）
OUTPUT_DIR = SUMMARY_BASE_PATH
# 结果文件命名模板（仅保留考虑建筑庇护，带降雨等级+全域标识）
OUTPUT_FILE_PATTERN = fr'全域人群暴露性计算结果_考虑建筑庇护_{{}}_1-648.csv'


# -------------------------- 2. 批量读取并合并1-648号summary文件 --------------------------
def batch_load_and_merge_summary() -> pd.DataFrame:
    """批量读取ZONE_START到ZONE_END的summary文件，合并为全域总数据"""
    df_list = []
    # 遍历1-648，按补零规则拼接文件名，带进度条
    for zone_id in tqdm(range(ZONE_START, ZONE_END + 1), desc="【1/4】批量读取summary文件"):
        file_name = SUMMARY_FILE_PATTERN.format(zone_id)
        file_path = os.path.join(SUMMARY_BASE_PATH, file_name)
        # 跳过不存在的文件（避免部分序号文件缺失报错）
        if not os.path.exists(file_path):
            print(f"⚠️  文件缺失，跳过：{file_name}")
            continue
        # 读取单文件并加入列表（utf-8-sig兼容Excel）
        df_single = pd.read_csv(file_path, encoding='utf-8-sig')
        df_list.append(df_single)
    # 校验是否读取到有效文件
    if not df_list:
        raise FileNotFoundError("❌ 未找到任何1-648号的summary文件，请检查文件路径和序号！")
    # 合并所有读取到的文件为全域数据，重置索引
    df_all_summary = pd.concat(df_list, ignore_index=True)
    # 基础去重（避免重复的「网格-星期-小时」记录，保证数据唯一性）
    df_all_summary = df_all_summary.drop_duplicates(
        subset=['final_grid_id', 'weekday', 'predict_hour'],
        keep='first'
    )
    # 输出合并统计信息
    print(f"✅ 【1/4】summary文件批量合并完成")
    print(
        f"   📊 有效文件数：{len(df_list)} 个 | 全域总记录数：{len(df_all_summary)} 条 | 全域网格数：{df_all_summary['final_grid_id'].nunique()} 个")
    # 校验核心字段是否存在
    assert 'total_ucnt' in df_all_summary.columns and 'total_usum' in df_all_summary.columns, \
        "❌ summary文件缺少total_ucnt/total_usum核心字段！"
    assert 'final_grid_id' in df_all_summary.columns, "❌ summary文件缺少final_grid_id（geohash7）字段！"
    return df_all_summary


# -------------------------- 3. 读取建筑容积数据并做匹配前预处理 --------------------------
def load_and_preprocess_building() -> pd.DataFrame:
    """读取建筑容积数据，统一geohash7列为final_grid_id，适配网格匹配"""
    print("\n" + "-" * 50)
    # 读取建筑容积数据
    df_building = pd.read_csv(BUILDING_V_PATH, encoding='utf-8-sig')
    # 统一geohash7列名（适配索引/列两种存储情况）
    if df_building.index.name == 'geohash7':
        df_building = df_building.reset_index()
    # 校验建筑数据核心字段
    assert 'geohash7' in df_building.columns and 'building_v' in df_building.columns, \
        "❌ 建筑容积数据缺少geohash7/building_v核心字段！"
    # 重命名为final_grid_id，和summary数据统一匹配键
    df_building.rename(columns={'geohash7': 'final_grid_id'}, inplace=True)
    # 去重（同一网格仅保留一条容积记录，避免重复匹配）
    df_building = df_building.drop_duplicates(subset=['final_grid_id'], keep='first')
    # 输出建筑数据统计信息
    print(f"✅ 【2/4】建筑容积数据读取预处理完成")
    print(f"   📊 建筑数据网格数：{df_building['final_grid_id'].nunique()} 个")
    return df_building


# -------------------------- 4. 全域数据融合 + 全域极值计算（仅计算1次，所有等级复用） --------------------------
def merge_data_and_calc_global_max(df_all_summary: pd.DataFrame, df_building: pd.DataFrame) -> pd.DataFrame:
    """融合全域summary和建筑容积数据，计算全域全时段的ucnt_max/usum_max/V_buildingmax（所有降雨等级复用该极值）"""
    print("\n" + "-" * 50)
    # 左连接融合数据（保留所有人群网格，无建筑数据的网格容积设为0，代表无庇护能力）
    df_merged = pd.merge(
        df_all_summary,
        df_building[['final_grid_id', 'building_v']],
        on='final_grid_id',
        how='left'
    )
    # 无建筑容积的网格填充0，避免后续计算报错
    df_merged['building_v'] = df_merged['building_v'].fillna(0)

    # 计算全域极值（公式所需：全区域全时段的最大值，仅计算1次，所有降雨等级复用）
    UCNT_MAX = df_merged['total_ucnt'].max()
    USUM_MAX = df_merged['total_usum'].max()
    # 避免建筑容积最大值为0导致分母为0，兜底设为1
    V_BUILDING_MAX = df_merged['building_v'].max() if df_merged['building_v'].max() > 0 else 1

    # 全域极值存入列，方便后续逐行计算
    df_merged['ucnt_max'] = UCNT_MAX
    df_merged['usum_max'] = USUM_MAX
    df_merged['v_building_max'] = V_BUILDING_MAX

    # 输出融合后统计信息
    print(f"✅ 【3/4】全域数据融合+全域极值计算完成")
    print(
        f"   📊 融合后总记录数：{len(df_merged)} 条 | 无建筑容积网格数：{len(df_merged[df_merged['building_v'] == 0])} 个")
    print(f"   ⚖️  全域极值（所有降雨等级复用）：")
    print(f"      → ucnt_max(全域最大聚集人数)：{UCNT_MAX}")
    print(f"      → usum_max(全域最大访问次数)：{USUM_MAX}")
    print(f"      → V_buildingmax(全域最大建筑容积)：{V_BUILDING_MAX:.2f}")
    return df_merged


# -------------------------- 5. 给定beta计算暴露性均值（敏感性分析专用，不归一化） --------------------------
def _calc_exposure_mean(df_merged: pd.DataFrame, alpha: float, beta: float) -> float:
    """
    给定beta值，计算暴露性e_people的全域均值（不归一化，保留原始量级）
    """
    df = df_merged.copy()

    # 步骤1：人群因子
    df['norm_ucnt'] = df['total_ucnt'] / df['ucnt_max']
    df['norm_usum'] = df['total_usum'] / df['usum_max']
    df['human_factor'] = alpha * df['norm_ucnt'] + (1 - alpha) * df['norm_usum']

    # 步骤2：建筑庇护因子
    df['v_ratio'] = df['building_v'] / df['v_building_max']
    df['exp_term'] = np.exp(-df['v_ratio'])
    df['building_factor'] = 1 - beta * (1 - df['exp_term'])

    # 步骤3：暴露性（未归一化）
    df['e_people'] = df['human_factor'] * df['building_factor']

    return float(df['e_people'].mean())


# -------------------------- 6. 参数单因子敏感性分析（BETA_MAP） --------------------------
def run_sensitivity_analysis(df_merged: pd.DataFrame, alpha: float):
    """
    针对BETA_MAP中各降雨情境的beta值进行单因子敏感性分析。
    每个情境分别取90%和110%（上限截断为1.0），以e_people全域均值为结果指标。
    敏感性指数 S = (ΔH / H_base) / Δβ
    """
    sens_records = []

    print("\n" + "=" * 70)
    print("开始 BETA_MAP 参数单因子敏感性分析")
    print("=" * 70)
    print("敏感性指数定义：S = (ΔH / H_base) / Δβ")
    print("其中 Δβ = β_110% − β_90%，ΔH = H_110% − H_90%")
    print("=" * 70)

    for rain_level, base_beta in BETA_MAP.items():
        print(f"\n{'=' * 60}")
        print(f"【降雨情境】{rain_level}（基准 β = {base_beta:.2f}）")
        print(f"{'=' * 60}")

        # 计算扰动值（110%上限截断为1.0）
        beta_90 = base_beta * 0.9
        beta_110 = min(base_beta * 1.1, 1.0)

        clipped_flag = ""
        if base_beta * 1.1 > 1.0:
            clipped_flag = "  ← 已截断（原1.1倍>1.0）"

        print(f"   β_90%  = {beta_90:.4f}")
        print(f"   β_110% = {beta_110:.4f}{clipped_flag}")

        # 基准计算
        h_base = _calc_exposure_mean(df_merged, alpha, base_beta)
        print(f"   基准暴露性均值 H_base = {h_base:.6f}")

        # 90%计算
        h_90 = _calc_exposure_mean(df_merged, alpha, beta_90)
        print(f"   β-90%  暴露性均值 = {h_90:.6f}")

        # 110%计算
        h_110 = _calc_exposure_mean(df_merged, alpha, beta_110)
        print(f"   β-110% 暴露性均值 = {h_110:.6f}")

        # 敏感性计算
        delta_beta = (beta_110 - beta_90)/base_beta
        if abs(h_base) > 1e-9 and abs(delta_beta) > 1e-9:
            sens = ((h_110 - h_90) / h_base) / delta_beta
        else:
            sens = 0.0
            print("⚠️  基准H接近0或参数变化为0，敏感性设为0")

        print(f"\n📊 {rain_level} 敏感性指数：S_β = {sens:.4f}")

        sens_records.append([
            rain_level,
            round(base_beta, 4),
            round(beta_90, 4),
            round(beta_110, 4),
            round(h_base, 6),
            round(h_90, 6),
            round(h_110, 6),
            round(sens, 4)
        ])

    # 输出汇总表
    print("\n" + "=" * 90)
    print("BETA_MAP 参数单因子敏感性分析汇总结果")
    print("=" * 90)
    header = (f"{'情境':<8} {'β_base':<8} {'β_90%':<8} {'β_110%':<8} "
              f"{'H_base':<12} {'H_90%':<12} {'H_110%':<12} {'S_β':<10}")
    print(header)
    print("-" * 90)
    for rec in sens_records:
        line = (f"{rec[0]:<8} {rec[1]:<8.4f} {rec[2]:<8.4f} {rec[3]:<8.4f} "
                f"{rec[4]:<12.6f} {rec[5]:<12.6f} {rec[6]:<12.6f} {rec[7]:<10.4f}")
        print(line)
    print("=" * 90)

    # 保存CSV
    csv_path = os.path.join(OUTPUT_DIR, 'sensitivity_analysis_beta.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '降雨情境', 'beta_base', 'beta_90%', 'beta_110%(实际值)',
            'H_base', 'H_90%', 'H_110%', 'S_beta'
        ])
        writer.writerows(sens_records)
    print(f"\n✅ 敏感性分析结果已保存至：{csv_path}")

    return sens_records


# -------------------------- 7. 多降雨等级批量计算人群暴露性（仅考虑建筑庇护） --------------------------
def batch_calculate_exposure(df_merged: pd.DataFrame, alpha: float):
    """自动遍历所有降雨等级，批量计算考虑建筑庇护的人群暴露性，仅保存对应结果"""
    print("\n" + "-" * 50)
    # 遍历所有降雨等级，带进度条
    for rain_level, beta in tqdm(BETA_MAP.items(), desc="【4/4】多降雨等级暴露性批量计算"):
        df_calc = df_merged.copy()
        # 步骤1：计算人群因子（归一化人数+访问次数，按α加权，公式核心项）
        df_calc['norm_ucnt'] = df_calc['total_ucnt'] / df_calc['ucnt_max']  # 人数归一化（0-1）
        df_calc['norm_usum'] = df_calc['total_usum'] / df_calc['usum_max']  # 访问次数归一化（0-1）
        df_calc['human_factor'] = alpha * df_calc['norm_ucnt'] + (1 - alpha) * df_calc['norm_usum']

        # 步骤2：计算建筑庇护因子（严格按公式实现，处理V=0特殊情况）
        df_calc['v_ratio'] = df_calc['building_v'] / df_calc['v_building_max']  # 容积归一化
        df_calc['exp_term'] = np.exp(-df_calc['v_ratio'])  # 指数项 e^(-V/Vmax)
        df_calc['building_factor'] = 1 - beta * (1 - df_calc['exp_term'])  # 建筑庇护因子

        # 步骤3：计算考虑建筑庇护的人群暴露性（最终结果，公式完整实现）
        df_calc['e_people'] = df_calc['human_factor'] * df_calc['building_factor']

        # 步骤4：暴露性结果归一化到[0,1]区间（便于跨等级/网格对比，值越大暴露性越高）
        e_min = df_calc['e_people'].min()
        e_max = df_calc['e_people'].max()
        if e_max - e_min > 1e-9:
            df_calc['e_people_norm'] = (df_calc['e_people'] - e_min) / (e_max - e_min)
        else:
            df_calc['e_people_norm'] = 0.0

        # 步骤5：整理最终结果列（仅保留核心字段，剔除中间计算列，简洁易读）
        core_columns = [
            'final_grid_id', 'weekday', 'predict_hour', 'total_ucnt', 'total_usum', 'building_v',
            'e_people', 'e_people_norm'
        ]
        # 兼容部分旧summary文件有weekday_name的情况，自动添加
        if 'weekday_name' in df_calc.columns:
            core_columns.insert(2, 'weekday_name')
        # 过滤结果列，保证无缺失
        core_columns = [col for col in core_columns if col in df_calc.columns]
        df_result = df_calc[core_columns].copy()

        # 步骤6：保存当前降雨等级的结果文件
        output_file = OUTPUT_FILE_PATTERN.format(rain_level)
        df_result.to_csv(
            os.path.join(OUTPUT_DIR, output_file),
            index=False,
            encoding='utf-8-sig'  # 兼容Excel打开无乱码
        )
        # 输出当前等级计算完成信息
        print(f"   ✅ {rain_level}（β={beta}）计算完成，结果文件：{output_file}")
        print(
            f"      📊 暴露性范围：{df_result['e_people'].min():.4f} ~ {df_result['e_people'].max():.4f} | 归一化范围：{df_result['e_people_norm'].min():.4f} ~ {df_result['e_people_norm'].max():.4f}")


# -------------------------- 主函数：执行全域多降雨等级批量计算全流程 --------------------------
if __name__ == "__main__":
    try:
        print("=" * 70)
        print("人群暴露性计算系统")
        print("=" * 70)
        print("\n请选择运行模式：")
        print("1. 批量计算多降雨等级暴露性（小雨→特大暴雨）")
        print("2. BETA_MAP 参数单因子敏感性分析")

        choice = input("\n请输入选择 (1-2): ").strip()

        # 公共前置步骤：读取数据并融合
        print("\n" + "=" * 70)
        print("【公共步骤】数据加载与融合")
        print("=" * 70)
        df_all_summary = batch_load_and_merge_summary()
        df_building = load_and_preprocess_building()
        df_merged = merge_data_and_calc_global_max(df_all_summary, df_building)

        if choice == "1":
            # 模式1：多降雨等级批量计算
            batch_calculate_exposure(df_merged, ALPHA)
            print("\n" + "=" * 60)
            print("🎉 【全流程完成】小雨→特大暴雨全等级暴露性计算完成！")
            print(f"📁 所有结果文件已保存至：{OUTPUT_DIR}")
            print("📌 结果文件说明：e_people=原始暴露性 | e_people_norm=归一化暴露性（0-1，值越大暴露性越高）")
            print("=" * 60)

        elif choice == "2":
            # 模式2：敏感性分析
            run_sensitivity_analysis(df_merged, ALPHA)
            print("\n" + "=" * 60)
            print("🎉 【敏感性分析完成】BETA_MAP 单因子敏感性分析结束！")
            print("=" * 60)

        else:
            print("无效选择，请输入1或2")

    except Exception as e:
        # 捕获异常并打印，方便排查问题
        print(f"\n❌ 计算流程失败：{str(e)}")
        raise  # 抛出异常，终止程序