import pandas as pd
import numpy as np
import os
from typing import Dict
from tqdm import tqdm  # 进度条，方便查看批量读取/计算进度

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


# -------------------------- 5. 多降雨等级批量计算人群暴露性（仅考虑建筑庇护） --------------------------
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
        df_calc['e_people_norm'] = (df_calc['e_people'] - df_calc['e_people'].min()) / \
                                   (df_calc['e_people'].max() - df_calc['e_people'].min())

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
            f"      📊 暴露性范围：{df_result['e_people'].min():.4f} ~ {df_result['e_people'].max():.4f} | 归一化范围：0.0000 ~ 1.0000")


# -------------------------- 主函数：执行全域多降雨等级批量计算全流程 --------------------------
if __name__ == "__main__":
    try:
        # 步骤1：批量读取并合并1-648号summary文件
        df_all_summary = batch_load_and_merge_summary()
        # 步骤2：读取并预处理建筑容积数据
        df_building = load_and_preprocess_building()
        # 步骤3：融合数据+计算全域极值（仅1次，所有等级复用）
        df_merged = merge_data_and_calc_global_max(df_all_summary, df_building)
        # 步骤4：多降雨等级批量计算暴露性（仅保留考虑建筑庇护的结果）
        batch_calculate_exposure(df_merged, ALPHA)
        # 全流程完成提示
        print("\n" + "=" * 60)
        print("🎉 【全流程完成】1-648号summary合并+小雨→特大暴雨全等级暴露性计算完成！")
        print(f"📁 所有结果文件已保存至：{OUTPUT_DIR}")
        print("📌 结果文件说明：e_people=原始暴露性 | e_people_norm=归一化暴露性（0-1，值越大暴露性越高）")
        print("=" * 60)
    except Exception as e:
        # 捕获异常并打印，方便排查问题
        print(f"\n❌ 计算流程失败：{str(e)}")
        raise  # 抛出异常，终止程序