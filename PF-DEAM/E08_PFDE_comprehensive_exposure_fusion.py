import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import os

# ===================== 1. 配置参数 =====================
# 文件路径配置
# 人群暴露性 CSV 路径模板（{降雨等级} 为占位符）
people_expo_path_template = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\Exposure\全域人群暴露性计算结果_考虑建筑庇护_{降雨等级}_1-648_东西城区筛选后.csv"
# GDP 暴露性 SHP 路径模板（{情境} 为占位符）
gdp_expo_path_template = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\Exposure\GDP暴露性_{情境}.shp"

# 输出根路径（所有降雨等级的结果会保存在此目录下）
output_root = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\Exposure\综合暴露性按降雨等级拆分"
# 创建输出目录（如果不存在）
os.makedirs(output_root, exist_ok=True)

# 降雨等级列表
rain_grades = ["小雨", "中雨", "大雨", "暴雨", "大暴雨", "特大暴雨"]
# GDP 暴露性场景列表
gdp_scenarios = ["workday", "worknight", "restday", "restnight"]


# 时间匹配规则
# - weekday: 0=周一,1=周二,2=周三,3=周四,4=周五,5=周六,6=周日
# - 白天（day）：7:00-18:00（hour 7-18）；黑夜（night）：19:00-6:00（hour 19-23 + 0-6）
def match_gdp_scenario(weekday, hour):
    """根据星期和小时匹配GDP暴露性场景"""
    # 判断工作日/休息日
    if weekday in [0, 1, 2, 3, 4]:  # 周一到周五：工作日
        # 判断白天/黑夜
        if 7 <= hour <= 18:
            return "workday"
        else:
            return "worknight"
    else:  # 周六、周日：休息日
        if 7 <= hour <= 18:
            return "restday"
        else:
            return "restnight"


# ===================== 2. 工具函数 =====================
def normalize_series(series):
    """对数值序列进行Min-Max归一化（0-1）"""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:  # 避免除以0
        return pd.Series(0, index=series.index)
    return (series - min_val) / (max_val - min_val)


# ===================== 3. 读取GDP暴露性数据（空间+时间场景） =====================
print("正在读取GDP暴露性数据...")
gdp_expo_dict = {}
for scenario in tqdm(gdp_scenarios):
    shp_path = gdp_expo_path_template.format(情境=scenario)
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"GDP暴露性文件不存在：{shp_path}")
    # 读取SHP，保留geohash7和暴露性字段（字段名改为RASTERVALU）
    gdf = gpd.read_file(shp_path)
    # 确保geohash7字段为字符串类型（避免匹配失败）
    gdf["geohash7"] = gdf["geohash7"].astype(str)
    # 提取核心字段：geohash7 + 暴露性值（字段名改为RASTERVALU）
    gdp_expo_dict[scenario] = gdf[["geohash7", "RASTERVALU"]].copy()
    # 对每个场景的GDP暴露性先归一化，新增列名expo_gdp_norm
    gdp_expo_dict[scenario]["expo_gdp_norm"] = normalize_series(gdp_expo_dict[scenario]["RASTERVALU"])

# ===================== 4. 处理人群暴露性数据（按降雨等级循环，单独保存） =====================
for rain_grade in tqdm(rain_grades, desc="处理各降雨等级数据"):
    # 读取当前降雨等级的人群暴露性CSV
    csv_path = people_expo_path_template.format(降雨等级=rain_grade)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"人群暴露性文件不存在：{csv_path}")
    df_people = pd.read_csv(csv_path)

    # 数据预处理：确保关键字段类型正确
    df_people["final_grid_id"] = df_people["final_grid_id"].astype(str)  # 空间匹配字段
    df_people["weekday"] = df_people["weekday"].astype(int)  # 星期（0-6）
    df_people["hour"] = df_people["predict_hour"].astype(int)  # 小时（0-23）
    df_people["e_people"] = pd.to_numeric(df_people["e_people"], errors="coerce")  # 人群暴露性值

    # 过滤无效值（NaN）
    df_people = df_people.dropna(subset=["e_people", "final_grid_id", "weekday", "hour"])

    # 步骤1：为每条人群暴露性记录匹配GDP场景
    df_people["gdp_scenario"] = df_people.apply(lambda row: match_gdp_scenario(row["weekday"], row["hour"]), axis=1)

    # 步骤2：对人群暴露性先归一化
    df_people["e_people_norm"] = normalize_series(df_people["e_people"])

    # 步骤3：空间+时间匹配GDP暴露性
    merged_dfs = []
    for scenario in gdp_scenarios:
        # 筛选当前场景的人群数据
        df_sub = df_people[df_people["gdp_scenario"] == scenario].copy()
        if df_sub.empty:
            continue
        # 获取当前场景的GDP数据（包含RASTERVALU和归一化值）
        df_gdp = gdp_expo_dict[scenario][["geohash7", "RASTERVALU", "expo_gdp_norm"]].copy()
        # 空间匹配：final_grid_id（人群） <-> geohash7（GDP）
        df_merged = pd.merge(
            df_sub,
            df_gdp,
            left_on="final_grid_id",
            right_on="geohash7",
            how="left"
        )
        merged_dfs.append(df_merged)

    # 合并所有场景的匹配结果
    df_merged_all = pd.concat(merged_dfs, ignore_index=True)

    # 步骤4：计算综合暴露性（归一化后的均值）
    # 填充匹配失败的GDP暴露性为0（或根据需求改为均值）
    df_merged_all["RASTERVALU"] = df_merged_all["RASTERVALU"].fillna(0)
    df_merged_all["expo_gdp_norm"] = df_merged_all["expo_gdp_norm"].fillna(0)
    # 计算均值
    df_merged_all["expo_combine_raw"] = (df_merged_all["e_people_norm"] + df_merged_all["expo_gdp_norm"]) / 2
    # 全局归一化（最终综合暴露性）
    df_merged_all["expo_combine"] = normalize_series(df_merged_all["expo_combine_raw"])

    # 添加降雨等级标识
    df_merged_all["rain_grade"] = rain_grade

    # 保留核心字段（将RASTERVALU作为原始GDP暴露性字段）
    df_result = df_merged_all[[
        "final_grid_id", "weekday", "hour", "rain_grade",
        "e_people", "e_people_norm", "RASTERVALU", "expo_gdp_norm",
        "expo_combine_raw", "expo_combine"
    ]].copy()

    # ===================== 5. 按降雨等级单独保存文件 =====================
    # 1. 保存CSV文件（每个降雨等级一个文件）
    output_csv_path = os.path.join(output_root, f"综合暴露性_{rain_grade}.csv")
    df_result.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n{rain_grade} CSV结果已保存至：{output_csv_path}")

    # 2. 保存SHP文件（每个降雨等级一个文件，可选）
    # 读取基础SHP（以workday为例）获取地理信息
    base_gdf = gpd.read_file(gdp_expo_path_template.format(情境="workday"))
    # 聚合综合暴露性（按geohash7+星期+小时）
    df_agg = df_result.groupby(["final_grid_id", "weekday", "hour"])["expo_combine"].mean().reset_index()
    # 关联地理信息
    gdf_final = pd.merge(
        base_gdf,
        df_agg,
        left_on="geohash7",
        right_on="final_grid_id",
        how="left"
    )
    # 保存SHP
    output_shp_path = os.path.join(output_root, f"综合暴露性_{rain_grade}.shp")
    gdf_final.to_file(output_shp_path, encoding="utf-8")
    print(f"{rain_grade} SHP结果已保存至：{output_shp_path}")

print("\n所有降雨等级的综合暴露性计算完成！结果已按降雨等级拆分保存至：", output_root)