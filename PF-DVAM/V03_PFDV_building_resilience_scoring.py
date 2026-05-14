import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Tuple

# -------------------------- 1. 核心配置参数 --------------------------
# 文件路径
INPUT_SHP = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\building_core\building_point_core.shp'
OUTPUT_SHP = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\building_core\building_point_core_with_resilience.shp'
# 特殊值转换：AF2018统一转为2022
AF2018_VALUE = "AF2018"  # 根据实际数据格式调整（字符串/数值）
AF2018_CONVERT = 2022
# 结果列名
RESILIENCE_COL = "V_resilience"


# -------------------------- 2. 数据读取与预处理 --------------------------
def load_and_preprocess_building_data(shp_path: str) -> Tuple[gpd.GeoDataFrame, pd.Series]:
    """
    读取建筑物SHP数据，预处理Age字段：
    1. 转换AF2018为2022
    2. 填充空值为全体建筑Age的众数
    3. 转换为数值类型
    返回：预处理后的GeoDataFrame、处理后的Age序列
    """
    # 读取SHP文件
    gdf = gpd.read_file(shp_path, encoding='utf-8')
    print(f"✅ 读取建筑物数据完成 | 总建筑数：{len(gdf)} | 原始字段：{list(gdf.columns)}")

    # 校验Age字段是否存在
    if "Age" not in gdf.columns:
        raise ValueError(f"❌ 缺失Age字段！SHP文件包含字段：{list(gdf.columns)}")

    # 步骤1：复制Age字段避免修改原始数据
    age_series = gdf["Age"].copy()

    # 步骤2：处理AF2018值（兼容字符串/数值类型）
    if isinstance(AF2018_VALUE, str):
        age_series = age_series.astype(str).replace(AF2018_VALUE, str(AF2018_CONVERT))
    else:
        age_series = age_series.replace(AF2018_VALUE, AF2018_CONVERT)

    # 步骤3：转换为数值类型（无法转换的设为NaN）
    age_series = pd.to_numeric(age_series, errors='coerce')

    # 步骤4：计算众数（填充空值）
    age_mode = age_series.mode()[0]  # 取第一个众数
    age_series = age_series.fillna(age_mode)

    # 步骤5：校验Age范围（确保无异常值）
    age_min = age_series.min()
    age_max = age_series.max()
    if age_min == age_max:
        raise ValueError(f"❌ Age字段所有值均为{age_min}，无法计算线性模型！")

    print(f"\n📊 Age字段预处理统计：")
    print(f"   - AF2018转换为：{AF2018_CONVERT}")
    print(f"   - 空值填充众数：{age_mode}")
    print(f"   - Age范围：{age_min} ~ {age_max}")
    print(f"   - 空值数量：{gdf['Age'].isna().sum()} → 填充后空值：{age_series.isna().sum()}")

    return gdf, age_series


# -------------------------- 3. 核心计算：御灾能力评分 --------------------------
def calculate_resilience_score(age_series: pd.Series) -> pd.Series:
    """
    按公式计算御灾能力评分：
    1. S_resilience = 1 + (age - min_age) / (max_age - min_age)
    2. V_resilience = (Max(S_resilience) - S_resilience) / (Max(S_resilience) - Min(S_resilience))
    返回：V_resilience序列（归一化到0~1，值越大御灾能力越弱）
    """
    # 步骤1：计算Age的极值
    age_min = age_series.min()
    age_max = age_series.max()

    # 步骤2：计算S_resilience（御灾能力原始评分，值越大能力越强）
    s_resilience = 1 + (age_series - age_min) / (age_max - age_min)

    # 步骤3：计算S_resilience的极值
    s_min = s_resilience.min()
    s_max = s_resilience.max()

    # 步骤4：计算V_resilience（归一化到0~1，值越大御灾能力越弱）
    v_resilience = (s_max - s_resilience) / (s_max - s_min)

    # 保留4位小数
    v_resilience = v_resilience.round(4)

    print(f"\n📊 御灾能力评分统计：")
    print(f"   - S_resilience范围：{s_min:.4f} ~ {s_max:.4f}")
    print(f"   - V_resilience范围：{v_resilience.min():.4f} ~ {v_resilience.max():.4f}")

    return v_resilience


# -------------------------- 4. 保存结果 --------------------------
def save_resilience_result(gdf: gpd.GeoDataFrame, v_resilience: pd.Series, output_path: str) -> None:
    """
    新增V_resilience列并保存为SHP文件
    """
    # 新增列
    gdf[RESILIENCE_COL] = v_resilience

    # 保存SHP文件（兼容中文编码）
    gdf.to_file(output_path, encoding='utf-8', driver='ESRI Shapefile')
    print(f"\n✅ 结果保存完成 | 输出文件：{output_path}")
    print(f"   - 新增字段：{RESILIENCE_COL}（御灾能力评分，0~1）")
    print(f"   - 字段说明：值越大 → 建筑御灾能力越弱；值越小 → 御灾能力越强")


# -------------------------- 5. 主函数：执行全流程 --------------------------
if __name__ == "__main__":
    try:
        print("=" * 70)
        print("📌 建筑物抗洪灾能力评分计算（基于建成年份）")
        print("=" * 70)

        # 步骤1：读取并预处理数据
        gdf, age_series = load_and_preprocess_building_data(INPUT_SHP)

        # 步骤2：计算御灾能力评分
        v_resilience = calculate_resilience_score(age_series)

        # 步骤3：保存结果
        save_resilience_result(gdf, v_resilience, OUTPUT_SHP)

        print("\n" + "=" * 70)
        print("🎉 建筑物抗洪灾能力评分计算全流程完成！")
        print(f"📌 核心公式回顾：")
        print(f"   1. S_resilience = 1 + (age - min_age)/(max_age - min_age)")
        print(f"   2. V_resilience = (Max(S) - S)/(Max(S) - Min(S))")
        print(f"📌 结果说明：{RESILIENCE_COL}∈[0,1]，值越大→御灾能力越弱")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 计算失败：{str(e)}")
        raise