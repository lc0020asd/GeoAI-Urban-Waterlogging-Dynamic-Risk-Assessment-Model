import pandas as pd
import numpy as np
import os
from typing import Dict, List

# -------------------------- 1. 核心配置（修正字段名，严格匹配你的最新说明） --------------------------
# 步骤1：场所类型→指标映射（使用你确认的中文别名+英文字段名，无乱码）
PLACE_INDICATORS: Dict[str, List[str]] = {
    "绿色休闲场所": ["风景名l"],  # 绿色休闲对应：风景名l
    "公共服务场所": ["科教文l", "医疗保l", "公共设l", "政府机l", "Public_ser"],  # 公共服务对应
    "生活场所": ["住宿服l", "商务住l", "Residence"],  # 生活场所对应
    "商业场所": [
        "购物服l", "Business_c", "生活服l", "公司企l",
        "金融保l", "汽车销l", "餐饮服l", "体育休l", "Office_cou"
    ],  # 商业场所对应
    "工业场所": ["Industry_c"],  # 工业场所对应
    "交通服务场所": [
        "公司企l", "汽车服l", "金融保l", "公共设l", "汽车维l"
    ]  # 交通服务：道路附l/公司企l/汽车服l/金融保l/公共设l/汽车维l（5个指标总和）
}

# 步骤2：AHP权重表（严格匹配表22，4个时间情境×6类场所）
PLACE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "工作日白天": {
        "绿色休闲场所": 0.020,
        "公共服务场所": 0.472,
        "生活场所": 0.035,
        "商业场所": 0.142,
        "工业场所": 0.068,
        "交通服务场所": 0.263
    },
    "工作日黑夜": {
        "绿色休闲场所": 0.020,
        "公共服务场所": 0.263,
        "生活场所": 0.472,
        "商业场所": 0.068,
        "工业场所": 0.035,
        "交通服务场所": 0.142
    },
    "休息日白天": {
        "绿色休闲场所": 0.068,
        "公共服务场所": 0.142,
        "生活场所": 0.472,
        "商业场所": 0.263,
        "工业场所": 0.020,
        "交通服务场所": 0.035
    },
    "休息日黑夜": {
        "绿色休闲场所": 0.020,
        "公共服务场所": 0.263,
        "生活场所": 0.472,
        "商业场所": 0.142,
        "工业场所": 0.035,
        "交通服务场所": 0.068
    }
}

# 步骤3：文件路径配置
INPUT_FILE = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\fishnet\geohash7\shp\result_core_area.csv'
OUTPUT_DIR = os.path.dirname(INPUT_FILE)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Geohash7设施重要性得分_4情境.csv")
# 归一化范围（指标值归一化到[0,1]）
NORMALIZE_RANGE = (0, 1)


# -------------------------- 2. 数据读取与预处理（增强字段校验提示） --------------------------
def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    读取CSV数据，处理缺失值，确保指标字段为数值类型
    增强：输出CSV实际列名，便于核对字段名
    """
    # 读取数据（兼容中文/特殊字符字段）
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # 输出CSV实际列名（便于你核对）
    print("📋 CSV文件实际列名列表：")
    for idx, col in enumerate(df.columns):
        print(f"   [{idx + 1}] {col}")

    # 保留核心字段：geohash7 + 所有指标字段
    core_fields = ['geohash7']
    # 收集所有需要的指标字段（去重）
    all_indicators = []
    for indicators in PLACE_INDICATORS.values():
        all_indicators.extend(indicators)
    all_indicators = list(set(all_indicators))

    # 校验指标字段是否存在
    missing_fields = [f for f in all_indicators if f not in df.columns]
    if missing_fields:
        raise ValueError(
            f"❌ 缺失指标字段：{missing_fields}\n"
            f"💡 请检查：1. CSV列名是否与上述'实际列名列表'一致；2. 指标别名是否准确；3. 字段名是否有空格/大小写差异"
        )

    # 保留核心字段+指标字段
    df = df[core_fields + all_indicators].copy()

    # 处理缺失值（填充为0）+ 转换为数值类型
    for col in all_indicators:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    print(f"\n✅ 数据读取完成 | 总网格数：{len(df)} | 有效Geohash7数：{df['geohash7'].nunique()}")
    return df


# -------------------------- 3. 指标归一化（类别内全局归一化） --------------------------
def normalize_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每类场所的指标总和做全局最小-最大归一化（得到F_normalized）
    步骤：1. 计算每类场所的指标总和 2. 对总和做归一化
    """
    df = df.copy()
    place_totals = {}
    place_normed = {}

    # 步骤1：计算每类场所的指标总和
    for place_name, indicators in PLACE_INDICATORS.items():
        # 求和：该类场所的所有指标值相加
        df[f'{place_name}_总和'] = df[indicators].sum(axis=1)
        place_totals[place_name] = df[f'{place_name}_总和']
        print(
            f"📊 {place_name} - 指标总和统计：最小值={df[f'{place_name}_总和'].min()} | 最大值={df[f'{place_name}_总和'].max()}")

    # 步骤2：对每类场所的总和做全局归一化
    for place_name, total_series in place_totals.items():
        total_min = total_series.min()
        total_max = total_series.max()

        # 处理全0情况（避免除零）
        if total_max == total_min:
            normed_series = pd.Series([NORMALIZE_RANGE[0]] * len(df), index=df.index)
        else:
            # 最小-最大归一化公式
            normed_series = (total_series - total_min) / (total_max - total_min)
            normed_series = normed_series * (NORMALIZE_RANGE[1] - NORMALIZE_RANGE[0]) + NORMALIZE_RANGE[0]

        df[f'{place_name}_归一化'] = normed_series.round(4)
        place_normed[place_name] = df[f'{place_name}_归一化']

    print("\n✅ 指标归一化完成 | 已计算6类场所的总和与归一化值")
    return df


# -------------------------- 4. 核心计算：按情境计算设施重要性得分 --------------------------
def calculate_facility_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    按公式 Vfacility(t) = Σ(W_i(t) * F_normalized(i)) 计算4个情境的得分
    """
    df = df.copy()
    # 遍历4个时间情境
    for scenario, weights in PLACE_WEIGHTS.items():
        # 初始化得分列
        df[f'{scenario}_得分'] = 0.0

        # 按权重加权求和
        for place_name, weight in weights.items():
            norm_col = f'{place_name}_归一化'
            df[f'{scenario}_得分'] += df[norm_col] * weight

        # 保留4位小数
        df[f'{scenario}_得分'] = df[f'{scenario}_得分'].round(4)

    # 保留最终结果字段：Geohash7 + 4个情境得分
    result_cols = ['geohash7'] + [f'{scenario}_得分' for scenario in PLACE_WEIGHTS.keys()]
    df_result = df[result_cols].copy()

    # 去重（确保每个Geohash7只保留一条记录）
    df_result = df_result.drop_duplicates(subset=['geohash7'], keep='first')

    print(f"✅ 设施重要性得分计算完成 | 最终有效网格数：{len(df_result)}")
    return df_result


# -------------------------- 5. 主函数：执行全流程 --------------------------
if __name__ == "__main__":
    try:
        print("=" * 70)
        print("📌 Geohash7网格设施重要性得分计算（4个时间情境）")
        print("=" * 70)

        # 步骤1：读取并预处理数据
        df_raw = load_and_preprocess_data(INPUT_FILE)
        print(df_raw)

        # 步骤2：计算各类场所指标总和并归一化
        df_normed = normalize_indicators(df_raw)

        # 步骤3：按4个情境计算设施重要性得分
        df_final = calculate_facility_importance(df_normed)

        # 步骤4：保存结果
        df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n✅ 结果已保存至：{OUTPUT_FILE}")

        # 输出统计信息
        print(f"\n📊 计算结果统计：")
        for scenario in PLACE_WEIGHTS.keys():
            score_col = f'{scenario}_得分'
            min_score = df_final[score_col].min()
            max_score = df_final[score_col].max()
            mean_score = df_final[score_col].mean().round(4)
            print(f"   {scenario}：最小值={min_score} | 最大值={max_score} | 平均值={mean_score}")

        print("\n" + "=" * 70)
        print("🎉 设施重要性得分计算全流程完成！")
        print(f"📌 结果说明：")
        print(f"   - 得分范围：0~1（值越大，该网格设施重要性越高）")
        print(f"   - 计算逻辑：Σ(场所权重 × 场所指标归一化值)")
        print(f"   - 权重来源：AHP层次分析（表22）")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 计算失败：{str(e)}")
        raise