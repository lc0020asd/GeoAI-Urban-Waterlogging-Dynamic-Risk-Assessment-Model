import os
import itertools
import pandas as pd
import numpy as np
import geopandas as gpd
import mapclassify as mc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# ====================== 1. 全局配置 ======================
BASE_PATH = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\洪涝风险最终结果"
OUTPUT_DIR = os.path.join(BASE_PATH, "地理探测器结果_综合风险驱动分析")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 降雨等级列表（与风险计算代码保持一致）
RAIN_GRADES = ["小雨", "中雨", "大雨", "暴雨", "大暴雨", "特大暴雨"]

# 定义因子名称映射（自变量：危险性、暴露性、脆弱性；因变量：综合风险）
FACTOR_MAP = {
    'HAZARD_CAT': '危险性(H)',
    'EXPO_CAT': '暴露性(E)',
    'VULN_CAT': '脆弱性(V)'
}

# 分类数配置（地理探测器建议分类数3-6类）
CLASS_NUM = 6


# ====================== 2. 核心辅助函数 ======================
def load_risk_data(rain_grade):
    """加载指定降雨等级的综合风险数据"""
    csv_path = os.path.join(BASE_PATH, f"洪涝风险_{rain_grade}.csv")
    shp_path = os.path.join(BASE_PATH, f"洪涝风险_{rain_grade}.shp")

    # 优先读取CSV（更快），失败则读取SHP
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        # 数据清洗
        df = df.dropna(subset=['risk_final', 'H_index', 'expo_combine', 'v_combine'])
        df = df[df['risk_final'] >= 0]  # 过滤异常值
        return df
    except:
        try:
            gdf = gpd.read_file(shp_path, encoding="utf-8")
            df = pd.DataFrame(gdf.drop(columns='geometry'))
            df = df.dropna(subset=['risk_final', 'H_index', 'expo_combine', 'v_combine'])
            df = df[df['risk_final'] >= 0]
            return df
        except Exception as e:
            print(f"❌ 加载{rain_grade}数据失败: {str(e)}")
            return None


def preprocess_factors(df):
    """预处理因子：对连续变量进行分类（自然断点法）"""
    # 危险性分类
    try:
        df['HAZARD_CAT'] = mc.NaturalBreaks(df['H_index'].dropna(), k=CLASS_NUM).yb
    except:
        df['HAZARD_CAT'] = pd.qcut(df['H_index'], q=CLASS_NUM, labels=False, duplicates='drop')

    # 暴露性分类
    try:
        df['EXPO_CAT'] = mc.NaturalBreaks(df['expo_combine'].dropna(), k=CLASS_NUM).yb
    except:
        df['EXPO_CAT'] = pd.qcut(df['expo_combine'], q=CLASS_NUM, labels=False, duplicates='drop')

    # 脆弱性分类
    try:
        df['VULN_CAT'] = mc.NaturalBreaks(df['v_combine'].dropna(), k=CLASS_NUM).yb
    except:
        df['VULN_CAT'] = pd.qcut(df['v_combine'], q=CLASS_NUM, labels=False, duplicates='drop')

    return df


def calculate_q_value(y, x_strata):
    """计算单因子q值（解释力）和P值（显著性）"""
    df = pd.DataFrame({'y': y, 'strata': x_strata}).dropna()
    # 样本量过小则返回无效值
    if len(df) < 20 or df['strata'].nunique() < 2:
        return 0.0, 1.0

    mean_total = df['y'].mean()
    sst = np.sum((df['y'] - mean_total) ** 2)
    # 因变量无变异则解释力为0
    if sst == 0:
        return 0.0, 1.0

    ssw = 0.0
    for _, group in df.groupby('strata'):
        if len(group) < 2:
            continue
        ssw += np.sum((group['y'] - group['y'].mean()) ** 2)

    # 计算q值（1 - 组内变异/总变异）
    q = 1 - (ssw / sst)

    # 计算F检验和P值（显著性）
    N, L = len(df), df['strata'].nunique()
    df1, df2 = L - 1, N - L
    if df2 <= 0:
        return q, 1.0
    F = ((sst - ssw) / df1) / (ssw / df2)
    p = stats.f.sf(F, df1, df2)

    return round(q, 4), round(p, 4)


def calculate_q_only(y, x_strata):
    """仅计算q值（用于交互探测，提高效率）"""
    df = pd.DataFrame({'y': y, 'strata': x_strata}).dropna()
    if len(df) < 20 or df['strata'].nunique() < 2:
        return 0.0

    mean_total = df['y'].mean()
    sst = np.sum((df['y'] - mean_total) ** 2)
    if sst == 0:
        return 0.0

    ssw = 0.0
    for _, group in df.groupby('strata'):
        if len(group) < 2:
            continue
        ssw += np.sum((group['y'] - group['y'].mean()) ** 2)

    return round(1 - (ssw / sst), 4)


def judge_interaction_type(q1, q2, q_inter):
    """判断双因子交互类型（基于王劲峰老师的地理探测器交互探测规则）"""
    q1, q2, q_inter = float(q1), float(q2), float(q_inter)
    if q_inter > q1 + q2:
        return "非线性增强"
    elif max(q1, q2) < q_inter < q1 + q2:
        return "双因子增强"
    elif q_inter == q1 + q2:
        return "线性增强"
    elif q_inter == max(q1, q2):
        return "独立"
    elif min(q1, q2) < q_inter < max(q1, q2):
        return "单因子非线性减弱"
    elif q_inter < min(q1, q2):
        return "非线性减弱"
    else:
        return "无法判断"


# ====================== 3. 主程序 ======================
def main():
    print("=" * 70)
    print("🚀 综合风险地理探测器分析 (单因子+双因子交互)")
    print("=" * 70)

    # 初始化结果存储
    summary_results = []  # 单因子探测结果
    interaction_results = []  # 双因子交互结果
    factor_pairs = list(itertools.combinations(FACTOR_MAP.keys(), 2))  # 因子两两组合

    # 遍历每个降雨等级
    for rain_grade in RAIN_GRADES:
        print(f"\n{'=' * 70}")
        print(f"📊 正在处理降雨等级: 【{rain_grade}】")
        print(f"{'=' * 70}")

        # 1. 加载风险数据
        df_risk = load_risk_data(rain_grade)
        if df_risk is None or len(df_risk) < 50:  # 样本量过小跳过
            print(f"⚠️ {rain_grade} 有效样本量不足，跳过分析")
            continue
        print(f"   ✅ 加载成功，有效样本量: {len(df_risk):,} 条")

        # 2. 预处理因子（连续变量分类）
        df_risk = preprocess_factors(df_risk)

        # 3. 单因子地理探测器分析
        print(f"\n   --- {rain_grade} 单因子探测结果 ---")
        grade_res = {'降雨等级': rain_grade}
        factor_q_values = {}  # 暂存单因子q值，用于交互分析

        for col, name in FACTOR_MAP.items():
            q, p = calculate_q_value(df_risk['risk_final'], df_risk[col])
            factor_q_values[col] = q
            grade_res[name] = q
            grade_res[f"{name}_P值"] = p
            print(f"   {name}: q={q} (P={p})")

        summary_results.append(grade_res)

        # 4. 双因子交互探测分析
        print(f"\n   --- {rain_grade} 双因子交互探测结果 ---")
        for (f1, f2) in factor_pairs:
            # 生成交互分层（两个因子分类的组合）
            df_risk['interaction_strata'] = df_risk[[f1, f2]].apply(tuple, axis=1)

            # 计算交互q值
            q1 = factor_q_values[f1]
            q2 = factor_q_values[f2]
            q_inter = calculate_q_only(df_risk['risk_final'], df_risk['interaction_strata'])
            inter_type = judge_interaction_type(q1, q2, q_inter)

            # 保存交互结果
            inter_res = {
                '降雨等级': rain_grade,
                '因子A': FACTOR_MAP[f1],
                '因子B': FACTOR_MAP[f2],
                'q(因子A)': q1,
                'q(因子B)': q2,
                'q(因子A∩B)': q_inter,
                '交互类型': inter_type
            }
            interaction_results.append(inter_res)
            print(f"   {FACTOR_MAP[f1]} ∩ {FACTOR_MAP[f2]}: q={q_inter} ({inter_type})")

        # 5. 保存该等级的详细数据
        grade_out_dir = os.path.join(OUTPUT_DIR, rain_grade)
        os.makedirs(grade_out_dir, exist_ok=True)
        df_risk[['risk_final', 'H_index', 'expo_combine', 'v_combine',
                 'HAZARD_CAT', 'EXPO_CAT', 'VULN_CAT']].to_csv(
            os.path.join(grade_out_dir, f"{rain_grade}_地理探测器输入数据.csv"),
            index=False, encoding='utf-8-sig'
        )

    # ====================== 4. 生成汇总可视化（风格完全匹配参考代码） ======================
    print("\n" + "=" * 70)
    print("📈 生成汇总可视化图表")
    print("=" * 70)

    # 全局字体配置（参考代码风格）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 4.1 保存汇总表
    df_summary = pd.DataFrame(summary_results)
    df_inter_summary = pd.DataFrame(interaction_results)

    df_summary.to_csv(os.path.join(OUTPUT_DIR, "1_单因子探测结果汇总表.csv"),
                      index=False, encoding='utf-8-sig')
    df_inter_summary.to_csv(os.path.join(OUTPUT_DIR, "2_双因子交互探测结果汇总表.csv"),
                            index=False, encoding='utf-8-sig')

    # 4.2 【核心修改】单因子对比图（完全参考代码风格）
    if not df_summary.empty:
        # 数据重塑
        df_plot = df_summary.melt(id_vars=['降雨等级'],
                                  value_vars=list(FACTOR_MAP.values()),
                                  var_name='驱动因子', value_name='q值')

        # 创建画布
        fig, ax = plt.subplots(figsize=(14, 7))

        # 绘制分组柱状图（参考代码风格：palette='deep'，无额外装饰）
        sns.barplot(data=df_plot,
                    x='降雨等级',
                    y='q值',
                    hue='驱动因子',
                    palette='deep',  # 改为参考代码的通用配色
                    ax=ax,
                    order=RAIN_GRADES)

        # 图表样式（完全参考代码）
        ax.set_title('不同降雨情境下综合风险单因子驱动探测结果', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='x-small')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "1_单因子对比图.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 单因子对比图已保存")

    # 4.3 【核心修改】双因子交互对比图（完全参考代码风格：仅取Top3）
    if not df_inter_summary.empty:
        # 数据重塑：仅取每个降雨等级的Top3交互因子
        df_inter_top = df_inter_summary.groupby('降雨等级').apply(
            lambda x: x.nlargest(3, 'q(因子A∩B)')
        ).reset_index(drop=True)
        df_inter_top['交互因子对'] = df_inter_top['因子A'] + " ∩ " + df_inter_top['因子B']

        # 创建画布
        fig, ax = plt.subplots(figsize=(14, 7))

        # 绘制分组柱状图
        sns.barplot(data=df_inter_top,
                    x='降雨等级',
                    y='q(因子A∩B)',
                    hue='交互因子对',
                    palette='deep',
                    ax=ax,
                    order=RAIN_GRADES)

        # 图表样式
        ax.set_title('不同降雨情境下综合风险Top3交互因子对比', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='x-small')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "2_交互因子对比图.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 交互因子对比图已保存")

    print("\n🎉 所有分析完成！结果保存至: " + OUTPUT_DIR)


if __name__ == "__main__":
    main()