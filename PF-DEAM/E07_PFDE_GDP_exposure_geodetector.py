import os
import itertools
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import mapclassify as mc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ====================== 1. 全局配置 ======================
BASE_PATH = r"D:\baidusyncdisk\学习资料\ysjs\开题\论文数据"
BOUNDARY_PATH = os.path.join(BASE_PATH, "北京市边界_110000_Shapefile_(poi86.com)", "东西城区.shp")

# --- 数据路径配置 ---
GDP_RASTER_ROOT = os.path.join(BASE_PATH, "GDP数据")
GDP_FACTOR_FILES = {
    'AGRI_GDP': os.path.join(GDP_RASTER_ROOT, "AgricultureGDP_clip.tif"),
    'INDU_GDP': os.path.join(GDP_RASTER_ROOT, "IndustryGDP_clip.tif"),
    'SERV_GDP': os.path.join(GDP_RASTER_ROOT, "ServiceGDP_clip.tif")
}
EXPOSURE_RASTER_ROOT = os.path.join(BASE_PATH, "辅助数据", "Exposure")
SCENARIOS = ["工作日白天", "工作日夜晚", "休息日白天", "休息日夜晚"]
SCENARIO_RASTER_FILES = {
    "工作日白天": "GDP设施暴露性_工作日白天.tif",
    "工作日夜晚": "GDP设施暴露性_工作日黑夜.tif",
    "休息日白天": "GDP设施暴露性_休息日白天.tif",
    "休息日夜晚": "GDP设施暴露性_休息日黑夜.tif"
}
OUTPUT_DIR = os.path.join(BASE_PATH, "地理探测器结果_GDP设施暴露性_栅格版_含交互探测_含显著性")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 因子名称映射 ---
FACTOR_MAP = {
    'AGRI_GDP_CAT': '农业GDP',
    'INDU_GDP_CAT': '工业GDP',
    'SERV_GDP_CAT': '服务业GDP'
}


# ====================== 2. 核心辅助函数 (已补充显著性计算) ======================
def load_and_reproject_boundary():
    gdf = gpd.read_file(BOUNDARY_PATH)
    return gdf.to_crs(epsg=4326)


def preprocess_all_rasters(boundary_gdf):
    temp_files = {'factors': {}, 'exposure': {}}
    print("🔧 正在统一预处理所有栅格数据...")

    # 预处理驱动因子
    with rasterio.open(list(GDP_FACTOR_FILES.values())[0]) as src:
        base_crs = src.crs

    for name, path in GDP_FACTOR_FILES.items():
        temp_path = os.path.join(OUTPUT_DIR, f"temp_factor_{name}.tif")
        if not os.path.exists(temp_path):
            with rasterio.open(path) as src:
                boundary_reproj = boundary_gdf.to_crs(src.crs)
                geoms = [mapping(geom) for geom in boundary_reproj.geometry]
                out_image, out_transform = mask(src, geoms, crop=True, nodata=src.nodata)
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
                                 "transform": out_transform, "crs": base_crs})
                with rasterio.open(temp_path, "w", **out_meta) as dest:
                    dest.write(out_image)
        temp_files['factors'][name] = temp_path

    # 预处理因变量
    for scenario, filename in SCENARIO_RASTER_FILES.items():
        raw_path = os.path.join(EXPOSURE_RASTER_ROOT, filename)
        temp_path = os.path.join(OUTPUT_DIR, f"temp_exposure_{scenario}.tif")
        if not os.path.exists(temp_path):
            with rasterio.open(raw_path) as src:
                boundary_reproj = boundary_gdf.to_crs(src.crs)
                geoms = [mapping(geom) for geom in boundary_reproj.geometry]
                out_image, out_transform = mask(src, geoms, crop=True, nodata=src.nodata)
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
                                 "transform": out_transform, "crs": base_crs})
                with rasterio.open(temp_path, "w", **out_meta) as dest:
                    dest.write(out_image)
        temp_files['exposure'][scenario] = temp_path

    print("✅ 所有栅格预处理完成")
    return temp_files


def extract_raster_matched_values(exposure_raster_path, factor_raster_paths):
    with rasterio.open(exposure_raster_path) as src_exp:
        exp_data = src_exp.read(1)
        exp_nodata = src_exp.nodata
        height, width = exp_data.shape
        transform = src_exp.transform

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    x_coords, y_coords = rasterio.transform.xy(transform, rows, cols)
    x_coords = np.array(x_coords).flatten()
    y_coords = np.array(y_coords).flatten()
    exp_values = exp_data.flatten()

    df = pd.DataFrame({'x': x_coords, 'y': y_coords, 'RASTERVALU': exp_values})
    df = df[df['RASTERVALU'] != exp_nodata].dropna(subset=['RASTERVALU'])
    if len(df) < 20:
        print(f"   ⚠️ 有效样本量不足，跳过")
        return None

    for name, path in factor_raster_paths.items():
        with rasterio.open(path) as src_factor:
            coords = list(zip(df['x'], df['y']))
            factor_values = [val[0] if val[0] != src_factor.nodata else np.nan for val in src_factor.sample(coords)]
            df[name] = factor_values

    df = df.dropna(subset=list(factor_raster_paths.keys()))
    if len(df) < 20:
        print(f"   ⚠️ 匹配后有效样本量不足，跳过")
        return None

    print(f"   有效匹配样本量: {len(df):,} 条")
    return df[['RASTERVALU'] + list(factor_raster_paths.keys())]


def calculate_q_value(y, x_strata):
    """
    【核心修改】地理探测器核心计算：同时返回 q值 和 显著性P值 (F检验)
    """
    df = pd.DataFrame({'y': y, 'strata': x_strata}).dropna()
    if len(df) < 20:
        return 0.0, 1.0  # 样本量不足，返回无效值

    mean_total = df['y'].mean()
    sst = np.sum((df['y'] - mean_total) ** 2)
    if sst == 0:
        return 0.0, 1.0

    ssw = 0.0
    for name, group in df.groupby('strata'):
        if len(group) < 2:
            continue
        ssw += np.sum((group['y'] - group['y'].mean()) ** 2)

    q = 1 - (ssw / sst)

    # --- 【新增】F检验计算P值 ---
    N, L = len(df), df['strata'].nunique()
    df1, df2 = L - 1, N - L
    if df2 <= 0:
        return q, 1.0
    F = ((sst - ssw) / df1) / (ssw / df2)
    p = stats.f.sf(F, df1, df2)

    return q, p


def judge_interaction_type(q1, q2, q_inter):
    if q_inter > q1 + q2:
        return "非线性增强"
    elif q_inter > max(q1, q2) and q_inter <= q1 + q2:
        return "双因子增强"
    elif q_inter == max(q1, q2):
        return "独立"
    elif q_inter < max(q1, q2) and q_inter > min(q1, q2):
        return "单因子非线性减弱"
    elif q_inter < min(q1, q2):
        return "非线性减弱"
    else:
        return "无法判断"


# ====================== 3. 主程序 (已更新，含显著性保存) ======================
def main():
    print("=" * 60)
    print("🚀 GDP设施暴露性地理探测器 (含显著性P值)")
    print("=" * 60)

    gdf_boundary = load_and_reproject_boundary()
    raster_temp_files = preprocess_all_rasters(gdf_boundary)
    factor_raster_paths = raster_temp_files['factors']
    exposure_raster_paths = raster_temp_files['exposure']

    all_factor_results = []
    all_interaction_results = []
    factor_pairs = list(itertools.combinations(FACTOR_MAP.keys(), 2))

    for scenario in SCENARIOS:
        print(f"\n{'=' * 60}")
        print(f"📊 正在处理情景: 【{scenario}】")
        print(f"{'=' * 60}")

        if scenario not in exposure_raster_paths:
            print(f"⚠️  该情景无有效栅格数据，跳过")
            continue

        df_scenario = extract_raster_matched_values(
            exposure_raster_path=exposure_raster_paths[scenario],
            factor_raster_paths=factor_raster_paths
        )
        if df_scenario is None:
            continue

        # 3. 数据预处理与单因子探测
        factor_q_values = {}
        print(f"\n   --- {scenario} 单因子探测结果 (含显著性) ---")
        for raw_col, cat_col in zip(factor_raster_paths.keys(), FACTOR_MAP.keys()):
            try:
                df_scenario[cat_col] = mc.NaturalBreaks(df_scenario[raw_col].dropna(), k=5).yb
            except:
                df_scenario[cat_col] = pd.qcut(df_scenario[raw_col], q=5, labels=False, duplicates='drop')

            # 【修改】同时获取q值和P值
            q, p_val = calculate_q_value(df_scenario['RASTERVALU'], df_scenario[cat_col])
            factor_q_values[cat_col] = q

            # 【修改】保存P值
            all_factor_results.append({
                '情景': scenario,
                '因子': FACTOR_MAP[cat_col],
                'q值': q,
                'P值': p_val
            })
            print(f"   {FACTOR_MAP[cat_col]}: q={q:.4f} (P={p_val:.4f})")

        # 4. 交互探测
        print(f"\n   --- {scenario} 交互探测结果 (含显著性) ---")
        for (f1, f2) in factor_pairs:
            df_scenario['interaction_strata'] = df_scenario[[f1, f2]].apply(tuple, axis=1)

            q1 = factor_q_values[f1]
            q2 = factor_q_values[f2]

            # 【修改】交互项也计算P值
            q_inter, p_inter = calculate_q_value(df_scenario['RASTERVALU'], df_scenario['interaction_strata'])
            inter_type = judge_interaction_type(q1, q2, q_inter)

            # 【修改】保存交互项P值
            res_row = {
                '情景': scenario,
                '因子A': FACTOR_MAP[f1],
                '因子B': FACTOR_MAP[f2],
                'q(A)': q1,
                'q(B)': q2,
                'q(A∩B)': q_inter,
                'q(A∩B)_P值': p_inter,  # 新增列
                '交互类型': inter_type
            }
            all_interaction_results.append(res_row)
            print(f"   {FACTOR_MAP[f1]} ∩ {FACTOR_MAP[f2]}: q={q_inter:.4f} (P={p_inter:.4f}, {inter_type})")

        # 5. 保存详细数据
        scenario_out_dir = os.path.join(OUTPUT_DIR, scenario)
        os.makedirs(scenario_out_dir, exist_ok=True)
        df_scenario.to_csv(os.path.join(scenario_out_dir, f"{scenario}_采样数据.csv"), index=False,
                           encoding='utf-8-sig')

    # ====================== 4. 结果汇总与可视化 ======================
    if len(all_factor_results) == 0:
        print("\n❌ 无有效分析结果")
        return

    print("\n" + "=" * 60)
    print("📈 正在生成汇总结果")

    df_factor = pd.DataFrame(all_factor_results)
    df_inter = pd.DataFrame(all_interaction_results)

    # 【修改】保存包含P值的CSV
    df_factor.to_csv(os.path.join(OUTPUT_DIR, "1_单因子探测结果汇总_含显著性.csv"), index=False, encoding='utf-8-sig')
    df_inter.to_csv(os.path.join(OUTPUT_DIR, "2_交互探测结果汇总_含显著性.csv"), index=False, encoding='utf-8-sig')

    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 图1：单因子对比
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=df_factor, x='情景', y='q值', hue='因子', palette='viridis', ax=ax, order=SCENARIOS)
    ax.set_title('不同情景下GDP设施暴露性单因子驱动探测 (q值)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('q统计量 (解释力)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_单因子探测对比图.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2：交互对比
    df_inter['交互对'] = df_inter['因子A'] + " ∩ " + df_inter['因子B']
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(data=df_inter, x='情景', y='q(A∩B)', hue='交互对', palette='coolwarm', ax=ax, order=SCENARIOS)
    ax.set_title('不同情景下GDP设施暴露性驱动因子交互探测 (q值)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('q统计量 (交互解释力)')
    ax.legend(title='交互因子对', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_交互探测对比图.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 图3：热力图
    target_scenario = SCENARIOS[-1]
    df_sub = df_inter[df_inter['情景'] == target_scenario]
    factor_names = list(FACTOR_MAP.values())
    matrix = pd.DataFrame(index=factor_names, columns=factor_names, dtype=float)
    for name in factor_names:
        q_single = df_factor[(df_factor['情景'] == target_scenario) & (df_factor['因子'] == name)]['q值'].values[0]
        matrix.loc[name, name] = q_single
    for idx, row in df_sub.iterrows():
        matrix.loc[row['因子A'], row['因子B']] = row['q(A∩B)']
        matrix.loc[row['因子B'], row['因子A']] = row['q(A∩B)']

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".4f", cmap="YlOrRd", ax=ax, linewidths=.5)
    ax.set_title(f'{target_scenario} - 驱动因子q值矩阵', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, f"3_{target_scenario}_交互矩阵热力图.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 所有结果已保存至: {OUTPUT_DIR}")
    print("\n🎉 分析完成！请查看CSV文件中的【P值】列确认显著性。")


if __name__ == "__main__":
    main()