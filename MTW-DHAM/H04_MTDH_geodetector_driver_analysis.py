import os
import itertools  # 【新增】用于生成两两因子组合
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
from tqdm import tqdm

# ====================== 1. 全局配置 ======================
BASE_PATH = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据"
BOUNDARY_PATH = os.path.join(BASE_PATH, "北京市边界_110000_Shapefile_(poi86.com)", "东西城区.shp")
HAZARD_ROOT = os.path.join(BASE_PATH, "output_enhanced_scs_cn_flood_more_enhanced", "hazard_index_output")
DEM_PATH = os.path.join(BASE_PATH, "DEM", "dem_chengliuqu.tif")
LUCC_PATH = os.path.join(BASE_PATH, "LUCC_bj", "CLCD_v01_2021_albert_province", "CLCD_v01_2021_albert_beijing_pro.tif")
SOIL_PATH = os.path.join(BASE_PATH, "rain_point", "水文土壤数据", "HYSOGs250mclip.tif")
OUTPUT_DIR = os.path.join(BASE_PATH, "地理探测器结果_分等级独立_含交互探测")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 降雨等级列表
RAIN_GRADES = ["小雨", "中雨", "大雨", "暴雨", "大暴雨", "特大暴雨"]
HOURS = [f"{h:02d}" for h in range(1, 25)]

# 定义因子名称映射
FACTOR_MAP = {
    'DEM_CAT': 'DEM(地形)',
    'LUCC': 'LUCC(地表覆盖)',
    'SOIL': 'SOIL(水文土壤组)'
}


# ====================== 2. 核心辅助函数 (新增交互探测相关) ======================
def load_and_reproject_boundary():
    return gpd.read_file(BOUNDARY_PATH)


def preprocess_rasters(boundary_gdf):
    temp_files = {}
    print("🔧 正在预先裁剪栅格数据...")
    for name, path in [('DEM', DEM_PATH), ('LUCC', LUCC_PATH), ('SOIL', SOIL_PATH)]:
        temp_path = os.path.join(OUTPUT_DIR, f"temp_{name}.tif")
        if not os.path.exists(temp_path):
            with rasterio.open(path) as src:
                boundary_reproj = boundary_gdf.to_crs(src.crs)
                geoms = [mapping(geom) for geom in boundary_reproj.geometry]
                out_image, out_transform = mask(src, geoms, crop=True, nodata=src.nodata)
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
                                 "transform": out_transform})
                with rasterio.open(temp_path, "w", **out_meta) as dest:
                    dest.write(out_image)
        temp_files[name] = temp_path
    return temp_files


def extract_values(shp_path, raster_paths):
    try:
        gdf = gpd.read_file(shp_path)
        if len(gdf) == 0: return None
    except:
        return None

    for name, path in raster_paths.items():
        with rasterio.open(path) as src:
            gdf_reproj = gdf.to_crs(src.crs)
            coords = [(x, y) for x, y in zip(gdf_reproj.geometry.x, gdf_reproj.geometry.y)]
            vals = [val[0] if val[0] != src.nodata else np.nan for val in src.sample(coords)]
            gdf[name] = vals
    return gdf[['H_index', 'DEM', 'LUCC', 'SOIL']]


def calculate_q_value(y, x_strata):
    """计算单因子q值和P值 (保留原逻辑)"""
    df = pd.DataFrame({'y': y, 'strata': x_strata}).dropna()
    if len(df) < 20: return 0.0, 1.0

    mean_total = df['y'].mean()
    sst = np.sum((df['y'] - mean_total) ** 2)
    if sst == 0: return 0.0, 1.0

    ssw = 0.0
    for name, group in df.groupby('strata'):
        if len(group) < 2: continue
        ssw += np.sum((group['y'] - group['y'].mean()) ** 2)

    q = 1 - (ssw / sst)

    N, L = len(df), df['strata'].nunique()
    df1, df2 = L - 1, N - L
    if df2 <= 0: return q, 1.0
    F = ((sst - ssw) / df1) / (ssw / df2)
    p = stats.f.sf(F, df1, df2)
    return q, p


def calculate_q_only(y, x_strata):
    """【新增】仅计算q值，用于交互探测 (提高效率)"""
    df = pd.DataFrame({'y': y, 'strata': x_strata}).dropna()
    if len(df) < 20: return 0.0

    mean_total = df['y'].mean()
    sst = np.sum((df['y'] - mean_total) ** 2)
    if sst == 0: return 0.0

    ssw = 0.0
    for name, group in df.groupby('strata'):
        if len(group) < 2: continue
        ssw += np.sum((group['y'] - group['y'].mean()) ** 2)

    q = 1 - (ssw / sst)
    return q


def judge_interaction_type(q1, q2, q_inter):
    """【新增】判断交互探测类型 (基于王劲峰老师的定义)"""
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


# ====================== 3. 主程序 (重构，含交互探测) ======================
def main():
    print("=" * 60)
    print("🚀 分降雨等级独立地理探测器分析 (含因子探测+交互探测)")
    print("=" * 60)

    gdf_boundary = load_and_reproject_boundary()
    raster_paths = preprocess_rasters(gdf_boundary)

    summary_results = []  # 存储单因子结果
    interaction_results = []  # 【新增】存储交互探测结果
    factor_pairs = list(itertools.combinations(FACTOR_MAP.keys(), 2))  # 【新增】生成两两因子组合

    # 外层循环：遍历6个降雨等级
    for rain_grade in RAIN_GRADES:
        print(f"\n{'=' * 60}")
        print(f"📊 正在处理等级: 【{rain_grade}】")
        print(f"{'=' * 60}")

        # 1. 读取该等级下的24个小时数据
        grade_data_list = []
        missing_count = 0

        pbar = tqdm(HOURS, desc=f"读取{rain_grade}数据")
        for hour_str in pbar:
            fname = f"hazard_index_results_{rain_grade}_hour_{hour_str}.shp"
            fpath = os.path.join(HAZARD_ROOT, fname)

            if os.path.exists(fpath):
                df_sub = extract_values(fpath, raster_paths)
                if df_sub is not None:
                    grade_data_list.append(df_sub)
            else:
                missing_count += 1

        if len(grade_data_list) == 0:
            print(f"⚠️  {rain_grade} 无有效数据，跳过")
            continue

        # 2. 合并该等级数据
        df_grade = pd.concat(grade_data_list, ignore_index=True)
        df_grade = df_grade.dropna(subset=['H_index', 'DEM', 'LUCC', 'SOIL'])
        print(f"   样本量: {len(df_grade):,} 条 (缺失文件: {missing_count})")

        # 3. 预处理
        df_grade['LUCC'] = df_grade['LUCC'].astype(int)
        df_grade['SOIL'] = df_grade['SOIL'].astype(int)
        try:
            df_grade['DEM_CAT'] = mc.NaturalBreaks(df_grade['DEM'].dropna(), k=6).yb
        except:
            df_grade['DEM_CAT'] = pd.qcut(df_grade['DEM'], q=6, labels=False, duplicates='drop')

        # 4. 运行单因子地理探测器
        print(f"\n   --- {rain_grade} 单因子探测结果 ---")
        grade_res = {'降雨等级': rain_grade}
        factor_q_values = {}  # 【新增】暂存单因子q值，用于交互对比

        for col, name in FACTOR_MAP.items():
            q, p_val = calculate_q_value(df_grade['H_index'], df_grade[col])
            factor_q_values[col] = q
            grade_res[name] = q
            grade_res[f"{name}_P"] = p_val
            print(f"   {name}: q={q:.4f} (P={p_val:.4f})")

        summary_results.append(grade_res)

        # 5. 【核心新增】运行交互探测
        print(f"\n   --- {rain_grade} 交互探测结果 ---")
        for (f1, f2) in factor_pairs:
            # 生成交互分层：两个因子分类的组合
            df_grade['interaction_strata'] = df_grade[[f1, f2]].apply(tuple, axis=1)

            q1 = factor_q_values[f1]
            q2 = factor_q_values[f2]
            q_inter = calculate_q_only(df_grade['H_index'], df_grade['interaction_strata'])
            inter_type = judge_interaction_type(q1, q2, q_inter)

            # 保存交互结果
            inter_res = {
                '降雨等级': rain_grade,
                '因子A': FACTOR_MAP[f1],
                '因子B': FACTOR_MAP[f2],
                'q(A)': q1,
                'q(B)': q2,
                'q(A∩B)': q_inter,
                '交互类型': inter_type
            }
            interaction_results.append(inter_res)
            print(f"   {FACTOR_MAP[f1]} ∩ {FACTOR_MAP[f2]}: q={q_inter:.4f} ({inter_type})")

        # 6. 保存该等级的详细结果
        grade_out_dir = os.path.join(OUTPUT_DIR, rain_grade)
        os.makedirs(grade_out_dir, exist_ok=True)
        df_grade.to_csv(os.path.join(grade_out_dir, f"{rain_grade}_采样数据.csv"), index=False)

        # 绘制该等级的单因子图 (保留原逻辑)
        res_df_single = pd.DataFrame({
            '影响因素': list(FACTOR_MAP.values()),
            'q统计量': [grade_res[name] for name in FACTOR_MAP.values()]
        })

        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='影响因素', y='q统计量', data=res_df_single, ax=ax, hue='影响因素', palette='viridis',
                    legend=False)

        for p in ax.patches:
            h = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., h + 0.01, f'{h:.4f}', ha='center')

        ax.set_title(f"{rain_grade} - 单因子驱动探测", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(grade_out_dir, f"{rain_grade}_单因子探测.png"), dpi=300)
        plt.close()

    # ====================== 4. 生成综合对比图 (含交互) ======================
    print("\n" + "=" * 60)
    print("📈 生成综合对比图与汇总表")

    # 保存单因子汇总表
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "1_单因子探测结果汇总表.csv"), index=False, encoding='utf-8-sig')

    # 【新增】保存交互探测汇总表
    df_inter_summary = pd.DataFrame(interaction_results)
    df_inter_summary.to_csv(os.path.join(OUTPUT_DIR, "2_交互探测结果汇总表.csv"), index=False, encoding='utf-8-sig')

    # 绘图全局配置
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 图1：单因子探测6等级综合对比图 (保留原逻辑) ---
    df_plot = df_summary.melt(id_vars=['降雨等级'], value_vars=list(FACTOR_MAP.values()), var_name='影响因素',
                              value_name='q统计量')
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=df_plot, x='降雨等级', y='q统计量', hue='影响因素', palette='coolwarm', ax=ax, order=RAIN_GRADES)
    ax.set_title('不同降雨等级下洪涝危险性单因子驱动演化对比 (q值)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('降雨等级 (从弱到强)', fontsize=14)
    ax.set_ylabel('q统计量 (解释力)', fontsize=14)
    ax.legend(title='影响因素', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_单因子探测演化对比图.png"), dpi=300, bbox_inches='tight')
    print(f"✅ 单因子综合对比图已保存")

    # --- 图2：【新增】交互探测6等级综合对比图 ---
    df_inter_plot = df_inter_summary.copy()
    df_inter_plot['交互对'] = df_inter_plot['因子A'] + " ∩ " + df_inter_plot['因子B']

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(data=df_inter_plot, x='降雨等级', y='q(A∩B)', hue='交互对', palette='coolwarm', ax=ax,
                order=RAIN_GRADES)
    ax.set_title('不同降雨等级下洪涝危险性驱动因子交互探测对比 (q值)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('降雨等级 (从弱到强)', fontsize=14)
    ax.set_ylabel('q统计量 (交互解释力)', fontsize=14)
    ax.legend(title='交互因子对', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_交互探测演化对比图.png"), dpi=300, bbox_inches='tight')
    print(f"✅ 交互探测综合对比图已保存")

    # --- 图3：【新增】特大暴雨情景交互矩阵热力图 (示例) ---
    target_rain_grade = RAIN_GRADES[-1]  # 取最后一个等级（特大暴雨）
    df_sub = df_inter_summary[df_inter_summary['降雨等级'] == target_rain_grade]
    factor_names = list(FACTOR_MAP.values())
    matrix = pd.DataFrame(index=factor_names, columns=factor_names, dtype=float)

    # 填充对角线（单因子q值）
    for name in factor_names:
        q_single = df_summary[df_summary['降雨等级'] == target_rain_grade][name].values[0]
        matrix.loc[name, name] = q_single

    # 填充非对角线（交互q值）
    for idx, row in df_sub.iterrows():
        matrix.loc[row['因子A'], row['因子B']] = row['q(A∩B)']
        matrix.loc[row['因子B'], row['因子A']] = row['q(A∩B)']

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".4f", cmap="YlOrRd", ax=ax, linewidths=.5)
    ax.set_title(f'{target_rain_grade} - 驱动因子q值矩阵 (对角为单因子，非对角为交互)', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, f"3_{target_rain_grade}_交互矩阵热力图.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {target_rain_grade}交互矩阵热力图已保存")

    print("\n🎉 所有分析完成！请查看: " + OUTPUT_DIR)


if __name__ == "__main__":
    main()