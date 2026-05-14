import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
import geopandas as gpd
import numpy as np
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ====================== 全局配置 ======================
# 路径配置
BASE_PATH = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\output_enhanced_scs_cn_flood_more_enhanced\hazard_index_output"
SHP_PATH = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\北京市边界_110000_Shapefile_(poi86.com)\东西城区.shp"
# 降雨等级列表（需与shp文件名中的降雨等级匹配）
RAIN_GRADES = ["小雨", "中雨", "大雨", "暴雨", "大暴雨", "特大暴雨"]
# 小时列表（0-23）
HOURS = list(range(1,25))
# 危险性配色：黄→红（低→高，符合危险性认知）
HAZARD_CMAP = "YlOrRd"
# 输出目录（生成的热力图保存路径）
OUTPUT_DIR = os.path.join(BASE_PATH, "危险性热力图结果_东西城区")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====================== 空间筛选函数 ======================
def load_district_boundary():
    """加载东西城区边界，并统一坐标系为WGS84（EPSG:4326）"""
    try:
        district_gdf = gpd.read_file(SHP_PATH)
        # 统一坐标系为WGS84
        if district_gdf.crs != "EPSG:4326":
            district_gdf = district_gdf.to_crs("EPSG:4326")
        print(f"✅ 成功加载东西城区边界，坐标系：{district_gdf.crs}")
        return district_gdf
    except Exception as e:
        raise FileNotFoundError(f"加载边界文件失败：{str(e)}")


def filter_hazard_by_district(hazard_gdf, district_gdf):
    """
    空间筛选：保留东西城区内的危险性数据
    :param hazard_gdf: 原始危险性GeoDataFrame
    :param district_gdf: 东西城区边界GeoDataFrame
    :return: 筛选后的GeoDataFrame
    """
    # 统一坐标系
    if hazard_gdf.crs != district_gdf.crs:
        hazard_gdf = hazard_gdf.to_crs(district_gdf.crs)

    # 空间裁剪
    filtered_gdf = gpd.clip(hazard_gdf, district_gdf)
    print(f"🔍 空间筛选完成：原始数据{len(hazard_gdf)}条 → 东西城区内{len(filtered_gdf)}条")
    return filtered_gdf


# ====================== 危险性数据处理函数 ======================
def load_hazard_data(rain_grade, hour, district_gdf):
    """
    加载指定降雨等级、指定小时的危险性shp文件，并进行空间筛选
    :param rain_grade: 降雨等级（如"特大暴雨"）
    :param hour: 小时数（0-23）
    :param district_gdf: 东西城区边界GeoDataFrame
    :return: 筛选后的H_index均值（该时段该等级的平均危险性）
    """
    # 构建shp文件路径（匹配文件名格式：hazard_index_results_特大暴雨_hour_14.shp）
    file_name = f"hazard_index_results_{rain_grade}_hour_{hour:02d}.shp"
    file_path = os.path.join(BASE_PATH, file_name)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"⚠️  危险性文件不存在：{file_path}")
        return np.nan

    # 读取shp文件
    try:
        hazard_gdf = gpd.read_file(file_path)
    except Exception as e:
        print(f"❌ 读取{rain_grade}_hour_{hour}文件失败：{str(e)}")
        return np.nan

    # 空间筛选
    filtered_gdf = filter_hazard_by_district(hazard_gdf, district_gdf)

    # 计算该时段该等级的H_index均值（过滤无效值）
    if len(filtered_gdf) > 0 and "H_index" in filtered_gdf.columns:
        mean_hazard = filtered_gdf["H_index"].replace([np.inf, -np.inf], np.nan).mean()
        return mean_hazard if not np.isnan(mean_hazard) else 0.0
    else:
        print(f"⚠️  {rain_grade}_hour_{hour}无有效H_index数据")
        return 0.0


def process_hazard_data(district_gdf, only_calculate_extremes=False):
    """
    处理所有降雨等级×小时的危险性数据，生成6×24的均值矩阵
    :param district_gdf: 东西城区边界GeoDataFrame
    :param only_calculate_extremes: 仅计算全局极值（用于统一色阶）
    :return: 6×24的均值矩阵（DataFrame）/ 全局极值（only_calculate_extremes=True）
    """
    # 初始化结果字典
    hazard_data = {}

    # 全局极值初始化
    global_min = float("inf")
    global_max = -float("inf")

    # 遍历所有降雨等级和小时
    for rain_grade in RAIN_GRADES:
        hazard_data[rain_grade] = []
        for hour in HOURS:
            print(f"\n📊 处理 {rain_grade} - {hour:02d}时 危险性数据...")
            mean_h = load_hazard_data(rain_grade, hour, district_gdf)

            # 更新全局极值
            if not np.isnan(mean_h):
                global_min = min(global_min, mean_h)
                global_max = max(global_max, mean_h)

            hazard_data[rain_grade].append(mean_h)

    if only_calculate_extremes:
        # 仅返回全局极值
        return global_min, global_max
    else:
        # 转换为DataFrame（行：降雨等级，列：小时）
        hazard_df = pd.DataFrame(
            hazard_data,
            index=HOURS
        ).T  # 转置：行=降雨等级，列=小时
        # 填充空值
        hazard_df = hazard_df.fillna(0)
        return hazard_df


# ====================== 热力图绘制函数 ======================
def plot_hazard_heatmap(data, title, save_name, cmap, vmin=None, vmax=None):
    """
    绘制6×24危险性热力图（降雨等级×小时）
    :param data: 6×24的均值矩阵（DataFrame）
    :param title: 图表标题
    :param save_name: 保存文件名
    :param cmap: 配色方案
    :param vmin: 色阶最小值
    :param vmax: 色阶最大值
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布（适配6行24列的布局）
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # 绘制热力图
    im = sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        annot=False,
        linewidths=0.2,
        linecolor="white",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={
            "label": "危险性指数（H_index）均值",
            "shrink": 0.8,
            "pad": 0.02,
            "format": mticker.FormatStrFormatter('%.3f')  # 保留三位小数
        }
    )

    # 美化设置
    ax.set_title(title, fontsize=14, pad=20, fontweight="bold")
    ax.set_xlabel("小时", fontsize=16, fontweight="bold")
    ax.set_ylabel("降雨等级", fontsize=16, fontweight="bold")
    # X轴：小时（两位数显示）
    ax.set_xticklabels([f"{h:02d}" for h in HOURS], fontsize=12)
    # Y轴：降雨等级
    ax.set_yticklabels(data.index, fontsize=12)

    # 调整色条标签格式
    cbar = im.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    # 保存
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 危险性热力图已保存：{save_path}")


# ====================== 主函数 ======================
def main():
    # 1. 加载东西城区边界
    print("🔍 开始加载东西城区边界...")
    try:
        district_gdf = load_district_boundary()
    except Exception as e:
        print(f"❌ 边界加载失败：{str(e)}")
        return

    # 2. 先计算所有降雨等级×小时的全局极值（统一色阶）
    print("\n🔍 计算危险性全局极值（统一色阶）...")
    hazard_global_min = float("inf")
    hazard_global_max = -float("inf")

    try:
        min_val, max_val = process_hazard_data(district_gdf, only_calculate_extremes=True)
        hazard_global_min = min(hazard_global_min, min_val)
        hazard_global_max = max(hazard_global_max, max_val)
    except Exception as e:
        print(f"⚠️  极值计算失败，将使用数据自身极值：{str(e)}")
        hazard_global_min = None
        hazard_global_max = None

    print(f"📊 危险性全局极值：min={hazard_global_min:.3f}, max={hazard_global_max:.3f}")

    # 3. 处理完整危险性数据并绘制热力图
    print("\n🔍 开始处理危险性数据并绘制热力图...")
    try:
        hazard_df = process_hazard_data(district_gdf, only_calculate_extremes=False)

        # 绘制危险性热力图（6×24：降雨等级×小时）
        plot_hazard_heatmap(
            data=hazard_df,
            title="东西城区 - 不同降雨等级×小时 洪涝危险性指数均值热力图",
            save_name="东西城区_洪涝危险性_6×24热力图.png",
            cmap=HAZARD_CMAP,
            vmin=hazard_global_min,
            vmax=hazard_global_max
        )

        # 额外：为每个降雨等级单独绘制24小时热力图（可选）
        for rain_grade in RAIN_GRADES:
            if rain_grade in hazard_df.index:
                single_grade_data = hazard_df.loc[[rain_grade]]  # 1×24矩阵
                plot_hazard_heatmap(
                    data=single_grade_data,
                    title=f"东西城区 - {rain_grade} 洪涝危险性指数24小时均值热力图",
                    save_name=f"东西城区_{rain_grade}_危险性_24小时热力图.png",
                    cmap=HAZARD_CMAP,
                    vmin=hazard_global_min,
                    vmax=hazard_global_max
                )

    except Exception as e:
        print(f"❌ 危险性热力图绘制失败：{str(e)}")
        return

    print("\n🎉 危险性热力图生成完成！")
    print(f"📁 热力图结果保存在：{OUTPUT_DIR}")


if __name__ == "__main__":
    # 检查依赖
    try:
        import geopandas
    except ImportError:
        print("🔧 正在安装geopandas依赖...")
        os.system("pip install geopandas")
        import geopandas

    main()