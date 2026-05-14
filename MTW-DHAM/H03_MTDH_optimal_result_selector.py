import os
import shutil
import geopandas as gpd
from pathlib import Path

# -------------------------- 配置参数（请确认路径正确性） --------------------------
# 基础文件路径模板
BASE_PATH = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\output_enhanced_scs_cn_flood_more_enhanced\hazard_index_output\hazard_index_results_{rain_grade}_hour_{hour}.shp"
# 降雨等级列表
RAIN_GRADES = ["小雨", "中雨", "大雨", "暴雨", "大暴雨", "特大暴雨"]
# 小时范围（01-24）
HOURS = [f"{h:02d}" for h in range(1, 25)]
# 危险性指数核心字段名（已修改为H_index）
HAZARD_FIELD = "H_index"
# 东西城区边界文件路径
DISTRICT_BOUND_PATH = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\北京市边界_110000_Shapefile_(poi86.com)\东西城区.shp"
# 结果输出目录（筛选出的6个文件会保存在这里，文件名与原文件一致）
OUTPUT_DIR = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\max_hazard_results"
# 坐标系匹配（如果边界和数据坐标系不一致，需修改此参数，默认WGS84: EPSG:4326）
TARGET_CRS = "EPSG:4326"


# --------------------------------------------------------------------------------

def load_district_boundary():
    """加载东西城区边界文件，并统一坐标系"""
    try:
        district_gdf = gpd.read_file(DISTRICT_BOUND_PATH)
        # 统一坐标系
        district_gdf = district_gdf.to_crs(TARGET_CRS)
        print(f"✅ 成功加载东西城区边界文件，包含 {len(district_gdf)} 个面要素")
        return district_gdf
    except Exception as e:
        print(f"❌ 加载边界文件失败：{str(e)}")
        raise SystemExit(1)


def get_avg_hazard_in_district(rain_grade, district_gdf):
    """
    获取指定降雨等级下，东西城区范围内H_index平均值最高的shp文件
    :param rain_grade: 降雨等级（如"小雨"）
    :param district_gdf: 东西城区边界GeoDataFrame
    :return: 最大平均值、对应的文件路径
    """
    max_avg_value = -float("inf")
    max_file = None

    for hour in HOURS:
        # 构建文件路径
        file_path = BASE_PATH.format(rain_grade=rain_grade, hour=hour)
        file_path = Path(file_path)

        # 检查文件是否存在
        if not file_path.exists():
            print(f"⚠️  文件不存在，跳过：{file_path}")
            continue

        try:
            # 读取shp文件并统一坐标系
            gdf = gpd.read_file(file_path)
            gdf = gdf.to_crs(TARGET_CRS)

            # 检查核心字段是否存在
            if HAZARD_FIELD not in gdf.columns:
                print(f"⚠️  字段 {HAZARD_FIELD} 不存在，跳过：{file_path}")
                continue

            # 空间裁剪：只保留东西城区范围内的数据
            # 使用intersection确保只保留重叠部分
            clipped_gdf = gpd.overlay(gdf, district_gdf, how="intersection")

            if clipped_gdf.empty:
                print(f"⚠️  东西城区范围内无数据，跳过：{file_path}")
                continue

            # 计算该文件在东西城区内的H_index平均值
            current_avg = clipped_gdf[HAZARD_FIELD].mean()

            # 更新最大值和对应文件
            if current_avg > max_avg_value:
                max_avg_value = current_avg
                max_file = file_path

        except Exception as e:
            print(f"❌ 处理文件失败：{file_path}，错误信息：{str(e)}")
            continue

    return max_avg_value, max_file


def main():
    # 1. 创建输出目录（若不存在）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 输出目录已准备：{OUTPUT_DIR}")

    # 2. 加载东西城区边界
    district_gdf = load_district_boundary()

    # 3. 存储每个降雨等级的筛选结果
    results = {}

    # 4. 遍历所有降雨等级，筛选最优文件
    print("\n" + "-" * 60)
    print("开始筛选每个降雨等级的最高平均危险性文件...")
    print("-" * 60)
    for grade in RAIN_GRADES:
        print(f"\n🔍 正在处理：{grade}")
        max_avg_val, max_file = get_avg_hazard_in_district(grade, district_gdf)

        if max_file:
            # 记录结果
            results[grade] = {
                "avg_h_index": max_avg_val,
                "file_path": max_file,
                "hour": max_file.name.split("_hour_")[1].split(".")[0]
            }

            # 复制文件到输出目录（保留原文件名）
            dest_file = os.path.join(OUTPUT_DIR, max_file.name)
            # 复制所有关联文件（.shp/.shx/.dbf/.prj等）
            for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"]:
                src_file = str(max_file).replace(".shp", ext)
                dest_file_ext = dest_file.replace(".shp", ext)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dest_file_ext)

            print(f"✅ 筛选完成：")
            print(f"   东西城区平均H_index：{max_avg_val:.6f}")
            print(f"   对应小时：{results[grade]['hour']}")
            print(f"   源文件：{max_file}")
            print(f"   输出文件：{dest_file}")
        else:
            results[grade] = {"avg_h_index": None, "file_path": None, "hour": None}
            print(f"❌ 未找到{grade}的有效文件")

    # 5. 打印最终汇总结果
    print("\n" + "=" * 60)
    print("📊 筛选结果汇总（每个降雨等级最优文件）")
    print("=" * 60)
    print(f"{'降雨等级':<8} {'平均H_index':<15} {'对应小时':<8} {'输出文件名':<50}")
    print("-" * 60)
    for grade, info in results.items():
        if info["avg_h_index"]:
            file_name = info["file_path"].name if info["file_path"] else "无"
            print(f"{grade:<8} {info['avg_h_index']:<15.6f} {info['hour']:<8} {file_name:<50}")
        else:
            print(f"{grade:<8} {'无数据':<15} {'无':<8} {'无':<50}")

    print(f"\n🎉 筛选完成！共筛选出 {len([v for v in results.values() if v['file_path']])} 个文件，已保存至：")
    print(f"   {OUTPUT_DIR}")


if __name__ == "__main__":
    # 设置geopandas警告过滤（可选）
    import warnings

    warnings.filterwarnings("ignore")
    main()