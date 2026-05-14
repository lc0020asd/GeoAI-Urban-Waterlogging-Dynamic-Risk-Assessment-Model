import geopandas as gpd
import pandas as pd
import numpy as np

# -------------------------- 1. 配置文件路径（按需修改） --------------------------
# Geohash网格点SHP路径
geohash_shp_path = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\building_core\clear_geohash_point_core.shp"
# 带V_resilience字段的建筑物点SHP路径（上一步输出的文件）
building_shp_path = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\building_core\building_point_core_with_resilience.shp"
# 输出网格点SHP路径（新增V_resilience_mean列）
output_shp_path = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\building_core\geohash_building_resilience_mean.shp"

# -------------------------- 2. 读取数据 --------------------------
try:
    gdf_geohash = gpd.read_file(geohash_shp_path)
    gdf_building = gpd.read_file(building_shp_path, encoding='utf-8')
    print("✅ 数据读取成功")
    print(f"geohash点数量：{len(gdf_geohash)}")
    print(f"建筑点数量：{len(gdf_building)}")
except Exception as e:
    raise Exception(f"❌ 读取shp失败：{e}")

# -------------------------- 3. 校验V_resilience字段是否存在 --------------------------
resilience_col = "V_resilien"
if resilience_col not in gdf_building.columns:
    raise ValueError(
        f"❌ 建筑点缺少{resilience_col}字段！\n"
        f"💡 请确认建筑点SHP是上一步输出的带抗灾能力评分的文件，字段名严格为{resilience_col}"
    )

# -------------------------- 4. 转换为投影坐标系（米为单位，解决匹配失真） --------------------------
# 自动匹配UTM投影，手动指定可替换为 target_crs = "EPSG:32650"（示例，按你的区域改）
target_crs = gdf_geohash.estimate_utm_crs() if gdf_geohash.crs else "EPSG:32650"
gdf_geohash = gdf_geohash.to_crs(target_crs)
gdf_building = gdf_building.to_crs(target_crs)
print(f"✅ 转换为投影坐标系（米）：{target_crs}")

# -------------------------- 5. 校验并处理V_resilience（数值转换+过滤无效值） --------------------------
# 强转数值，过滤空值/非0-1范围的异常值
gdf_building[resilience_col] = pd.to_numeric(gdf_building[resilience_col], errors="coerce")
gdf_building = gdf_building[
    (gdf_building[resilience_col] >= 0) & (gdf_building[resilience_col] <= 1)
].dropna(subset=[resilience_col])
eff_build = len(gdf_building)
print(f"✅ 过滤后有效建筑点（V_resilience∈[0,1]）：{eff_build}")

# -------------------------- 6. GeoHash点新增唯一ID（分组核心，无列名依赖） --------------------------
gdf_geohash["geohash_id"] = range(len(gdf_geohash))
geo_id_col = "geohash_id"  # 固定分组列名，全程复用
print(f"✅ GeoHash点新增唯一分组列：{geo_id_col}")

# -------------------------- 7. 空间近邻连接（多对一核心：保留所有建筑点） --------------------------
gdf_sjoin = gpd.sjoin_nearest(
    left_df=gdf_geohash,
    right_df=gdf_building,
    how="right",               # 右连接=保留所有建筑点→多对一核心
    max_distance=300,          # 按网格大小调：100/200/500（米）
    lsuffix="_geo",
    rsuffix="_bld",
    distance_col="match_dist_m"# 匹配距离列（米），可查看建筑点到GeoHash的距离
)
# 过滤无匹配的建筑点（超出max_distance）
gdf_sjoin = gdf_sjoin[gdf_sjoin[geo_id_col].notna()]
match_build = len(gdf_sjoin)
print(f"✅ 近邻匹配完成：成功匹配{match_build}个建筑点（{match_build/eff_build:.2%}）")

# -------------------------- 8. 核心修改：按网格聚合V_resilience均值 --------------------------
# 聚合逻辑：按geohash_id分组，计算V_resilience均值 + 建筑点数量
gdf_agg = gdf_sjoin.groupby(geo_id_col, as_index=False).agg(
    V_resilience_mean=(resilience_col, "mean"),  # 网格内建筑抗灾能力均值
    building_count=("geometry", "size")          # 网格内建筑点数量（无列依赖）
)
# 保留4位小数，便于后续分析
gdf_agg["V_resilience_mean"] = gdf_agg["V_resilience_mean"].round(4)

print("✅ 按GeoHash点聚合完成：V_resilience均值 + 建筑点数量")

# -------------------------- 9. 合并聚合结果与原始GeoHash点（恢复几何+所有属性） --------------------------
gdf_final = pd.merge(
    left=gdf_geohash,
    right=gdf_agg,
    on=geo_id_col,
    how="left"  # 保留所有GeoHash点，无匹配则为NaN
)

# 空值填充：无建筑点的GeoHash，均值设为NaN（或按需设为0/1），数量置0
gdf_final["building_count"] = gdf_final["building_count"].fillna(0).astype(int)
# 可选：无建筑点的网格均值设为0（代表无抗灾能力风险），取消下面注释即可
# gdf_final["V_resilience_mean"] = gdf_final["V_resilience_mean"].fillna(0)

# 恢复GeoDataFrame格式，确保坐标系正确
gdf_final = gpd.GeoDataFrame(gdf_final, geometry="geometry", crs=target_crs)

# -------------------------- 10. 转回EPSG:4326经纬度（方便GIS打开） --------------------------
gdf_final = gdf_final.to_crs("EPSG:4326")
print("✅ 结果转回经纬度坐标系EPSG:4326，方便GIS软件打开")

# -------------------------- 11. 保存结果（utf-8+gbk兜底，解决中文乱码） --------------------------
try:
    gdf_final.to_file(output_shp_path, encoding="utf-8")
    print(f"✅ 结果保存成功：{output_shp_path}（utf-8）")
except:
    gdf_final.to_file(output_shp_path, encoding="gbk")
    print(f"✅ 结果保存成功：{output_shp_path}（gbk，解决中文乱码）")

# -------------------------- 12. 多对一匹配验证 --------------------------
total_count = gdf_final["building_count"].sum()
has_bld_geo = len(gdf_final[gdf_final["building_count"]>0])
max_count = gdf_final["building_count"].max()
avg_count = total_count / has_bld_geo if has_bld_geo > 0 else 0

print("\n" + "="*60)
print("🔍 网格建筑抗灾能力均值计算结果验证")
print(f"  1. 匹配建筑点总数：{total_count}（应≈{match_build}）")
print(f"  2. 有建筑点的GeoHash数量：{has_bld_geo}")
print(f"  3. 单个GeoHash匹配最大建筑点数：{max_count} → ✅>1即多对一成功！")
print(f"  4. 平均每个GeoHash匹配建筑点数：{avg_count:.2f}")
print(f"  5. 网格V_resilience均值范围：{gdf_final['V_resilience_mean'].min():.4f} ~ {gdf_final['V_resilience_mean'].max():.4f}")
print("="*60)

# 预览前10行核心结果
print("\n📌 前10行结果预览（GeoHashID | 建筑点数量 | V_resilience均值）：")
print(gdf_final[[geo_id_col, "building_count", "V_resilience_mean"]].head(10))