import geopandas as gpd
import pandas as pd

# -------------------------- 1. 配置文件路径（无需修改） --------------------------
geohash_shp_path = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\E_and_V_calculation\辅助数据\clear_geohash_point_core.shp"
building_shp_path = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\E_and_V_calculation\辅助数据/building_point_core.shp"
output_shp_path = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\E_and_V_calculation\辅助数据\geohash_building_volume_sum.shp"

# -------------------------- 2. 读取数据 --------------------------
try:
    gdf_geohash = gpd.read_file(geohash_shp_path)
    gdf_building = gpd.read_file(building_shp_path)
    print("✅ 数据读取成功")
    print(f"geohash点数量：{len(gdf_geohash)}")
    print(f"建筑点数量：{len(gdf_building)}")
except Exception as e:
    raise Exception(f"❌ 读取shp失败：{e}")

# -------------------------- 3. 转换为投影坐标系（米为单位，解决匹配失真） --------------------------
# 自动匹配UTM投影，手动指定可替换为 target_crs = "EPSG:32650"（示例，按你的区域改）
target_crs = gdf_geohash.estimate_utm_crs() if gdf_geohash.crs else "EPSG:32650"
gdf_geohash = gdf_geohash.to_crs(target_crs)
gdf_building = gdf_building.to_crs(target_crs)
print(f"✅ 转换为投影坐标系（米）：{target_crs}")

# -------------------------- 4. 校验并处理area/Height（数值转换+过滤无效值） --------------------------
required_cols = ["area", "Height"]
if not all(col in gdf_building.columns for col in required_cols):
    raise ValueError(f"❌ 建筑点缺少属性：{[col for col in required_cols if col not in gdf_building.columns]}")

# 强转数值，过滤空值/非正数
for col in required_cols:
    gdf_building[col] = pd.to_numeric(gdf_building[col], errors="coerce")
gdf_building = gdf_building[(gdf_building["area"]>0) & (gdf_building["Height"]>0)].dropna(subset=required_cols)
eff_build = len(gdf_building)
print(f"✅ 过滤后有效建筑点：{eff_build}")

# -------------------------- 5. 计算体积（area×Height） --------------------------
gdf_building["volume"] = gdf_building["area"] * gdf_building["Height"]
print("✅ 完成建筑体积计算：volume = area × Height")

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

# -------------------------- 8. 核心修复：无列名依赖的聚合（彻底杜绝KeyError） --------------------------
# 聚合逻辑：按geohash_id分组，volume求和（体积总），直接计数行数（建筑点数量）
# 无需任何带后缀的列，直接用grouper.size()计数→绝对无列名问题
gdf_agg = gdf_sjoin.groupby(geo_id_col, as_index=False).agg(
    building_volume_sum=("volume", "sum"),  # 体积总和
    building_count=("volume", lambda x: x.count())  # 非空计数=建筑点数量，或直接用size()
)
# 【备选更稳妥的计数方式】直接统计分组内的行数（完全无列依赖），取消下面注释即可
# gdf_agg = gdf_sjoin.groupby(geo_id_col, as_index=False).agg(
#     building_volume_sum=("volume", "sum"),
#     building_count=("geometry", "size")  # size()=统计分组内所有行数，无列依赖
# )

print("✅ 按GeoHash点聚合完成：体积和+建筑点数量")

# -------------------------- 9. 合并聚合结果与原始GeoHash点（恢复几何+所有属性） --------------------------
gdf_final = pd.merge(
    left=gdf_geohash,
    right=gdf_agg,
    on=geo_id_col,
    how="left"  # 保留所有GeoHash点，无匹配则为0
)

# 空值填充：无建筑点的GeoHash，数量和体积置0
gdf_final["building_count"] = gdf_final["building_count"].fillna(0).astype(int)
gdf_final["building_volume_sum"] = gdf_final["building_volume_sum"].fillna(0)

# 恢复GeoDataFrame格式，确保坐标系正确
gdf_final = gpd.GeoDataFrame(gdf_final, geometry="geometry", crs=target_crs)

# -------------------------- 10. 可选：转回EPSG:4326经纬度（方便GIS打开，无需修改） --------------------------
gdf_final = gdf_final.to_crs("EPSG:4326")
print("✅ 结果转回经纬度坐标系EPSG:4326，方便GIS软件打开")

# -------------------------- 11. 保存结果（utf-8+gbk兜底，解决中文乱码） --------------------------
try:
    gdf_final.to_file(output_shp_path, encoding="utf-8")
    print(f"✅ 结果保存成功：{output_shp_path}（utf-8）")
except:
    gdf_final.to_file(output_shp_path, encoding="gbk")
    print(f"✅ 结果保存成功：{output_shp_path}（gbk，解决中文乱码）")

# -------------------------- 12. 多对一匹配验证（核心看max_count是否>1） --------------------------
total_count = gdf_final["building_count"].sum()
has_bld_geo = len(gdf_final[gdf_final["building_count"]>0])
max_count = gdf_final["building_count"].max()
avg_count = total_count / has_bld_geo if has_bld_geo > 0 else 0

print("\n" + "="*50)
print("🔍 多对一匹配结果验证（核心看最大数量是否>1）")
print(f"  1. 匹配建筑点总数：{total_count}（应≈{match_build}）")
print(f"  2. 有建筑点的GeoHash数量：{has_bld_geo}")
print(f"  3. 单个GeoHash匹配最大建筑点数：{max_count} → ✅>1即多对一成功！")
print(f"  4. 平均每个GeoHash匹配建筑点数：{avg_count:.2f}")
print("="*50)

# 预览前10行核心结果
print("\n📌 前10行结果预览（GeoHashID | 建筑点数量 | 体积总和）：")
print(gdf_final[[geo_id_col, "building_count", "building_volume_sum"]].head(10))