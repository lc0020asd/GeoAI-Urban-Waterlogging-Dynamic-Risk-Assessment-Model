
'''# ===================== 东西城区数据精准筛选：gh7网格+手机信令 =====================
# 适配大文件、高精度解码、结果落盘、数据校验
import pandas as pd
import pygeohash  # 高精度解码，安装：pip install geohash
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告

# ---------------------- 1. 配置核心参数（请确认路径和经纬度范围，无需修改其他） ----------------------
# 原文件路径
FISHNET_CSV = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\fishnet\geohash7\shp\result.csv'
DAAS_CSV = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\all_hourly_sum_chengliuqu_grid_matched.csv'
# 东西城区经纬度范围（核心区）
LON_MIN, LON_MAX = 116.315, 116.45  # 东经
LAT_MIN, LAT_MAX = 39.855, 39.975  # 北纬
# 分块大小（处理大信令文件，内存不足可调小，如2_000_000）
CHUNK_SIZE = 2_000_000
# 解码精度（保留6位小数，匹配geohash7理论精度）
DECIMAL_PREC = 6

# ---------------------- 2. 定义工具函数（高精度解码、路径处理） ----------------------
def high_prec_gh7_decode(gh_str):
    """
    geohash7高精度解码，返回(lat, lon)浮点数，异常值返回nan
    :param gh_str: geohash7字符串
    :return: (纬度lat, 经度lon) 或 (nan, nan)
    """
    try:
        lat, lon = pygeohash.decode(gh_str)
        return round(lat, DECIMAL_PREC), round(lon, DECIMAL_PREC)
    except:
        return np.nan, np.nan

def get_out_path(origin_path, suffix='_core_area'):
    """
    生成筛选后文件的路径：原目录+原文件名+suffix+原后缀
    例：result.csv → result_core_area.csv
    """
    p = Path(origin_path)
    return str(p.parent / f"{p.stem}{suffix}{p.suffix}")

# ---------------------- 3. 第一步：筛选东西城区范围内的gh7网格（result.csv） ----------------------
print("="*80)
print("🔍 开始筛选【东西城区有效gh7网格】—— result.csv")
print(f"筛选范围：经度{LON_MIN}~{LON_MAX}，纬度{LAT_MIN}~{LAT_MAX}")
print("="*80)

# 读取原gh7文件（仅保留geohash7列，后续解码）
gh7_origin = pd.read_csv(FISHNET_CSV, usecols=['geohash7'])
# 预处理：去重、过滤空值
gh7_origin = gh7_origin.drop_duplicates(subset=['geohash7']).dropna(subset=['geohash7'])
print(f"原gh7网格总数（去重后）：{len(gh7_origin):,}")

# 高精度解码geohash7为经纬度
gh7_origin[['lat', 'lon']] = pd.Series(
    gh7_origin['geohash7'].apply(high_prec_gh7_decode),
    index=gh7_origin.index
).apply(pd.Series).astype(float)

# 过滤解码失败的行+经纬度在东西城区范围内的行
gh7_core = gh7_origin.dropna(subset=['lat', 'lon'])
mask_core = (gh7_core['lon'] >= LON_MIN) & (gh7_core['lon'] <= LON_MAX) & \
            (gh7_core['lat'] >= LAT_MIN) & (gh7_core['lat'] <= LAT_MAX)
gh7_core = gh7_core[mask_core].reset_index(drop=True)

# 提取核心区有效gh7集合（用于后续筛选信令数据）
valid_gh7_set = set(gh7_core['geohash7'].tolist())
print(f"东西城区有效gh7网格数：{len(gh7_core):,}")
print(f"有效gh7占比：{len(gh7_core)/len(gh7_origin):.2%}")

# 读取原result.csv全量列，筛选有效gh7（保留所有原列，仅做行筛选）
gh7_full = pd.read_csv(FISHNET_CSV)
gh7_full_core = gh7_full[gh7_full['geohash7'].isin(valid_gh7_set)].reset_index(drop=True)

# 保存筛选后的gh7网格文件
gh7_out_path = get_out_path(FISHNET_CSV)
gh7_full_core.to_csv(gh7_out_path, index=False, encoding='utf-8')
print(f"✅ 东西城区gh7网格文件已保存：{gh7_out_path}")
print(f"✅ 保存行数：{len(gh7_full_core):,}（与有效gh7数一致）")
print("-"*80)

# ---------------------- 4. 第二步：筛选信令数据中核心区的行（匹配有效gh7） ----------------------
print("🔍 开始筛选【东西城区手机信令数据】—— all_hourly_sum_chengliuqu_grid_matched.csv")
print(f"分块大小：{CHUNK_SIZE:,} 行/块，有效gh7匹配筛选")
print("="*80)

# 定义信令文件输出路径
daas_out_path = get_out_path(DAAS_CSV)
# 初始化：记录是否是第一块（用于写入文件头）
is_first_chunk = True
# 记录筛选后总行数
total_filtered = 0

# 分块读取原信令文件，筛选有效gh7行
for idx, daas_chunk in enumerate(pd.read_csv(DAAS_CSV, chunksize=CHUNK_SIZE), 1):
    # 预处理：过滤final_grid_id空值
    daas_chunk = daas_chunk.dropna(subset=['final_grid_id'])
    # 筛选：final_grid_id在核心区有效gh7集合中
    mask_daas = daas_chunk['final_grid_id'].isin(valid_gh7_set)
    daas_chunk_core = daas_chunk[mask_daas].reset_index(drop=True)
    filtered_num = len(daas_chunk_core)
    total_filtered += filtered_num

    # 写入文件：第一块写表头，后续块追加（不写表头）
    daas_chunk_core.to_csv(
        daas_out_path,
        index=False,
        encoding='utf-8',
        header=is_first_chunk,
        mode='w' if is_first_chunk else 'a'
    )

    # 进度提示
    print(f"处理第{idx}块 → 本块原行数：{len(daas_chunk):,} → 筛选后行数：{filtered_num:,}")
    # 第一块后更新标记
    if is_first_chunk:
        is_first_chunk = False

# ---------------------- 5. 数据校验与结果汇总 ----------------------
print("="*80)
print("📊 东西城区数据筛选【最终结果汇总】")
print("="*80)
print(f"1. 原gh7网格文件（{Path(FISHNET_CSV).name}）：")
print(f"   去重后总行数：{len(gh7_origin):,} → 核心区有效行数：{len(gh7_full_core):,}")
print(f"   筛选后文件：{gh7_out_path}")
print(f"2. 原信令数据文件（{Path(DAAS_CSV).name}）：")
print(f"   核心区筛选后总行数：{total_filtered:,}")
print(f"   筛选后文件：{daas_out_path}")
print("="*80)
print("🎉 所有筛选完成！后续滑窗分块请直接使用上述两个_core_area后缀的文件！")'''



# ===================== 滑窗版断点续跑-空间分割版本【性能优化版 | 移除预过滤】 =====================
import pandas as pd
import numpy as np
from pathlib import Path
import pygeohash  # 安装：pip install pygeohash

# 0. 核心变量（已对接筛选后的_core_area文件，无需修改）
SRC_CSV = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\all_hourly_sum_chengliuqu_grid_matched_core_area.csv'
DST_DIR = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令'
FISHNET_CSV = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\fishnet\geohash7\shp\result_core_area.csv'
CHUNK_DAAS = 2_000_000
WIN_SIZE =  0.005  # 滑窗 0.005°×0.005°
STEP_SIZE = WIN_SIZE      # 不重叠；重叠则设为WIN_SIZE的1/2或1/3

# 优化：提前定义聚合列/规则、轻量化dtype（保留所有性能优化）
GROUP_COLS = ['city', 'final_grid_id', 'date', 'hour', 'gender', 'age']
AGG_RULES = {'ucnt': 'sum', 'usum': 'sum'}
# 东西城整体经纬度范围0.005
LON_MIN, LON_MAX = 116.315, 116.45
LAT_MIN, LAT_MAX = 39.855, 39.975

# 1. 读取geohash7并高精度解码（保留pygeohash解码，修复精度/列顺序问题）
gh7_df = (pd.read_csv(FISHNET_CSV, usecols=['geohash7'])
          .drop_duplicates()
          .dropna(subset=['geohash7']))
# pygeohash.decode返回(lat, lon)，列顺序匹配原逻辑，高精度浮点数
gh7_df[['lat', 'lon']] = pd.Series(gh7_df['geohash7'].apply(pygeohash.decode),
                                   index=gh7_df.index).apply(pd.Series).astype(float)

# 2. 构建滑窗（保留原逻辑，仅过滤空滑窗）
windows = []          # 每个元素： (block_id, lon0, lon1, lat0, lat1, gh7_list)
block_id = 0
lat0 = LAT_MIN
while lat0 < LAT_MAX:
    lat1 = lat0 + WIN_SIZE
    lon0 = LON_MIN
    while lon0 < LON_MAX:
        lon1 = lon0 + WIN_SIZE
        # 匹配滑窗内的gh7
        mask = (
            (gh7_df['lon'] >= lon0) & (gh7_df['lon'] < lon1) &
            (gh7_df['lat'] >= lat0) & (gh7_df['lat'] < lat1)
        )
        gh7_block = gh7_df.loc[mask, 'geohash7'].tolist()
        if gh7_block:               # 只保留非空窗口
            block_id += 1
            windows.append((block_id, lon0, lon1, lat0, lat1, gh7_block))
        lon0 += STEP_SIZE
    lat0 += STEP_SIZE

print(f'共生成 {len(windows)} 个 滑窗块')

# 3. 已落盘块号（断点续跑核心，保留原逻辑）
done_blocks = {int(f.stem.split('_')[-1])
               for f in Path(DST_DIR).glob('aggregated_by_geohash7_blocks_core_*.csv')}
print(f'已完成 {len(done_blocks)} 块，待处理 {len(windows)-len(done_blocks)} 块')

# 4. 滑窗分块处理（移除预过滤，直接分块读取原文件做过滤聚合，保留所有性能优化）
print('\n🚀 开始处理待处理滑窗块...')
for block_id, lon0, lon1, lat0, lat1, gh7_block in windows:
    if block_id in done_blocks:
        print(f'>>> 第 {block_id} 块已存在，跳过')
        continue

    print(f'>>> 处理第 {block_id}/{len(windows)} 个滑窗块（窗内 {len(gh7_block)} 个 grid）...')
    gh7_set = set(gh7_block)
    agg_list = []

    # 直接分块读取原SRC_CSV，过滤当前滑窗的gh7并聚合（保留numpy.in1d高性能过滤）
    for daas_chunk in pd.read_csv(SRC_CSV, chunksize=CHUNK_DAAS):
        # 高性能过滤：numpy.in1d替代pandas.isin
        mask = np.in1d(daas_chunk['final_grid_id'].values, list(gh7_set))
        if mask.any():
            # 过滤后聚合，保留轻量化dtype和预定义聚合规则
            agg_chunk = daas_chunk.loc[mask].groupby(GROUP_COLS, as_index=False).agg(AGG_RULES)
            agg_list.append(agg_chunk)

    # 拼接聚合结果，最终一次聚合去重（保留性能优化，减少重复计算）
    if agg_list:
        block_df = pd.concat(agg_list, ignore_index=True).groupby(GROUP_COLS, as_index=False).agg(AGG_RULES)
    else:
        block_df = pd.DataFrame(columns=GROUP_COLS + list(AGG_RULES.keys()))

    # 保存当前滑窗结果，Path路径更安全
    out_file = Path(DST_DIR) / f'aggregated_by_geohash7_blocks_core_{block_id}.csv'
    block_df.to_csv(out_file, index=False)
    print(f'    已保存 → {out_file.name}  行数：{len(block_df):,}')

print('\n🎉 所有滑窗块处理完成！')