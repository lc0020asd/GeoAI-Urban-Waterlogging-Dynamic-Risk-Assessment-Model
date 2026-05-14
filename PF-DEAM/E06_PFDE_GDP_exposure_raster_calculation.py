import numpy as np
import rasterio
import os
from tqdm import tqdm
from typing import Tuple, Dict, Any

# -------------------------- 1. 核心配置参数（无需修改，完全匹配研究要求） --------------------------
# AHP层次分析权重表（严格匹配表17，4个时间情境×3部门）
WEIGHT_MAP = {
    "工作日白天": {"农业": 0.06, "工业": 0.23, "服务业": 0.71},
    "工作日黑夜": {"农业": 0.05, "工业": 0.19, "服务业": 0.76},
    "休息日白天": {"农业": 0.08, "工业": 0.20, "服务业": 0.72},
    "休息日黑夜": {"农业": 0.06, "工业": 0.27, "服务业": 0.67}
}
# 时间情境列表（自动遍历计算）
TIME_SCENARIOS = ["工作日白天", "工作日黑夜", "休息日白天", "休息日黑夜"]
# 三部门GDP TIFF数据路径（匹配你的实际路径）
GDP_DATA_DIR = r'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\gdp'
GDP_RASTER_PATHS = {
    "农业": os.path.join(GDP_DATA_DIR, "AgricultureGDP_clip.tif"),
    "工业": os.path.join(GDP_DATA_DIR, "IndustryGDP_clip.tif"),
    "服务业": os.path.join(GDP_DATA_DIR, "ServiceGDP_clip.tif")
}
# 结果保存路径+命名模板（和原始GDP同目录，带情境标识）
OUTPUT_DIR = GDP_DATA_DIR
OUTPUT_TIF_PATTERN = "GDP设施暴露性_{}.tif"
# 归一化范围（GDP值归一化到[0,1]，公式中F_normalized(i)要求）
NORMALIZE_RANGE = (0, 1)
# 栅格无效值填充（代表无GDP/无暴露性的区域）
NODATA_VALUE = -9999


# -------------------------- 2. 栅格数据读取+无效值处理（修复meta类型注解报错） --------------------------
def read_gdp_raster(ras_path: str, sector: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    读取单部门GDP TIFF栅格，处理无效值（NaN/负数→0）
    返回：像素值数组、栅格元数据（投影/分辨率/范围，用于输出新栅格）
    修复：rasterio.meta → Dict[str, Any] 类型注解，解决AttributeError
    """
    try:
        with rasterio.open(ras_path) as src:
            # 读取单波段栅格数据
            raster_arr = src.read(1)
            # 处理无效值：NaN/负数设为0（无该部门GDP）
            # 先转换为float32，避免整数数组无法存储NaN
            raster_arr = raster_arr.astype(np.float32)
            raster_arr = np.where(np.isnan(raster_arr) | (raster_arr < 0), 0.0, raster_arr)
            # 深拷贝元数据，用于后续输出新栅格（兼容所有rasterio版本）
            meta = src.meta.copy()
            # 强制设置输出属性：单波段、float32、无效值
            meta.update(
                dtype=rasterio.float32,
                nodata=NODATA_VALUE,
                count=1  # 确保单波段
            )
        print(f"✅ 读取{sector}GDP栅格完成 | 尺寸：{raster_arr.shape} | 投影：{meta['crs']}")
        return raster_arr, meta
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 未找到{sector}GDP文件：{ras_path}，请检查路径！")
    except Exception as e:
        raise Exception(f"❌ 读取{sector}GDP失败：{str(e)}")


# -------------------------- 3. 逐像素最小-最大归一化（处理除零，适配栅格数组） --------------------------
def min_max_normalize_raster(raster_arr: np.ndarray, target_range: tuple = (0, 1)) -> np.ndarray:
    """
    对栅格数组做全局最小-最大归一化，映射到target_range
    公式：norm = (arr - min) / (max - min) * (max_t - min_t) + min_t
    处理max=min的情况（全区域0），直接返回min_t
    """
    arr_min = raster_arr.min()
    arr_max = raster_arr.max()
    if arr_max == arr_min:
        return np.full_like(raster_arr, target_range[0], dtype=np.float32)
    # 归一化计算（保留浮点精度，避免除零警告）
    norm_arr = (raster_arr - arr_min) / (arr_max - arr_min + 1e-8)
    norm_arr = norm_arr * (target_range[1] - target_range[0]) + target_range[0]
    return norm_arr.astype(np.float32)


# -------------------------- 4. 核心计算：三部门栅格加权融合（逐像素计算暴露性） --------------------------
def calculate_exposure_raster(agri_arr, indu_arr, serv_arr, meta) -> None:
    """
    1. 对三部门GDP栅格做全局归一化（得到F_normalized(i)）
    2. 按4个情境的权重逐像素计算暴露性 E = Σ(W_i * F_normalized(i))
    3. 直接输出各情境的TIFF栅格（和原始GDP同投影/分辨率/范围）
    """
    print("\n" + "-" * 50)
    print("🔢 开始三部门栅格归一化+加权计算")
    # 步骤1：三部门GDP全局归一化（公式中的F_normalized(i)，[0,1]）
    agri_norm = min_max_normalize_raster(agri_arr, NORMALIZE_RANGE)
    indu_norm = min_max_normalize_raster(indu_arr, NORMALIZE_RANGE)
    serv_norm = min_max_normalize_raster(serv_arr, NORMALIZE_RANGE)

    # 步骤2：遍历4个时间情境，逐像素计算暴露性并输出TIFF
    for scenario in tqdm(TIME_SCENARIOS, desc="4个情境暴露性计算+输出TIFF"):
        weights = WEIGHT_MAP[scenario]
        # 核心公式：E_facility = W农*F农 + W工*F工 + W服*F服（逐像素运算，float32精度）
        exposure_arr = (
                weights["农业"] * agri_norm +
                weights["工业"] * indu_norm +
                weights["服务业"] * serv_norm
        ).astype(np.float32)

        # 处理无GDP区域：三部门均为0的像素设为无效值（避免浮点判断误差，加1e-8）
        no_gdp_mask = (agri_arr < 1e-8) & (indu_arr < 1e-8) & (serv_arr < 1e-8)
        exposure_arr[no_gdp_mask] = NODATA_VALUE

        # 步骤3：输出当前情境的TIFF栅格
        output_tif = os.path.join(OUTPUT_DIR, OUTPUT_TIF_PATTERN.format(scenario))
        with rasterio.open(output_tif, 'w', **meta) as dst:
            dst.write(exposure_arr, 1)  # 写入单波段（count=1）

        # 输出当前情境统计信息（过滤无效值）
        valid_arr = exposure_arr[exposure_arr != NODATA_VALUE]
        if len(valid_arr) > 0:
            print(f"✅ {scenario} | 结果：{os.path.basename(output_tif)}")
            print(f"   有效像素暴露性范围：{valid_arr.min():.4f} ~ {valid_arr.max():.4f}")
        else:
            print(f"⚠️  {scenario} 无有效GDP像素，生成的栅格全为无效值！")


# -------------------------- 主函数：执行GDP暴露性栅格计算全流程 --------------------------
if __name__ == "__main__":
    try:
        print("=" * 70)
        print("📌 GDP设施暴露性计算（直接输出TIFF栅格，4个时间情境）")
        print("=" * 70)
        # 步骤1：批量读取三部门GDP栅格（统一用农业栅格的元数据输出，保证三栅格同范围/分辨率）
        agri_arr, meta = read_gdp_raster(GDP_RASTER_PATHS["农业"], "农业")
        indu_arr, _ = read_gdp_raster(GDP_RASTER_PATHS["工业"], "工业")
        serv_arr, _ = read_gdp_raster(GDP_RASTER_PATHS["服务业"], "服务业")

        # 校验三部门栅格尺寸一致（必须同范围/分辨率才能逐像素计算）
        assert agri_arr.shape == indu_arr.shape == serv_arr.shape, \
            "❌ 三部门GDP栅格尺寸不一致！请用QGIS/ArcGIS做【重采样+按掩膜裁剪】统一后再运行！"

        # 步骤2：核心计算+输出4个情境的TIFF栅格
        calculate_exposure_raster(agri_arr, indu_arr, serv_arr, meta)

        # 全流程完成提示
        print("\n" + "=" * 70)
        print("🎉 【全流程完成】4个时间情境GDP暴露性TIFF栅格生成完成！")
        print(f"📁 所有栅格保存至：{OUTPUT_DIR}")
        print(f"📌 栅格说明：值越大=GDP设施暴露性越高 | 无效值={NODATA_VALUE}（无经济活动区域）")
        print(f"📌 栅格属性：和原始GDP同投影/分辨率/范围 | 单波段 | float32精度")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ 计算失败：{str(e)}")
        raise