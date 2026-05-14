
# 径流积水转化改进20251130 - 支持流速与危险性指数输出版
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from rasterio.features import geometry_mask
from scipy import ndimage
from shapely.geometry import Point
import os
from scipy.ndimage import sobel
from collections import deque
import matplotlib
import json
import csv

matplotlib.rc("font", family='SimHei')


class SCSCNFloodSimulator:
    def __init__(self, dem_path, shp_path, landcover_path, out_path,
                 soil_type_path=None, zone_id_field=None, use_physical_route=True):
        """初始化模拟器"""
        self.dem_path = dem_path
        self.shp_path = shp_path
        self.landcover_path = landcover_path
        self.soil_type_path = soil_type_path
        self.out_path = out_path
        self.zone_id_field = zone_id_field
        self.use_physical_route = use_physical_route

        # 数据存储
        self.dem_data = None
        self.dem_transform = None
        self.dem_crs = None
        self.dem_profile = None
        self.landcover_data = None
        self.soil_data = None
        self.zones_gdf = None

        # 分区数据
        self.zone_dems = {}
        self.zone_landcovers = {}
        self.zone_soils = {}
        self.zone_results = {}

        # D8汇流存储
        self.flow_delay_queues = {}

        # 模型参数
        self._setup_parameters()

    def _setup_parameters(self):
        """设置模型参数"""
        # 降雨等级定义
        self.rainfall_levels = {
            '小雨': (0.1, 19.9), '中雨': (20.0, 49.9), '大雨': (50.0, 99.9),
            '暴雨': (100.0, 199.9), '大暴雨': (200.0, 499.9), '特大暴雨': (500.0, 1000.0)
        }

        # 芝加哥雨型参数
        self.chicago_r = 0.375
        self.chicago_peak_factor = 2.5

        # 水文参数
        self.base_drainage_rate = 2.0  # mm/h
        self.urban_drainage_rate = 15.0  # mm/h
        self.evaporation_rate = 0.05  # mm/h
        self.ia_ratio = 0.2  # 初损率
        self.manning_n = 0.05  # 曼宁系数

        # Williams坡度修正参数
        self.williams_use_correction = True
        self.williams_ref_slope = 0.05
        self.williams_k1 = 2.533
        self.williams_k2 = 0.0636
        self.williams_k3 = 13.86

        # 地表覆盖分类
        self.landcover_types = {
            1: '农田', 2: '森林', 3: '灌木', 4: '草地',
            5: '水体', 6: '冰雪', 7: '荒地', 8: '不透水面', 9: '湿地'
        }

        # 地表覆盖对应的曼宁系数
        self.landcover_manning_n = {
            1: 0.3228,  # 农田
            2: 0.2,  # 森林
            3: 0.4,  # 灌木
            4: 0.15,  # 草地
            5: 0.02,  # 水体
            6: 0.025,  # 冰雪
            7: 0.013,  # 荒地
            8: 0.013,  # 不透水面
            9: 0.1825  # 湿地
        }

        # 土壤类型到土壤组的映射
        self.soil_value_to_group = {
            12: 0, 9: 1, 11: 1, 4: 2, 5: 2, 6: 2,
            7: 2, 8: 2, 10: 2, 1: 3, 2: 3, 3: 3
        }

        # 地表覆盖的CN值（按土壤组A,B,C,D排序）
        self.landcover_cn_values = {
            1: [67, 78, 85, 89],  # 农田
            2: [36, 60, 73, 79],  # 森林
            3: [35, 56, 70, 77],  # 灌木
            4: [49, 69, 74, 84],  # 草地
            5: [100, 100, 100, 100],  # 水体
            6: [50, 50, 50, 50],  # 冰雪
            7: [50, 50, 50, 50],  # 荒地
            8: [89, 92, 94, 95],  # 不透水面
            9: [45, 66, 77, 83]  # 湿地
        }

    def load_data(self):
        """加载所有输入数据"""
        print("=" * 50)
        print("开始加载数据...")

        try:
            # 加载DEM
            with rasterio.open(self.dem_path) as src:
                self.dem_data = src.read(1)
                self.dem_transform = src.transform
                self.dem_crs = src.crs
                self.dem_profile = src.profile
            print(f"✓ DEM加载成功: {self.dem_data.shape}")

            # 加载分区矢量
            self.zones_gdf = gpd.read_file(self.shp_path)
            if self.zones_gdf.crs != self.dem_crs:
                self.zones_gdf = self.zones_gdf.to_crs(self.dem_crs)
            if self.zone_id_field is None or self.zone_id_field not in self.zones_gdf.columns:
                self.zones_gdf['zone_id'] = range(len(self.zones_gdf))
                self.zone_id_field = 'zone_id'
            print(f"✓ 分区数据加载成功: {len(self.zones_gdf)} 个分区")

            # 加载地表覆盖
            with rasterio.open(self.landcover_path) as src:
                self.landcover_data = src.read(1)
            print(f"✓ 地表覆盖加载成功: {self.landcover_data.shape}")

            # 加载土壤数据（可选）
            if self.soil_type_path and os.path.exists(self.soil_type_path):
                with rasterio.open(self.soil_type_path) as src:
                    self.soil_data = src.read(1)
                print(f"✓ 土壤数据加载成功: {self.soil_data.shape}")
            else:
                print("⚠ 未加载土壤数据，将使用启发式方法")

            print("=" * 50)

        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def clip_dem_by_zones(self):
        """按分区裁剪数据"""
        print("\n开始按分区裁剪数据...")

        with rasterio.open(self.dem_path) as dem_src, \
                rasterio.open(self.landcover_path) as lc_src:

            soil_src = None
            if self.soil_data is not None:
                soil_src = rasterio.open(self.soil_type_path)

            try:
                for idx, zone in self.zones_gdf.iterrows():
                    zone_id = zone[self.zone_id_field]
                    geometry = [zone.geometry]

                    try:
                        # 裁剪DEM
                        clipped_dem_data, dem_tf = mask(dem_src, geometry, crop=True, nodata=dem_src.nodata)
                        clipped_dem = clipped_dem_data[0]

                        # 裁剪地表覆盖
                        clipped_lc_data, lc_tf = mask(lc_src, geometry, crop=True, nodata=lc_src.nodata)
                        clipped_lc = clipped_lc_data[0]

                        # 裁剪土壤数据（如果存在）
                        clipped_soil = None
                        if soil_src:
                            clipped_soil_data, soil_tf = mask(soil_src, geometry, crop=True, nodata=soil_src.nodata)
                            clipped_soil = clipped_soil_data[0]

                        # 确保数据形状一致
                        if clipped_dem.shape != clipped_lc.shape:
                            clipped_lc = self._resample_to_target(clipped_lc, clipped_dem.shape)
                        if clipped_soil is not None and clipped_dem.shape != clipped_soil.shape:
                            clipped_soil = self._resample_to_target(clipped_soil, clipped_dem.shape)

                        # 创建有效掩膜
                        if dem_src.nodata is not None:
                            valid_mask = clipped_dem != dem_src.nodata
                        else:
                            valid_mask = np.ones_like(clipped_dem, dtype=bool)

                        # 存储分区数据
                        self.zone_dems[zone_id] = {
                            'dem': clipped_dem,
                            'transform': dem_tf,
                            'valid_mask': valid_mask,
                            'geometry': zone.geometry
                        }
                        self.zone_landcovers[zone_id] = {'landcover': clipped_lc}

                        if clipped_soil is not None:
                            soil_group = self._map_soil_to_group(clipped_soil, valid_mask)
                            self.zone_soils[zone_id] = {'soil_group': soil_group}

                        print(f"✓ 分区 {zone_id} 裁剪完成: {clipped_dem.shape}")

                    except Exception as e:
                        print(f"✗ 分区 {zone_id} 裁剪失败: {e}")
                        continue

            finally:
                if soil_src:
                    soil_src.close()

        print(f"✓ 共 {len(self.zone_dems)} 个分区裁剪成功\n")

    def _resample_to_target(self, array, target_shape):
        """重采样数组到目标形状"""
        from scipy.ndimage import zoom

        scale_y = target_shape[0] / array.shape[0]
        scale_x = target_shape[1] / array.shape[1]
        resampled = zoom(array, (scale_y, scale_x), order=0, mode='nearest')

        if resampled.shape != target_shape:
            result = np.full(target_shape, 0, dtype=resampled.dtype)
            min_rows = min(resampled.shape[0], target_shape[0])
            min_cols = min(resampled.shape[1], target_shape[1])
            result[:min_rows, :min_cols] = resampled[:min_rows, :min_cols]
            return result

        return resampled

    def _map_soil_to_group(self, soil_array, valid_mask):
        """将土壤类型映射到土壤组"""
        soil_group = np.zeros_like(soil_array, dtype=np.int16)  # 改为int16避免溢出

        for soil_val, group_idx in self.soil_value_to_group.items():
            mask = (soil_array == soil_val) & valid_mask
            soil_group[mask] = group_idx

        # 处理未映射的土壤类型
        unmapped = ~np.isin(soil_array, list(self.soil_value_to_group.keys())) & valid_mask
        soil_group[unmapped] = 2  # 默认设为C组

        return soil_group

    def generate_chicago_rainfall(self, total_mm, duration_hours):
        """生成芝加哥雨型降雨过程"""
        time_points = np.arange(0, duration_hours, 1.0)
        n_steps = len(time_points)
        rainfall_dist = np.zeros(n_steps)

        # 计算峰值时间
        peak_time = duration_hours * self.chicago_r
        peak_idx = int(peak_time)

        # 生成雨型分布
        for i in range(n_steps):
            t = i + 0.5
            if t <= peak_time:
                factor = np.power(t / peak_time, 1.5) if peak_time > 0 else 0
            else:
                remaining = duration_hours - t
                total_remaining = duration_hours - peak_time
                factor = np.power(remaining / total_remaining, 0.8) if total_remaining > 0 else 0
            rainfall_dist[i] = factor

        # 增强峰值
        if peak_idx < n_steps:
            rainfall_dist[peak_idx] *= self.chicago_peak_factor

        # 标准化到总降雨量
        total_factor = np.sum(rainfall_dist)
        if total_factor > 0:
            rainfall_dist = rainfall_dist * total_mm / total_factor

        return {
            'time_hours': time_points,
            'rainfall_per_hour': rainfall_dist,
            'peak_time': peak_time,
            'peak_intensity': np.max(rainfall_dist)
        }

    def calculate_cn_parameters(self, zone_id, amc='II'):
        """计算CN相关参数"""
        zone_data = self.zone_dems[zone_id]
        lc_data = self.zone_landcovers[zone_id]

        dem = zone_data['dem']
        valid_mask = zone_data['valid_mask']
        landcover = lc_data['landcover']

        # 计算坡度
        pixel_area = self._pixel_area_m2(zone_data['transform'], dem.shape)
        res_m = np.sqrt(pixel_area)
        gy, gx = np.gradient(dem, res_m)
        slope = np.sqrt(gx ** 2 + gy ** 2)
        slope[~valid_mask] = 0

        # 分配土壤组
        soil_group = self.assign_soil_groups(zone_id, landcover, valid_mask)

        # 计算基础CN值
        cn_base = self.calculate_cn_values(landcover, soil_group, valid_mask)

        # Williams坡度修正
        if self.williams_use_correction:
            cn_slope_corrected = self.williams_slope_correction(cn_base, slope, valid_mask)
        else:
            cn_slope_corrected = cn_base

        # AMC修正
        cn_adjusted = self.adjust_cn_for_amc(cn_slope_corrected, amc)

        return {
            'cn_base': cn_base,
            'cn_slope_corrected': cn_slope_corrected,
            'cn_adjusted': cn_adjusted,
            'slope': slope,
            'soil_group': soil_group,
            'valid_mask': valid_mask
        }

    def assign_soil_groups(self, zone_id, landcover, valid_mask):
        """分配土壤组（如果缺少土壤数据）"""
        if zone_id in self.zone_soils:
            return self.zone_soils[zone_id]['soil_group']

        dem = self.zone_dems[zone_id]['dem']

        # 基于地形和地表覆盖的启发式土壤组分配
        gy, gx = np.gradient(dem)
        slope = np.sqrt(gx ** 2 + gy ** 2)

        soil_group = np.ones_like(landcover, dtype=np.int16)  # 改为int16避免溢出

        # 水体和不透水面为D组
        soil_group[np.isin(landcover, [5, 9]) & valid_mask] = 3
        # 陡坡森林为A组
        soil_group[np.isin(landcover, [2, 3]) & (slope > 0.03) & valid_mask] = 0
        # 不透水面为D组
        soil_group[(landcover == 8) & valid_mask] = 3

        return soil_group

    def calculate_cn_values(self, landcover, soil_group, valid_mask):
        """计算CN值"""
        cn = np.zeros_like(landcover, dtype=np.float32)

        for lc_type, cn_vals in self.landcover_cn_values.items():
            for soil_idx in range(4):
                mask = (landcover == lc_type) & (soil_group == soil_idx) & valid_mask
                if np.any(mask):
                    cn[mask] = cn_vals[soil_idx]

        # 处理未知地表覆盖类型
        unknown = ~np.isin(landcover, list(self.landcover_cn_values.keys())) & valid_mask
        default_cn = [39, 61, 74, 80]  # 默认CN值（A,B,C,D组）
        for soil_idx in range(4):
            cn[unknown & (soil_group == soil_idx)] = default_cn[soil_idx]

        return np.clip(cn, 30, 100)

    def williams_slope_correction(self, cn2, slope, valid_mask):
        """Williams坡度修正"""
        cn2s = cn2.copy()
        cn3 = self.adjust_cn_for_amc(cn2, 'III')

        # 仅对高坡度区域进行修正
        high_slope_mask = (slope > self.williams_ref_slope) & valid_mask

        if np.any(high_slope_mask):
            alpha = (1.0 / 3.0) * (1.0 - 2.0 * np.exp(-self.williams_k3 * slope[high_slope_mask]))
            cn2s[high_slope_mask] = (cn2[high_slope_mask] +
                                     (cn3[high_slope_mask] - cn2[high_slope_mask]) * alpha)

        cn2s = np.clip(cn2s, 30, 100)

        # 输出修正统计
        if np.any(high_slope_mask):
            original_mean = np.mean(cn2[high_slope_mask])
            corrected_mean = np.mean(cn2s[high_slope_mask])
            print(f"  Williams坡度修正: 原始CN={original_mean:.1f}, 修正后CN={corrected_mean:.1f}")

        return cn2s

    def adjust_cn_for_amc(self, cn, amc='II'):
        """前期土壤湿度条件(AMC)修正"""
        if amc == 'I':  # 干旱条件
            return cn / (2.334 - 0.01334 * cn)
        elif amc == 'III':  # 湿润条件
            return cn / (0.427 + 0.00573 * cn)
        return cn  # 正常条件

    def calculate_scs_runoff(self, rainfall, cn):
        """SCS-CN径流计算"""
        if np.isscalar(rainfall):
            rainfall = np.full_like(cn, rainfall, dtype=np.float32)

        # 计算潜在最大滞留量
        s = 25400.0 / np.maximum(cn, 1) - 254.0
        s = np.maximum(s, 0)

        # 计算初损
        ia = self.ia_ratio * s

        # 计算径流
        runoff = np.zeros_like(rainfall, dtype=np.float32)
        mask = rainfall > ia
        if np.any(mask):
            p_minus_ia = rainfall[mask] - ia[mask]
            runoff[mask] = (p_minus_ia ** 2) / (p_minus_ia + s[mask])

        return runoff

    def preprocess_dem(self, dem):
        """DEM预处理：洼地填充"""
        filled = dem.copy()
        valid = ~np.isnan(dem) & (dem != -9999)

        # 迭代填充洼地
        for _ in range(20):
            old = filled.copy()
            filled = ndimage.grey_dilation(filled, size=3)
            filled = np.minimum(filled, dem + 0.5)
            filled[~valid] = dem[~valid]
            if np.allclose(filled[valid], old[valid], rtol=1e-6):
                break

        return filled

    def d8_flow_dir(self, dem, pixel_area):
        """计算D8流向"""
        res = np.sqrt(pixel_area)

        # 计算梯度
        gy = -sobel(dem, axis=0, mode='constant') / (8 * res)
        gx = -sobel(dem, axis=1, mode='constant') / (8 * res)

        # 计算坡度和坡向
        slope = np.sqrt(gx ** 2 + gy ** 2)
        aspect = np.arctan2(gy, gx)

        # 转换为D8编码（1:E, 2:NE, 4:N, 8:NW, 16:W, 32:SW, 64:S, 128:SE）
        direction = np.mod(450 - np.degrees(aspect), 360)
        code = np.round(direction / 45) % 8
        d8 = 2 ** code.astype(np.uint8)  # 这里保持uint8，因为D8编码最大128

        # 平坦区域无流向
        d8[slope < 1e-6] = 0

        return d8, slope

    def _calculate_travel_time(self, slope, pixel_area, landcover_type=None, base_depth=0.01):
        """计算水流传播时间（小时）和流速（m/s）"""
        # 水力半径
        hydraulic_rad = base_depth

        # 根据地表覆盖类型获取曼宁系数
        if landcover_type is not None and landcover_type in self.landcover_manning_n:
            manning_n = self.landcover_manning_n[landcover_type]
        else:
            manning_n = self.manning_n  # 默认值

        # 曼宁公式计算流速
        velocity = (1 / manning_n) * hydraulic_rad ** (2 / 3) * np.sqrt(np.maximum(slope, 1e-4))
        velocity = np.clip(velocity, 1e-4, 10.0)  # 限制流速范围（单位：m/s）

        # 像元边长
        res = np.sqrt(pixel_area)

        # 传播时间（小时）
        travel_time_h = res / (velocity * 3600)
        travel_time_h = np.clip(travel_time_h, 0.001, 10)

        return travel_time_h, velocity

    def route_runoff_to_flood(self, runoff_mm, flood_mm_prev, dem, filled_dem,
                              valid_mask, landcover, pixel_area, dt=3600, rainfall_level='暴雨'):
        """
        改进版D8物理汇流：考虑积水动态影响
        返回：积水深度(mm)和流速(m/s)
        """
        h, w = dem.shape
        dt_h = dt / 3600  # 转换为小时

        # 1. 计算当前实际地形（考虑积水深度）
        # 将积水深度转换为米，加到DEM上
        current_flood_m = flood_mm_prev * 1e-3
        dynamic_elevation = dem.copy().astype(np.float32)  # 确保使用浮点型

        # 仅在有效区域和有积水的区域更新高程
        water_mask = (valid_mask) & (current_flood_m > 0.001)  # 1mm阈值
        dynamic_elevation[water_mask] = dem[water_mask] + current_flood_m[water_mask]

        # 2. 对动态高程进行洼地填充
        dynamic_filled = self.preprocess_dem(dynamic_elevation)

        # 3. 基于动态高程计算流向
        d8, slope = self.d8_flow_dir(dynamic_filled, pixel_area)

        # 4. 单位转换
        runoff_volume = (runoff_mm * 1e-3) * pixel_area
        runoff_rate = runoff_volume / 3600.0

        # 5. 计算当前积水体积
        depth_prev = flood_mm_prev * 1e-3
        vol_prev = depth_prev * pixel_area

        # 6. D8延迟汇流
        zone_key = id(dem)
        if zone_key not in self.flow_delay_queues:
            self.flow_delay_queues[zone_key] = [[deque() for _ in range(w)] for _ in range(h)]

        inflow_vol, updated_queues = self._route_with_delay(
            d8, slope, runoff_rate, self.flow_delay_queues[zone_key], dt_h, pixel_area, landcover
        )
        self.flow_delay_queues[zone_key] = updated_queues

        # 7. 考虑积水表面的坡度计算（水往低处流）
        # 基于动态高程计算实际的水力坡度
        water_surface = dynamic_elevation.copy()

        # 计算水面高程的坡度（更加物理真实）
        res = np.sqrt(pixel_area)
        gy, gx = np.gradient(water_surface, res)
        water_slope = np.sqrt(gx ** 2 + gy ** 2)
        water_slope[~valid_mask] = 0

        # 8. 洼地调蓄计算（考虑已占用的调蓄容量）
        depression_depth = filled_dem - dem
        occupied_depth = np.minimum(current_flood_m, depression_depth)
        remaining_depression = np.maximum(0, depression_depth - occupied_depth)

        # 根据降雨等级调整调蓄参数
        storage_params = self._get_storage_params(rainfall_level)
        storage_multiplier = storage_params['multiplier']
        storage_max_limit = storage_params['max_limit']

        max_storage_mm = storage_multiplier * remaining_depression * 1000
        max_storage_mm = np.clip(max_storage_mm, 0, storage_max_limit)

        # 确保最小调蓄深度
        min_storage = 50
        max_storage_mm = np.maximum(max_storage_mm, min_storage)
        max_storage_vol = max_storage_mm * 1e-3 * pixel_area

        # 9. 水量平衡计算
        vol_current = vol_prev + inflow_vol

        # 调蓄体积（考虑已占用部分）
        storage_vol = np.minimum(vol_current, max_storage_vol)

        # 超额体积（参与出流）
        excess_vol = np.maximum(0, vol_current - max_storage_vol)

        # 10. 出流计算（基于水面坡度）
        water_depth = excess_vol / pixel_area
        # water_depth = np.clip(water_depth, 1e-6, 10)

        # 使用水面坡度而不是地形坡度
        hydraulic_rad = water_depth

        # 创建曼宁系数数组（基于地表覆盖类型）
        manning_n_array = np.full_like(landcover, self.manning_n, dtype=np.float32)
        for lc_type, n_value in self.landcover_manning_n.items():
            manning_n_array[landcover == lc_type] = n_value

        # 曼宁公式计算流速
        velocity = (1 / manning_n_array) * hydraulic_rad ** (2 / 3) * np.sqrt(np.maximum(water_slope, 1e-4))
        velocity = np.clip(velocity, 1e-4, 10.0)

        outflow_volume = velocity * hydraulic_rad * res * float(dt)

        # 11. 排水和蒸发损失
        drainage_rate, evap_rate = self._calculate_loss_rates(landcover, pixel_area)
        drainage_volume = drainage_rate * float(dt)
        evap_volume = evap_rate * float(dt)

        # 12. 总损失
        total_loss = outflow_volume + drainage_volume + evap_volume

        # 13. 更新积水体积
        vol_new = storage_vol + np.maximum(0, excess_vol - total_loss)
        vol_new[~valid_mask] = 0

        # 转换回毫米深度
        flood_mm_new = vol_new / pixel_area * 1000
        flood_mm_new[~valid_mask] = 0

        # 14. 计算流速分布
        # 无水区域流速为0
        # velocity[flood_mm_new < 0.1] = 0
        velocity[~valid_mask] = 0

        return flood_mm_new, velocity

    def _route_with_delay(self, d8, slope, runoff_rate, queues, dt, pixel_area, landcover=None):
        """改进版延迟汇流：考虑坡度对传播时间的影响"""
        h, w = d8.shape

        # 1. 将当前径流加入延迟队列
        for i, j in np.ndindex(h, w):
            if runoff_rate[i, j] > 0:
                # 获取当前像元的地表覆盖类型
                lc_type = int(landcover[i, j]) if landcover is not None and 0 <= i < h and 0 <= j < w else None

                # 基于当前像元坡度和地表覆盖类型计算传播时间
                travel_time, _ = self._calculate_travel_time(slope[i, j], pixel_area, lc_type)

                # 动态调整传播时间（考虑水流深度）
                if travel_time >= 0.001:
                    queues[i][j].append({
                        'volume': float(runoff_rate[i, j]) * float(dt) * 3600.0,
                        'arrival_h': float(travel_time),
                        'origin': (i, j)  # 记录来源，便于调试
                    })

        # 2. 处理队列，计算到达水量
        inflow = np.zeros((h, w), dtype=np.float32)

        for i, j in np.ndindex(h, w):
            current_queue = queues[i][j]
            remaining_queue = deque()

            while current_queue:
                item = current_queue.popleft()
                item['arrival_h'] -= float(dt)

                if item['arrival_h'] <= 0:
                    # 根据D8流向将水量传递到下游
                    di, dj = self._get_d8_offset(d8[i, j])
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        inflow[ni, nj] += item['volume']
                    else:
                        # 如果流向边界，则视为出流损失
                        pass
                else:
                    remaining_queue.append(item)

            queues[i][j] = remaining_queue

        return inflow, queues

    def _get_storage_params(self, rainfall_level):
        """获取调蓄参数"""
        params = {
            '特大暴雨': {'multiplier': 2.0, 'max_limit': 2000},
            '大暴雨': {'multiplier': 1.5, 'max_limit': 1000},
            '暴雨': {'multiplier': 1.2, 'max_limit': 800},
            '大雨': {'multiplier': 1.0, 'max_limit': 600},
            '中雨': {'multiplier': 0.8, 'max_limit': 400},
            '小雨': {'multiplier': 0.6, 'max_limit': 300}
        }
        return params.get(rainfall_level, {'multiplier': 1.0, 'max_limit': 500})

    def _calculate_loss_rates(self, landcover, pixel_area):
        """计算排水和蒸发损失率"""
        base_drainage_m3s = float(self.base_drainage_rate) * 1e-3 * float(pixel_area) / 3600.0
        urban_drainage_m3s = float(self.urban_drainage_rate) * 1e-3 * float(pixel_area) / 3600.0

        drainage_rate = np.full(landcover.shape, base_drainage_m3s, dtype=np.float32)
        drainage_rate[landcover == 8] = urban_drainage_m3s

        evap_rate = np.full(landcover.shape,
                            float(self.evaporation_rate) * 1e-3 * float(pixel_area) / 3600.0,
                            dtype=np.float32)

        return drainage_rate, evap_rate

    def _get_d8_offset(self, d8_code):
        """获取D8编码对应的行列偏移"""
        d8_offsets = {
            1: (0, 1),  # 东
            2: (-1, 1),  # 东北
            4: (-1, 0),  # 北
            8: (-1, -1),  # 西北
            16: (0, -1),  # 西
            32: (1, -1),  # 西南
            64: (1, 0),  # 南
            128: (1, 1)  # 东南
        }
        return d8_offsets.get(d8_code, (0, 0))

    def calculate_hazard_index(self, flood_mm, velocity):
        """
        计算城市内涝危险性指数 H
        公式: H = d * v + 0.5 + DF
        d: 积水深度(m), v: 流速(m/s)
        DF: 水深危害参数, d<=0.15m时DF=0.5, d>0.15m时DF=1.0
        """
        # 积水深度单位转换: mm -> m
        d = flood_mm * 1e-3

        # 确保流速为浮点型数组
        v = np.array(velocity, dtype=np.float32)

        # 计算水深危害参数 DF
        df = np.where(d <= 0.15, 0.5, 1.0).astype(np.float32)

        # 计算危险性指数
        h = d * (v + 0.5) + df

        # 限制有效范围，避免异常值
        h = np.clip(h, 0.0, 50.0)

        return h.astype(np.float32)

    def simulate_zone_scs_cn(self, zone_id, rainfall_level='暴雨', duration_hours=6, amc='II'):
        """分区SCS-CN模拟"""
        # 清空汇流队列，避免历史模拟残留影响当前模拟
        self.flow_delay_queues = {}

        print(f"\n{'=' * 50}")
        print(f"模拟分区 {zone_id} - {rainfall_level} (AMC {amc})")
        print(f"{'=' * 50}")

        # 检查分区是否存在
        if zone_id not in self.zone_dems:
            raise ValueError(f"分区 {zone_id} 不存在")

        # 获取分区数据
        zone_data = self.zone_dems[zone_id]
        lc_data = self.zone_landcovers[zone_id]

        dem = zone_data['dem']
        valid_mask = zone_data['valid_mask']
        landcover = lc_data['landcover']

        # 验证降雨等级
        if rainfall_level not in self.rainfall_levels:
            raise ValueError(f"未知的降雨等级: {rainfall_level}")

        # 获取降雨总量
        _, max_rain = self.rainfall_levels[rainfall_level]
        total_rainfall = max_rain

        # 生成芝加哥雨型
        chicago_rain = self.generate_chicago_rainfall(total_rainfall, duration_hours)
        print(f"总降雨: {total_rainfall}mm, 峰值时间: {chicago_rain['peak_time']:.1f}h")

        # DEM预处理
        filled_dem = self.preprocess_dem(dem)
        pixel_area = self._pixel_area_m2(zone_data['transform'], dem.shape)

        # 计算CN参数
        cn_params = self.calculate_cn_parameters(zone_id, amc)
        cn_adjusted = cn_params['cn_adjusted']

        print(f"CN值统计: 平均={np.mean(cn_adjusted[valid_mask]):.1f}, "
              f"最小={np.min(cn_adjusted[valid_mask]):.1f}, "
              f"最大={np.max(cn_adjusted[valid_mask]):.1f}")

        # 初始化积水深度、流速和危险性指数
        current_flood = np.zeros_like(dem, dtype=np.float32)
        current_velocity = np.zeros_like(dem, dtype=np.float32)
        hourly_floods = []
        hourly_velocities = []
        hourly_hazards = []

        print("\n逐小时模拟:")
        # 逐小时模拟
        for hour, hourly_rain in enumerate(chicago_rain['rainfall_per_hour']):
            # 计算径流
            runoff = self.calculate_scs_runoff(hourly_rain, cn_adjusted)
            runoff[~valid_mask] = 0

            # 汇流计算 - 返回积水和流速
            current_flood, current_velocity = self.route_runoff_to_flood(
                runoff, current_flood, dem, filled_dem, valid_mask, landcover, pixel_area, 3600, rainfall_level
            )

            hourly_floods.append(current_flood.copy())
            hourly_velocities.append(current_velocity.copy())

            # 计算危险性指数
            hazard = self.calculate_hazard_index(current_flood, current_velocity)
            hourly_hazards.append(hazard.copy())

            # 输出本小时结果
            max_depth = np.max(current_flood[valid_mask]) if np.any(valid_mask) else 0
            avg_depth = np.mean(current_flood[valid_mask]) if np.any(valid_mask) else 0
            max_velocity = np.max(current_velocity[valid_mask]) if np.any(valid_mask) else 0
            avg_velocity = np.mean(current_velocity[valid_mask]) if np.any(valid_mask) else 0
            flooded_cells = np.sum(current_flood[valid_mask] > 1)
            flow_cells = np.sum(current_velocity[valid_mask] > 0.01)

            print(f"  第{hour + 1:2d}h: 降雨{hourly_rain:5.1f}mm, "
                  f"平均积水{avg_depth:5.1f}mm, "
                  f"最大积水{max_depth:5.1f}mm, "
                  f"平均流速{avg_velocity:5.2f}m/s, "
                  f"积水像元{flooded_cells:5d}个")

        # 整理结果
        result = {
            'zone_id': zone_id,
            'rainfall_level': rainfall_level,
            'amc': amc,
            'chicago_pattern': chicago_rain,
            'hourly_floods': hourly_floods,
            'hourly_velocities': hourly_velocities,
            'hourly_hazards': hourly_hazards,
            'final_flood': current_flood,
            'final_velocity': current_velocity,
            'final_hazard': hazard,
            'cn_array': cn_adjusted,
            'valid_mask': valid_mask,
            'max_flood_depth': np.max(current_flood[valid_mask]) if np.any(valid_mask) else 0,
            'max_velocity': np.max(current_velocity[valid_mask]) if np.any(valid_mask) else 0,
            'max_hazard': np.max(hazard[valid_mask]) if np.any(valid_mask) else 0,
            'total_flood_volume': np.sum(current_flood[valid_mask]) * pixel_area * 1e-3 if np.any(valid_mask) else 0,
            'williams_correction_applied': self.williams_use_correction
        }

        print(f"\n✓ 模拟完成! 最大积水深度: {result['max_flood_depth']:.1f}mm, "
              f"最大流速: {result['max_velocity']:.2f}m/s, "
              f"最大危险性指数: {result['max_hazard']:.2f}")
        return result

    def batch_simulate_all_zones(self, duration_hours=6, amc='II'):
        """批量模拟所有分区"""
        print("\n" + "=" * 50)
        print("批量模拟所有分区")
        print("=" * 50)

        all_results = {}

        for zone_id in self.zone_dems.keys():
            zone_results = {}

            for level in self.rainfall_levels.keys():
                try:
                    result = self.simulate_zone_scs_cn(zone_id, level, duration_hours, amc)
                    zone_results[level] = result
                except Exception as e:
                    print(f"✗ 分区{zone_id}-{level}模拟失败: {e}")
                    import traceback
                    traceback.print_exc()  # 添加详细错误信息
                    continue

            all_results[zone_id] = zone_results

        self.zone_results = all_results
        print(f"\n✓ 批量模拟完成! 共处理{len(all_results)}个分区")
        return all_results

    def save_results(self, save_dir=None):
        """保存模拟结果"""
        if save_dir is None:
            save_dir = os.path.join(self.out_path, 'simulation_results')

        os.makedirs(save_dir, exist_ok=True)

        print(f"\n保存模拟结果到: {save_dir}")

        # 保存逐小时结果
        hourly_dir = self._save_hourly_results(save_dir)
        # 保存逐小时拼接结果
        hourly_mosaic_dir = self._save_hourly_mosaic_results(save_dir)

        print(f"✓ 结果保存完成!")
        return {
            'hourly_results': hourly_dir,
            'hourly_mosaic_results': hourly_mosaic_dir
        }

    def save_zone_results(self, zone_id, zone_results, save_dir=None):
        """保存单个区域的模拟结果"""
        if save_dir is None:
            save_dir = os.path.join(self.out_path, 'simulation_results')
        os.makedirs(save_dir, exist_ok=True)

        # 保存当前区域的逐小时结果
        self._save_single_zone_hourly_results(save_dir, zone_id, zone_results)

        print(f"✓ 区域{zone_id}结果保存完成!")
        return save_dir

    def _save_single_zone_hourly_results(self, base_dir, zone_id, zone_results):
        """保存单个区域的逐小时积水结果和流速结果"""
        hourly_dir = os.path.join(base_dir, 'hourly')
        os.makedirs(hourly_dir, exist_ok=True)

        total_files = 0

        # 只处理指定区域的结果
        for rainfall_level, result in zone_results.items():
            zone_level_dir = os.path.join(hourly_dir, f'zone_{zone_id}', rainfall_level)
            os.makedirs(zone_level_dir, exist_ok=True)

            zone_data = self.zone_dems[zone_id]
            profile = self._create_output_profile(zone_data)

            # 保存积水深度数据
            for hour, flood_data in enumerate(result['hourly_floods']):
                filename = f"flood_{zone_id}_{rainfall_level}_hour_{hour + 1:02d}.tif"
                filepath = os.path.join(zone_level_dir, filename)

                output = flood_data.copy()
                output[~result['valid_mask']] = -9999.0

                with rasterio.open(filepath, 'w', **profile) as dst:
                    dst.write(output, 1)

                total_files += 1

            # 保存流速数据
            for hour, velocity_data in enumerate(result['hourly_velocities']):
                filename = f"velocity_{zone_id}_{rainfall_level}_hour_{hour + 1:02d}.tif"
                filepath = os.path.join(zone_level_dir, filename)

                output = velocity_data.copy()
                output[~result['valid_mask']] = -9999.0

                with rasterio.open(filepath, 'w', **profile) as dst:
                    dst.write(output, 1)

                total_files += 1

            # 保存危险性指数数据
            if 'hourly_hazards' in result:
                for hour, hazard_data in enumerate(result['hourly_hazards']):
                    filename = f"hazard_{zone_id}_{rainfall_level}_hour_{hour + 1:02d}.tif"
                    filepath = os.path.join(zone_level_dir, filename)

                    output = hazard_data.copy()
                    output[~result['valid_mask']] = -9999.0

                    with rasterio.open(filepath, 'w', **profile) as dst:
                        dst.write(output, 1)

                    total_files += 1

        print(f"  区域{zone_id}逐小时结果: {total_files} 个文件")
        return hourly_dir

    def _save_hourly_results(self, base_dir):
        """保存逐小时积水结果和流速结果"""
        hourly_dir = os.path.join(base_dir, 'hourly')
        os.makedirs(hourly_dir, exist_ok=True)

        total_files = 0

        for zone_id, zone_results in self.zone_results.items():
            for rainfall_level, result in zone_results.items():
                zone_level_dir = os.path.join(hourly_dir, f'zone_{zone_id}', rainfall_level)
                os.makedirs(zone_level_dir, exist_ok=True)

                zone_data = self.zone_dems[zone_id]
                profile = self._create_output_profile(zone_data)

                # 保存积水深度数据
                for hour, flood_data in enumerate(result['hourly_floods']):
                    filename = f"flood_{zone_id}_{rainfall_level}_hour_{hour + 1:02d}.tif"
                    filepath = os.path.join(zone_level_dir, filename)

                    output = flood_data.copy()
                    output[~result['valid_mask']] = -9999.0

                    with rasterio.open(filepath, 'w', **profile) as dst:
                        dst.write(output, 1)

                    total_files += 1

                # 保存流速数据
                for hour, velocity_data in enumerate(result['hourly_velocities']):
                    filename = f"velocity_{zone_id}_{rainfall_level}_hour_{hour + 1:02d}.tif"
                    filepath = os.path.join(zone_level_dir, filename)

                    output = velocity_data.copy()
                    output[~result['valid_mask']] = -9999.0

                    with rasterio.open(filepath, 'w', **profile) as dst:
                        dst.write(output, 1)

                    total_files += 1

                # 保存危险性指数数据
                if 'hourly_hazards' in result:
                    for hour, hazard_data in enumerate(result['hourly_hazards']):
                        filename = f"hazard_{zone_id}_{rainfall_level}_hour_{hour + 1:02d}.tif"
                        filepath = os.path.join(zone_level_dir, filename)

                        output = hazard_data.copy()
                        output[~result['valid_mask']] = -9999.0

                        with rasterio.open(filepath, 'w', **profile) as dst:
                            dst.write(output, 1)

                        total_files += 1

        print(f"  逐小时结果: {total_files} 个文件（包含积水、流速和危险性指数）")
        return hourly_dir

    def _save_hourly_mosaic_results(self, base_dir):
        """保存逐小时拼接的积水结果和流速结果"""
        hourly_mosaic_dir = os.path.join(base_dir, 'hourly_mosaic')
        os.makedirs(hourly_mosaic_dir, exist_ok=True)

        print("\n开始拼接逐小时积水结果和流速结果...")

        # 获取模拟的小时数
        if not self.zone_results:
            print("⚠ 没有模拟结果可拼接")
            return hourly_mosaic_dir

        # 获取第一个分区第一个降雨等级的小时数
        first_zone_id = list(self.zone_results.keys())[0]
        first_rainfall_level = list(self.zone_results[first_zone_id].keys())[0]
        n_hours = len(self.zone_results[first_zone_id][first_rainfall_level]['hourly_floods'])

        # 为每个降雨等级和每个小时进行拼接
        for rainfall_level in self.rainfall_levels.keys():
            print(f"  处理降雨等级: {rainfall_level}")

            for hour in range(n_hours):
                try:
                    # 拼接积水深度
                    flood_mosaic = self.mosaic_hourly_results(rainfall_level, hour, data_type='flood')

                    if flood_mosaic is not None:
                        # 保存积水深度拼接结果
                        output_file = os.path.join(hourly_mosaic_dir,
                                                   f'flood_{rainfall_level}_hour_{hour + 1:02d}.tif')

                        profile = self.dem_profile.copy()
                        profile.update(dtype='float32', nodata=-9999, count=1)

                        with rasterio.open(output_file, 'w', **profile) as dst:
                            dst.write(flood_mosaic, 1)

                        # 输出积水统计信息
                        valid_pixels = flood_mosaic[flood_mosaic != -9999]
                        if len(valid_pixels) > 0:
                            max_depth = np.max(valid_pixels)
                            mean_depth = np.mean(valid_pixels)
                            flooded_area = np.sum(valid_pixels > 1)

                            print(f"    ✓ 积水第{hour + 1:02d}小时: 最大{max_depth:.1f}mm, "
                                  f"平均{mean_depth:.1f}mm, 积水像元{flooded_area}个")
                    else:
                        print(f"    ⚠ 积水第{hour + 1:02d}小时: 无有效数据")

                    # 拼接流速
                    velocity_mosaic = self.mosaic_hourly_results(rainfall_level, hour, data_type='velocity')

                    if velocity_mosaic is not None:
                        # 保存流速拼接结果
                        output_file = os.path.join(hourly_mosaic_dir,
                                                   f'velocity_{rainfall_level}_hour_{hour + 1:02d}.tif')

                        profile = self.dem_profile.copy()
                        profile.update(dtype='float32', nodata=-9999, count=1)

                        with rasterio.open(output_file, 'w', **profile) as dst:
                            dst.write(velocity_mosaic, 1)

                        # 输出流速统计信息
                        valid_pixels = velocity_mosaic[velocity_mosaic != -9999]
                        if len(valid_pixels) > 0:
                            max_velocity = np.max(valid_pixels)
                            mean_velocity = np.mean(valid_pixels)
                            flow_area = np.sum(valid_pixels > 0.01)  # 流速大于0.01m/s的像元

                            print(f"    ✓ 流速第{hour + 1:02d}小时: 最大{max_velocity:.2f}m/s, "
                                  f"平均{mean_velocity:.2f}m/s, 流动像元{flow_area}个")
                    else:
                        print(f"    ⚠ 流速第{hour + 1:02d}小时: 无有效数据")

                    # 拼接危险性指数
                    hazard_mosaic = self.mosaic_hourly_results(rainfall_level, hour, data_type='hazard')

                    if hazard_mosaic is not None:
                        # 保存危险性指数拼接结果
                        output_file = os.path.join(hourly_mosaic_dir,
                                                   f'hazard_{rainfall_level}_hour_{hour + 1:02d}.tif')

                        profile = self.dem_profile.copy()
                        profile.update(dtype='float32', nodata=-9999, count=1)

                        with rasterio.open(output_file, 'w', **profile) as dst:
                            dst.write(hazard_mosaic, 1)

                        # 输出危险性指数统计信息
                        valid_pixels = hazard_mosaic[hazard_mosaic != -9999]
                        if len(valid_pixels) > 0:
                            max_value = np.max(valid_pixels)
                            mean_value = np.mean(valid_pixels)
                            hazard_cells = np.sum(valid_pixels > 1.0)  # H>1.0视为危险

                            print(f"    ✓ 危险性第{hour + 1:02d}小时: 最大{max_value:.2f}, "
                                  f"平均{mean_value:.2f}, 危险像元{hazard_cells}个")
                    else:
                        print(f"    ⚠ 危险性第{hour + 1:02d}小时: 无有效数据")

                except Exception as e:
                    print(f"    ✗ 第{hour + 1:02d}小时拼接失败: {e}")
                    continue

        print(f"✓ 逐小时拼接完成: {hourly_mosaic_dir}")
        return hourly_mosaic_dir

    def mosaic_hourly_results(self, rainfall_level, hour, data_type='flood'):
        """
        拼接指定小时、降雨等级和数据类型的所有分区结果

        参数:
        - rainfall_level: 降雨等级
        - hour: 小时索引（从0开始）
        - data_type: 数据类型，'flood'、'velocity'或'hazard'

        返回:
        - 拼接后的完整栅格数据
        """
        # 初始化输出网格
        output = np.full_like(self.dem_data, -9999.0, dtype=np.float32)

        # 统计有效分区数
        valid_zones = 0

        # 拼接各分区指定小时的结果
        for zone_id, zone_results in self.zone_results.items():
            if rainfall_level not in zone_results:
                continue

            result = zone_results[rainfall_level]
            zone_data = self.zone_dems[zone_id]

            # 检查小时索引是否有效
            if hour >= len(result['hourly_floods']):
                continue

            # 根据数据类型获取数据
            if data_type == 'flood':
                hourly_data = result['hourly_floods'][hour]
            elif data_type == 'velocity':
                hourly_data = result['hourly_velocities'][hour]
            elif data_type == 'hazard':
                if 'hourly_hazards' not in result or hour >= len(result['hourly_hazards']):
                    continue
                hourly_data = result['hourly_hazards'][hour]
            else:
                raise ValueError(f"未知的数据类型: {data_type}")

            valid_mask = result['valid_mask']

            # 合并到输出网格
            self._merge_zone_to_output(output, hourly_data, valid_mask, zone_data)
            valid_zones += 1

        if valid_zones == 0:
            print(f"    ⚠ 降雨等级 {rainfall_level} 第{hour + 1}小时无有效分区数据")
            return None

        return output

    def _save_final_results(self, base_dir):
        """保存最终积水结果和流速结果"""
        final_dir = os.path.join(base_dir, 'final')
        os.makedirs(final_dir, exist_ok=True)

        for rainfall_level in self.rainfall_levels.keys():
            try:
                # 保存最终积水结果
                flood_output_file = os.path.join(final_dir, f'flood_{rainfall_level}.tif')
                flood_result = self.mosaic_results(rainfall_level, flood_output_file, data_type='flood')

                # 保存最终流速结果
                velocity_output_file = os.path.join(final_dir, f'velocity_{rainfall_level}.tif')
                velocity_result = self.mosaic_results(rainfall_level, velocity_output_file, data_type='velocity')

                # 保存最终危险性指数结果
                hazard_output_file = os.path.join(final_dir, f'hazard_{rainfall_level}.tif')
                hazard_result = self.mosaic_results(rainfall_level, hazard_output_file, data_type='hazard')
            except Exception as e:
                print(f"✗ 保存{rainfall_level}最终结果失败: {e}")

        return final_dir

    def mosaic_results(self, rainfall_level='暴雨', output_file=None, data_type='flood'):
        """拼接分区结果为完整栅格"""
        print(f"\n拼接 {rainfall_level} {data_type} 最终结果...")

        if output_file is None:
            if data_type == 'flood':
                output_file = os.path.join(self.out_path, f'flood_{rainfall_level}.tif')
            elif data_type == 'velocity':
                output_file = os.path.join(self.out_path, f'velocity_{rainfall_level}.tif')
            elif data_type == 'hazard':
                output_file = os.path.join(self.out_path, f'hazard_{rainfall_level}.tif')
            else:
                raise ValueError(f"未知的数据类型: {data_type}")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 初始化输出网格
        output = np.full_like(self.dem_data, -9999.0, dtype=np.float32)

        # 拼接各分区结果
        for zone_id, zone_results in self.zone_results.items():
            if rainfall_level not in zone_results:
                continue

            result = zone_results[rainfall_level]
            zone_data = self.zone_dems[zone_id]

            # 根据数据类型获取数据
            if data_type == 'flood':
                data_to_mosaic = result['final_flood']
            elif data_type == 'velocity':
                data_to_mosaic = result['final_velocity']
            elif data_type == 'hazard':
                data_to_mosaic = result['final_hazard']
            else:
                raise ValueError(f"未知的数据类型: {data_type}")

            self._merge_zone_to_output(output, data_to_mosaic,
                                       result['valid_mask'], zone_data)

        # 保存结果
        profile = self.dem_profile.copy()
        profile.update(dtype='float32', nodata=-9999, count=1)

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(output, 1)

        # 输出统计信息
        valid_pixels = output[output != -9999]
        if len(valid_pixels) > 0:
            max_value = np.max(valid_pixels)
            mean_value = np.mean(valid_pixels)

            if data_type == 'flood':
                threshold_area = np.sum(valid_pixels > 1)
                unit = "mm"
            elif data_type == 'velocity':
                threshold_area = np.sum(valid_pixels > 0.01)
                unit = "m/s"
            elif data_type == 'hazard':
                threshold_area = np.sum(valid_pixels > 1.0)
                unit = ""

            print(f"✓ {data_type}拼接完成: {output_file}")
            print(f"  统计: 最大{max_value:.2f}{unit}, 平均{mean_value:.2f}{unit}, "
                  f"有效像元{threshold_area}个")

        return output_file

    def _merge_zone_to_output(self, output, zone_data, valid_mask, zone_info):
        """将分区数据合并到输出网格（优化版本）"""
        zone_tf = zone_info['transform']
        geom = zone_info['geometry']

        # 计算分区在输出网格中的位置
        zone_left = zone_tf.c
        zone_top = zone_tf.f
        col_start, row_start = ~self.dem_transform * (zone_left, zone_top)
        col_start, row_start = int(round(col_start)), int(round(row_start))

        h, w = zone_data.shape
        row_end = row_start + h
        col_end = col_start + w

        # 边界检查
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(output.shape[0], row_end)
        col_end = min(output.shape[1], col_end)

        if row_end <= row_start or col_end <= col_start:
            return

        # 提取有效数据区域
        data_row_start = max(0, -row_start)
        data_col_start = max(0, -col_start)
        data_row_end = data_row_start + (row_end - row_start)
        data_col_end = data_col_start + (col_end - col_start)

        # 确保索引不越界
        data_row_end = min(data_row_end, zone_data.shape[0])
        data_col_end = min(data_col_end, zone_data.shape[1])

        zone_subset = zone_data[data_row_start:data_row_end, data_col_start:data_col_end]
        valid_subset = valid_mask[data_row_start:data_row_end, data_col_start:data_col_end]

        # 创建几何掩膜
        geom_mask = self._create_geometry_mask(zone_info, row_start, row_end, col_start, col_end)

        # 合并数据
        write_mask = geom_mask & valid_subset
        output_slice = output[row_start:row_end, col_start:col_end]

        # 确保输出切片与写入掩膜形状一致
        if output_slice.shape != write_mask.shape:
            min_rows = min(output_slice.shape[0], write_mask.shape[0])
            min_cols = min(output_slice.shape[1], write_mask.shape[1])
            output_slice = output_slice[:min_rows, :min_cols]
            write_mask = write_mask[:min_rows, :min_cols]
            zone_subset = zone_subset[:min_rows, :min_cols]

        # 首次写入或取最大值（处理重叠区域）
        first_time = write_mask & (output_slice == -9999)
        output_slice[first_time] = zone_subset[first_time]

        existing = write_mask & (output_slice != -9999)
        output_slice[existing] = np.maximum(output_slice[existing], zone_subset[existing])

    def _create_geometry_mask(self, zone_info, row_start, row_end, col_start, col_end):
        """创建几何掩膜（优化性能版本）"""
        rows = row_end - row_start
        cols = col_end - col_start
        geom_mask = np.zeros((rows, cols), dtype=bool)
        geom = zone_info['geometry']

        # 批量计算坐标（提高性能）
        col_indices, row_indices = np.meshgrid(
            np.arange(col_start, col_end),
            np.arange(row_start, row_end),
            indexing='xy'
        )

        # 计算所有点的坐标
        x_coords = self.dem_transform.c + col_indices * self.dem_transform.a
        y_coords = self.dem_transform.f + row_indices * self.dem_transform.e

        # 批量检查点是否在几何体内
        points = [Point(x, y) for x, y in zip(x_coords.ravel(), y_coords.ravel())]
        contains_results = [geom.contains(point) for point in points]

        geom_mask = np.array(contains_results).reshape(rows, cols)

        return geom_mask

    def _create_output_profile(self, zone_data):
        """创建输出文件的元数据"""
        return {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999.0,
            'width': zone_data['dem'].shape[1],
            'height': zone_data['dem'].shape[0],
            'count': 1,
            'crs': self.dem_crs,
            'transform': zone_data['transform']
        }

    @staticmethod
    def _pixel_area_m2(transform, shape):
        """计算像元面积(m²) - 考虑地理坐标系"""
        from math import radians, cos

        # 中心纬度
        top_lat = transform.f
        pix_h = abs(transform.e)
        center_lat = top_lat - pix_h * shape[0] / 2.0

        # 经纬度到米的转换系数
        lat_len = 111320.0  # 1°纬度 ≈ 111.32 km
        lon_len = 40075000.0 * cos(radians(center_lat)) / 360.0

        # 像元尺寸（度）
        deg_x = abs(transform.a)
        deg_y = abs(transform.e)

        return lon_len * deg_x * lat_len * deg_y


class AdvancedRainfallPatternGenerator:
    """基于中国暴雨分级标准的多模式降雨生成器"""

    def __init__(self, region='north_china'):
        """
        初始化降雨生成器

        参数:
        - region: 地区类型，可选'north_china'(华北)、'south_china'(华南)、
                 'east_china'(华东)、'central_china'(华中)
        """
        # 中国气象局降雨分级标准（24小时降雨量，单位：mm）
        self.rainfall_levels_china = {
            '小雨': {'min': 0.1, 'max': 9.9, 'color': '#87CEFA', 'intensity_type': 'light'},
            '中雨': {'min': 10.0, 'max': 24.9, 'color': '#6495ED', 'intensity_type': 'moderate'},
            '大雨': {'min': 25.0, 'max': 49.9, 'color': '#4169E1', 'intensity_type': 'heavy'},
            '暴雨': {'min': 50.0, 'max': 99.9, 'color': '#DC143C', 'intensity_type': 'torrential'},
            '大暴雨': {'min': 100.0, 'max': 249.9, 'color': '#8B0000', 'intensity_type': 'severe_torrential'},
            '特大暴雨': {'min': 250.0, 'max': 500.0, 'color': '#4B0082', 'intensity_type': 'extreme_torrential'}
        }

        # 添加缓存字典，用于存储已经生成的雨型
        self._rainfall_cache = {}

        # 不同地区的雨型特征参数
        self.region_params = {
            'north_china': {  # 华北地区：降雨集中，历时短，强度大
                'peak_position': 0.3,  # 峰值位置（相对时间比例）
                'peak_sharpness': 2.5,  # 峰值尖锐度
                'short_duration_ratio': 0.7,  # 短历时比例
                'multi_peak_prob': 0.3  # 多峰概率
            },
            'south_china': {  # 华南地区：降雨历时长，可能有多峰
                'peak_position': 0.4,
                'peak_sharpness': 1.8,
                'short_duration_ratio': 0.4,
                'multi_peak_prob': 0.6
            },
            'east_china': {  # 华东地区：介于华北华南之间
                'peak_position': 0.35,
                'peak_sharpness': 2.0,
                'short_duration_ratio': 0.5,
                'multi_peak_prob': 0.5
            },
            'central_china': {  # 华中地区
                'peak_position': 0.38,
                'peak_sharpness': 2.2,
                'short_duration_ratio': 0.6,
                'multi_peak_prob': 0.4
            }
        }

        self.region = region
        self.region_param = self.region_params.get(region, self.region_params[region])

        # 不同降雨等级的雨型特征
        self.level_pattern_params = {
            '小雨': {'pattern_type': 'uniform', 'peak_multiplier': 1.2, 'duration_factor': 1.0},
            '中雨': {'pattern_type': 'frontal', 'peak_multiplier': 1.5, 'duration_factor': 0.8},
            '大雨': {'pattern_type': 'frontal', 'peak_multiplier': 2.0, 'duration_factor': 0.6},
            '暴雨': {'pattern_type': 'convective', 'peak_multiplier': 3.0, 'duration_factor': 0.5},
            '大暴雨': {'pattern_type': 'convective', 'peak_multiplier': 4.0, 'duration_factor': 0.4},
            '特大暴雨': {'pattern_type': 'extreme', 'peak_multiplier': 5.0, 'duration_factor': 0.3}
        }

        # 随机种子
        np.random.seed(42)

    def generate_comprehensive_rainfall(self, rainfall_level, duration_hours=24,
                                        pattern_type=None, return_period=None):
        """
        生成综合降雨时间序列

        参数:
        - rainfall_level: 降雨等级（如'暴雨'）
        - duration_hours: 总历时（小时）
        - pattern_type: 雨型类型，可选'uniform','frontal','convective','typhoon','extreme','auto'
        - return_period: 重现期（年），用于调整降雨强度分布

        返回:
        - 降雨时间序列字典
        """
        # 创建缓存键，包含所有影响雨型生成的参数
        cache_key = (rainfall_level, duration_hours, pattern_type, return_period, self.region)

        # 检查缓存中是否已存在该雨型
        if cache_key in self._rainfall_cache:
            print('返回已有雨型')
            return self._rainfall_cache[cache_key]

        # 1. 获取降雨量范围
        level_info = self.rainfall_levels_china[rainfall_level]
        min_rain = level_info['min']
        max_rain = level_info['max']

        # 根据重现期调整降雨量
        if return_period:
            total_rainfall = self._adjust_by_return_period(max_rain, return_period)
        else:
            # 使用等级上限作为设计值
            total_rainfall = max_rain

        # 2. 确定雨型
        if pattern_type is None or pattern_type == 'auto':
            pattern_type = self._determine_pattern_type(rainfall_level, duration_hours)

        # 3. 生成基础雨型
        if pattern_type == 'uniform':
            hourly_rain = self._generate_uniform_pattern(total_rainfall, duration_hours)
        elif pattern_type == 'frontal':
            hourly_rain = self._generate_frontal_pattern(total_rainfall, duration_hours)
        elif pattern_type == 'convective':
            hourly_rain = self._generate_convective_pattern(total_rainfall, duration_hours)
        elif pattern_type == 'typhoon':
            hourly_rain = self._generate_typhoon_pattern(total_rainfall, duration_hours)
        elif pattern_type == 'extreme':
            hourly_rain = self._generate_extreme_pattern(total_rainfall, duration_hours)
        else:
            hourly_rain = self._generate_scs_type_ii_pattern(total_rainfall, duration_hours)

        # 4. 根据降雨等级调整峰值
        level_param = self.level_pattern_params[rainfall_level]
        hourly_rain = self._enhance_peak(hourly_rain, level_param['peak_multiplier'])

        # 5. 应用地区特征
        hourly_rain = self._apply_region_characteristics(hourly_rain)

        # 6. 标准化到总降雨量
        hourly_rain = self._normalize_to_total(hourly_rain, total_rainfall)

        # 7. 计算统计指标
        stats = self._calculate_rainfall_statistics(hourly_rain)

        # 生成结果
        result = {
            'time_hours': np.arange(duration_hours),
            'rainfall_per_hour': hourly_rain,
            'cumulative_rainfall': np.cumsum(hourly_rain),
            'total_rainfall': total_rainfall,
            'rainfall_level': rainfall_level,
            'pattern_type': pattern_type,
            'region': self.region,
            'return_period': return_period,
            'statistics': stats
        }

        # 将结果存入缓存
        self._rainfall_cache[cache_key] = result

        return result

    def _determine_pattern_type(self, rainfall_level, duration_hours):
        """根据降雨等级和历时确定雨型"""
        # 短历时（≤6小时）通常为对流性降雨
        if duration_hours <= 6:
            return 'convective'

        # 根据降雨等级确定
        level_mapping = {
            '小雨': 'uniform',
            '中雨': 'frontal',
            '大雨': 'frontal',
            '暴雨': 'convective',
            '大暴雨': 'convective',
            '特大暴雨': 'extreme'
        }

        return level_mapping.get(rainfall_level, 'convective')

    def _generate_uniform_pattern(self, total_mm, duration_hours):
        """生成均匀降雨型（适用于小雨）"""
        # 基础均匀分布
        base = np.ones(duration_hours)

        # 添加轻微随机波动
        noise = np.random.normal(0, 0.1, duration_hours)
        base = base + noise
        base = np.maximum(base, 0.3)  # 确保最小值

        return base

    def _generate_frontal_pattern(self, total_mm, duration_hours):
        """生成锋面降雨型（适用于中雨，历时长，强度相对均匀）"""
        # 创建双峰或三峰分布
        time_points = np.linspace(0, 1, duration_hours)

        # 主峰位置
        main_peak = self.region_param['peak_position']

        # 创建多峰分布
        pattern = np.zeros(duration_hours)

        # 主峰
        pattern += self._gaussian_peak(time_points, main_peak,
                                       self.region_param['peak_sharpness'] * 0.8)

        # 可能添加次峰
        if np.random.random() < self.region_param['multi_peak_prob']:
            # 添加前峰
            if main_peak > 0.3:
                pre_peak = main_peak - 0.15
                pattern += self._gaussian_peak(time_points, pre_peak,
                                               self.region_param['peak_sharpness'] * 0.5) * 0.6

            # 添加后峰
            if main_peak < 0.7:
                post_peak = main_peak + 0.15
                pattern += self._gaussian_peak(time_points, post_peak,
                                               self.region_param['peak_sharpness'] * 0.5) * 0.4

        # 添加基础背景降雨
        pattern += 0.2

        return pattern

    def _generate_convective_pattern(self, total_mm, duration_hours):
        """生成对流降雨型（适用于大雨、暴雨，短时强降雨）"""
        # 对流性降雨：峰值尖锐，历时短
        time_points = np.linspace(0, 1, duration_hours)

        # 主峰位置（偏前）
        main_peak = self.region_param['peak_position'] * 0.8

        # 创建尖锐单峰
        pattern = self._gaussian_peak(time_points, main_peak,
                                      self.region_param['peak_sharpness'] * 1.5)

        # 可能添加小范围对流单体
        for _ in range(np.random.randint(1, 4)):
            sub_peak = np.random.uniform(0.1, 0.9)
            if abs(sub_peak - main_peak) > 0.2:  # 确保不重叠
                pattern += self._gaussian_peak(time_points, sub_peak,
                                               self.region_param['peak_sharpness'] * 2.0) * 0.3

        # 确保最小值
        pattern = np.maximum(pattern, 0.1)

        return pattern

    def _generate_typhoon_pattern(self, total_mm, duration_hours):
        """生成台风降雨型（适用于大暴雨，多峰，历时长）"""
        time_points = np.linspace(0, 1, duration_hours)
        pattern = np.zeros(duration_hours)

        # 台风眼壁降雨（主峰）
        eye_wall_peak = 0.4
        pattern += self._gaussian_peak(time_points, eye_wall_peak,
                                       self.region_param['peak_sharpness'] * 1.2)

        # 螺旋雨带（多个次峰）
        n_rainbands = np.random.randint(2, 5)
        for i in range(n_rainbands):
            band_peak = 0.2 + 0.6 * i / n_rainbands
            intensity = 0.5 + 0.3 * np.random.random()
            pattern += self._gaussian_peak(time_points, band_peak,
                                           self.region_param['peak_sharpness'] * 0.8) * intensity

        # 台风外围降雨（背景）
        pattern += 0.3

        return pattern

    def _generate_extreme_pattern(self, total_mm, duration_hours):
        """生成极端降雨型（适用于特大暴雨）"""
        time_points = np.linspace(0, 1, duration_hours)

        # 极端降雨：多个强对流单体叠加
        pattern = np.zeros(duration_hours)

        # 主对流单体（非常强）
        main_peak = self.region_param['peak_position']
        pattern += self._gaussian_peak(time_points, main_peak,
                                       self.region_param['peak_sharpness'] * 3.0) * 2.0

        # 多个次对流单体
        n_cells = np.random.randint(3, 6)
        for i in range(n_cells):
            cell_peak = np.random.uniform(0.1, 0.9)
            if abs(cell_peak - main_peak) > 0.15:  # 避免重叠
                intensity = 0.8 + 0.4 * np.random.random()
                sharpness = self.region_param['peak_sharpness'] * (2.0 + np.random.random())
                pattern += self._gaussian_peak(time_points, cell_peak, sharpness) * intensity

        # 确保最小值
        pattern = np.maximum(pattern, 0.2)

        return pattern

    def _generate_scs_type_ii_pattern(self, total_mm, duration_hours):
        """生成SCS Type II雨型（作为基准）"""
        # 这里调用之前定义的SCS Type II生成方法
        # 简化实现
        time_points = np.linspace(0, 1, duration_hours)

        # SCS Type II的近似分布
        pattern = np.zeros(duration_hours)

        # 主要降雨集中在中间时段
        main_start = 0.4
        main_end = 0.7

        for i, t in enumerate(time_points):
            if main_start <= t <= main_end:
                # 主要降雨期：抛物线分布
                center = (main_start + main_end) / 2
                distance = abs(t - center) / ((main_end - main_start) / 2)
                pattern[i] = 1.0 - distance ** 2
            else:
                # 次要降雨期
                pattern[i] = 0.2

        return pattern

    def _chicago_peak(self, time_points, peak_position, peak_factor=2.5):
        """生成芝加哥雨型单峰分布"""
        # 芝加哥雨型公式：峰值前 (t/peak_position)^1.5，峰值后 ((1-t)/(1-peak_position))^0.8
        pattern = np.zeros_like(time_points)
        for i, t in enumerate(time_points):
            if t <= peak_position:
                if peak_position > 0:
                    factor = np.power(t / peak_position, 1.5)
                else:
                    factor = 0
            else:
                if peak_position < 1:
                    factor = np.power((1 - t) / (1 - peak_position), 0.8)
                else:
                    factor = 0
            pattern[i] = factor
        # 归一化使峰值高度为1
        max_val = np.max(pattern)
        if max_val > 0:
            pattern = pattern / max_val
        # 应用峰值增强因子
        peak_idx = np.argmax(pattern)
        pattern[peak_idx] *= peak_factor
        # 重新归一化使峰值高度为peak_factor
        max_val = np.max(pattern)
        if max_val > 0:
            pattern = pattern / max_val
        return pattern

    def _gaussian_peak(self, time_points, center, sharpness):
        """生成芝加哥雨型（替换原高斯峰）"""
        # 将sharpness映射为峰值增强因子：1 + sharpness/10
        peak_factor = 1 + sharpness / 10
        return self._chicago_peak(time_points, center, peak_factor)

    def _enhance_peak(self, pattern, multiplier):
        """增强峰值"""
        pattern = pattern.copy()
        peak_idx = np.argmax(pattern)
        pattern[peak_idx] *= multiplier

        # 平滑过渡
        n_smooth = min(3, len(pattern) // 10)
        for i in range(1, n_smooth + 1):
            if peak_idx - i >= 0:
                pattern[peak_idx - i] *= (1 + (multiplier - 1) * (n_smooth - i) / n_smooth)
            if peak_idx + i < len(pattern):
                pattern[peak_idx + i] *= (1 + (multiplier - 1) * (n_smooth - i) / n_smooth)

        return pattern

    def _apply_region_characteristics(self, pattern):
        """应用地区特征"""
        # 华北地区：压缩历时，增强峰值
        if self.region == 'north_china':
            # 压缩非峰值时段
            peak_idx = np.argmax(pattern)
            for i in range(len(pattern)):
                if abs(i - peak_idx) > len(pattern) // 4:
                    pattern[i] *= 0.7

        # 华南地区：延长降雨历时
        elif self.region == 'south_china':
            # 平滑化，延长降雨
            from scipy.ndimage import gaussian_filter1d
            pattern = gaussian_filter1d(pattern, sigma=2)

        return pattern

    def _normalize_to_total(self, pattern, total_mm):
        """标准化到总降雨量"""
        current_total = np.sum(pattern)
        if current_total > 0:
            pattern = pattern * total_mm / current_total
        return pattern

    def _adjust_by_return_period(self, base_rainfall, return_period):
        """根据重现期调整降雨量"""
        # 使用中国暴雨强度公式的一般形式
        # P = P₀ * (1 + C * log(T/T₀))

        # 基准重现期（年）
        T0 = 10

        # 地区调整系数
        region_coefficients = {
            'north_china': {'C': 0.65, 'exponent': 0.85},
            'south_china': {'C': 0.75, 'exponent': 0.78},
            'east_china': {'C': 0.70, 'exponent': 0.82},
            'central_china': {'C': 0.68, 'exponent': 0.84}
        }

        coeff = region_coefficients.get(self.region, region_coefficients['north_china'])

        # 计算调整后的降雨量
        if return_period <= T0:
            adjusted = base_rainfall * (return_period / T0) ** coeff['exponent']
        else:
            adjusted = base_rainfall * (1 + coeff['C'] * np.log10(return_period / T0))

        return adjusted

    def _calculate_rainfall_statistics(self, hourly_rain):
        """计算降雨统计指标"""
        stats = {
            'total': np.sum(hourly_rain),
            'max_hourly': np.max(hourly_rain),
            'mean_hourly': np.mean(hourly_rain),
            'peak_time': np.argmax(hourly_rain) + 0.5,  # 时段中点
            'peak_intensity': np.max(hourly_rain),
            'peak_to_mean_ratio': np.max(hourly_rain) / np.mean(hourly_rain) if np.mean(hourly_rain) > 0 else 0,
            'rainfall_duration': self._calculate_effective_duration(hourly_rain),
            'rainfall_centroid': self._calculate_rainfall_centroid(hourly_rain)
        }

        return stats

    def _calculate_effective_duration(self, hourly_rain, threshold_ratio=0.1):
        """计算有效降雨历时"""
        max_intensity = np.max(hourly_rain)
        threshold = max_intensity * threshold_ratio

        # 找出超过阈值的时段
        significant_hours = np.where(hourly_rain > threshold)[0]

        if len(significant_hours) == 0:
            return 0

        # 有效历时 = 最后一个超过阈值的时段 - 第一个超过阈值的时段 + 1
        effective_duration = significant_hours[-1] - significant_hours[0] + 1

        return effective_duration

    def _calculate_rainfall_centroid(self, hourly_rain):
        """计算降雨质心（降雨分布的中心时刻）"""
        if np.sum(hourly_rain) == 0:
            return 0

        weighted_sum = np.sum(hourly_rain * np.arange(len(hourly_rain)))
        centroid = weighted_sum / np.sum(hourly_rain)

        return centroid

    def visualize_pattern_comparison(self, duration_hours=24):
        """可视化不同降雨等级的雨型对比"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        levels = list(self.rainfall_levels_china.keys())

        for idx, level in enumerate(levels):
            if idx >= len(axes):
                break

            # 生成降雨序列
            result = self.generate_comprehensive_rainfall(level, duration_hours, pattern_type='auto')
            hourly_rain = result['rainfall_per_hour']
            stats = result['statistics']

            ax = axes[idx]

            # 柱状图
            bars = ax.bar(np.arange(duration_hours), hourly_rain,
                          alpha=0.7,
                          color=self.rainfall_levels_china[level]['color'])

            # 标记峰值
            peak_idx = int(stats['peak_time'] - 0.5)
            if 0 <= peak_idx < len(bars):
                bars[peak_idx].set_alpha(1.0)
                bars[peak_idx].set_edgecolor('black')
                bars[peak_idx].set_linewidth(2)

            ax.set_title(f"{level} ({stats['total']:.1f}mm)",
                         fontsize=12, fontweight='bold',
                         color=self.rainfall_levels_china[level]['color'])

            ax.set_xlabel("时间(小时)", fontsize=10)
            ax.set_ylabel("降雨量(mm/h)", fontsize=10)

            # 添加统计信息
            info_text = (f"峰值: {stats['peak_intensity']:.1f}mm/h\n"
                         f"历时: {stats['rainfall_duration']:.0f}h\n"
                         f"质心: {stats['rainfall_centroid']:.1f}h")

            ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

        plt.suptitle(f"不同降雨等级雨型对比 - {self.region}地区", fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存图片
        plt.savefig(f'rainfall_patterns_{self.region}.png', dpi=300, bbox_inches='tight')
        plt.show()


class SCSCNFloodSimulatorEnhanced(SCSCNFloodSimulator):
    """增强版SCS-CN洪水模拟器（集成改进的降雨生成）"""

    def __init__(self, dem_path, shp_path, landcover_path, out_path,
                 soil_type_path=None, zone_id_field=None, use_physical_route=True,
                 rainfall_region='north_china', rainfall_pattern='auto'):
        """初始化增强版模拟器"""
        super().__init__(dem_path, shp_path, landcover_path, out_path,
                         soil_type_path, zone_id_field, use_physical_route)

        # 初始化改进的降雨生成器
        self.rainfall_generator = AdvancedRainfallPatternGenerator(region=rainfall_region)
        self.rainfall_region = rainfall_region
        self.rainfall_pattern = rainfall_pattern

        # 新增：降雨-径流转换参数
        self._setup_enhanced_parameters()

    def _setup_enhanced_parameters(self):
        """设置增强参数"""
        # 降雨历时与等级关系
        self.rainfall_duration_by_level = {
            '小雨': 12,  # 小雨历时长
            '中雨': 12,
            '大雨': 8,
            '暴雨': 6,  # 暴雨历时短但强度大
            '大暴雨': 6,
            '特大暴雨': 6
        }

        # 不同降雨等级的产流参数调整
        self.runoff_coefficient_by_level = {
            '小雨': 0.1,  # 小雨产流系数低
            '中雨': 0.2,
            '大雨': 0.3,
            '暴雨': 0.5,
            '大暴雨': 0.7,
            '特大暴雨': 0.9
        }

        # 前期土壤湿度条件（AMC）与降雨等级的关系
        self.amc_by_level = {
            '小雨': 'I',  # 小雨前土壤较干
            '中雨': 'II',
            '大雨': 'II',
            '暴雨': 'III',  # 暴雨前土壤较湿
            '大暴雨': 'III',
            '特大暴雨': 'III'
        }

    def generate_rainfall_series(self, rainfall_level, duration_hours=None, pattern_type=None):
        """生成降雨时间序列（使用改进的方法）"""
        # 保持与visualize_pattern_comparison的参数一致性
        # 如果没有指定历时，使用24小时以匹配可视化函数的默认值
        if duration_hours is None:
            duration_hours = 24

        if pattern_type is None:
            pattern_type = 'auto'

        # 获取降雨量范围
        level_info = self.rainfall_generator.rainfall_levels_china[rainfall_level]
        max_rain = level_info['max']

        # 使用改进的降雨生成器，确保参数与visualize_pattern_comparison一致
        rainfall_data = self.rainfall_generator.generate_comprehensive_rainfall(
            rainfall_level=rainfall_level,
            duration_hours=duration_hours,
            pattern_type=pattern_type,
            return_period=None
        )

        # 输出降雨特征
        stats = rainfall_data['statistics']
        print(f"🌧️  降雨特征:")
        print(f"   等级: {rainfall_level}, 总雨量: {stats['total']:.1f}mm")
        print(f"   历时: {duration_hours}h, 峰值: {stats['peak_intensity']:.1f}mm/h")
        print(f"   雨型: {rainfall_data['pattern_type']}, 地区: {self.rainfall_region}")
        print(f"   质心: {stats['rainfall_centroid']:.1f}h, 峰均比: {stats['peak_to_mean_ratio']:.1f}")

        return rainfall_data

    def simulate_zone_enhanced(self, zone_id, rainfall_level='暴雨',
                               duration_hours=None, amc=None):
        """增强版分区模拟"""
        # 清空汇流队列，避免历史模拟残留影响当前模拟
        self.flow_delay_queues = {}

        print(f"\n{'=' * 60}")
        print(f"增强模拟分区 {zone_id} - {rainfall_level}")
        print(f"{'=' * 60}")

        # 检查分区是否存在
        if zone_id not in self.zone_dems:
            raise ValueError(f"分区 {zone_id} 不存在")

        # 自动确定AMC
        if amc is None:
            amc = self.amc_by_level.get(rainfall_level, 'II')

        # 自动确定历时
        if duration_hours is None:
            duration_hours = self.rainfall_duration_by_level.get(rainfall_level, 6)

        # 获取分区数据
        zone_data = self.zone_dems[zone_id]
        lc_data = self.zone_landcovers[zone_id]

        dem = zone_data['dem']
        valid_mask = zone_data['valid_mask']
        landcover = lc_data['landcover']

        # 生成改进的降雨时间序列
        rainfall_data = self.generate_rainfall_series(rainfall_level, duration_hours)
        hourly_rain = rainfall_data['rainfall_per_hour']

        # DEM预处理
        filled_dem = self.preprocess_dem(dem)
        pixel_area = self._pixel_area_m2(zone_data['transform'], dem.shape)

        # 计算CN参数（使用自动确定的AMC）
        cn_params = self.calculate_cn_parameters(zone_id, amc)
        cn_adjusted = cn_params['cn_adjusted']

        print(f"📊  CN值统计: 平均={np.mean(cn_adjusted[valid_mask]):.1f}, "
              f"峰均比={np.max(cn_adjusted[valid_mask]) / np.mean(cn_adjusted[valid_mask]):.2f}")

        # 根据降雨等级调整产流参数
        runoff_coeff = 1
        # 初始化积水深度、流速和危险性指数
        current_flood = np.zeros_like(dem, dtype=np.float32)
        current_velocity = np.zeros_like(dem, dtype=np.float32)
        hourly_floods = []
        hourly_velocities = []
        hourly_hazards = []
        hourly_runoffs = []

        print("\n⏳ 逐小时模拟:")
        # 逐小时模拟
        for hour, hourly_rain_mm in enumerate(hourly_rain):
            # 计算径流（考虑降雨等级的产流系数）
            base_runoff = self.calculate_scs_runoff(hourly_rain_mm, cn_adjusted)

            # 应用降雨等级调整
            adjusted_runoff = base_runoff  # 取消产流系数修改
            adjusted_runoff[~valid_mask] = 0

            # 汇流计算（使用改进的汇流方法）
            current_flood, current_velocity = self.route_runoff_to_flood_enhanced(
                adjusted_runoff, current_flood, dem, filled_dem,
                valid_mask, landcover, pixel_area, 3600, rainfall_level
            )

            hourly_floods.append(current_flood.copy())
            hourly_velocities.append(current_velocity.copy())
            hourly_runoffs.append(adjusted_runoff.copy())

            # 计算危险性指数
            hazard = self.calculate_hazard_index(current_flood, current_velocity)
            hourly_hazards.append(hazard.copy())

            # 输出本小时结果
            max_depth = np.max(current_flood[valid_mask]) if np.any(valid_mask) else 0
            avg_depth = np.mean(current_flood[valid_mask]) if np.any(valid_mask) else 0
            max_velocity = np.max(current_velocity[valid_mask]) if np.any(valid_mask) else 0
            avg_velocity = np.mean(current_velocity[valid_mask]) if np.any(valid_mask) else 0
            flooded_cells = np.sum(current_flood[valid_mask] > 0)
            flow_cells = np.sum(current_velocity[valid_mask] > 0)

            if hourly_rain_mm > 0 or max_depth > 0:
                print(f"  第{hour + 1:2d}h: 降雨{hourly_rain_mm:5.1f}mm, "
                      f"径流{np.mean(adjusted_runoff[valid_mask]):5.1f}mm, "
                      f"平均积水{avg_depth:5.1f}mm, "
                      f"平均流速{avg_velocity:5.2f}m/s, "
                      f"积水像元{flooded_cells}个，"
                      f"最大积水{max_depth:5.1f}mm，"
                      f"最大流速{max_velocity:5.2f}m/s, "
                      f"流水单元{flow_cells}个")

        # 整理结果
        result = {
            'zone_id': zone_id,
            'rainfall_level': rainfall_level,
            'rainfall_data': rainfall_data,
            'amc': amc,
            'hourly_floods': hourly_floods,
            'hourly_velocities': hourly_velocities,
            'hourly_hazards': hourly_hazards,
            'hourly_runoffs': hourly_runoffs,
            'final_flood': current_flood,
            'final_velocity': current_velocity,
            'final_hazard': hazard,
            'cn_array': cn_adjusted,
            'valid_mask': valid_mask,
            'runoff_coefficient': runoff_coeff,
            'max_flood_depth': np.max(current_flood[valid_mask]) if np.any(valid_mask) else 0,
            'max_velocity': np.max(current_velocity[valid_mask]) if np.any(valid_mask) else 0,
            'max_hazard': np.max(hazard[valid_mask]) if np.any(valid_mask) else 0,
            'total_flood_volume': np.sum(current_flood[valid_mask]) * pixel_area * 1e-3 if np.any(valid_mask) else 0
            # m³
        }

        # 输出最终统计
        total_rain = np.sum(hourly_rain)
        total_runoff = np.sum([np.mean(r[valid_mask]) for r in hourly_runoffs if np.any(valid_mask)])
        overall_runoff_coeff = total_runoff / total_rain if total_rain > 0 else 0

        print(f"\n✅ 模拟完成!")
        print(f"   总降雨: {total_rain:.1f}mm")
        print(f"   总径流: {total_runoff:.1f}mm")
        print(f"   综合径流系数: {overall_runoff_coeff:.3f}")
        print(f"   最大积水深度: {result['max_flood_depth']:.1f}mm")
        print(f"   最大流速: {result['max_velocity']:.2f}m/s")
        print(f"   最大危险性指数: {result['max_hazard']:.2f}")
        print(f"   总积水量: {result['total_flood_volume']:.1f} m³")

        return result

    def route_runoff_to_flood_enhanced(self, runoff_mm, flood_mm_prev, dem, filled_dem,
                                       valid_mask, landcover, pixel_area, dt=3600,
                                       rainfall_level='暴雨'):
        """
        增强版汇流计算：集成动态地形和多模式汇流
        返回：积水深度(mm)和流速(m/s)
        """
        # 这里可以集成之前讨论的动态地形法、多流向法等改进
        # 暂时调用父类方法，后续可替换为改进版本
        return super().route_runoff_to_flood(runoff_mm, flood_mm_prev, dem, filled_dem,
                                             valid_mask, landcover, pixel_area, dt, rainfall_level)

    def run_sensitivity_analysis(self, zone_id=42, duration_hours=24):
        """
        参数单因子敏感性分析
        针对主峰位置参数r(peak_position)和主峰增强系数p(peak_multiplier)
        每个参数取标准值、90%、110%，计算结果变化幅度/参数变化幅度
        结果采用各降雨情境下final_hazard的空间均值
        """
        rainfall_levels = ['小雨', '中雨', '大雨', '暴雨', '大暴雨', '特大暴雨']
        sens_records = []

        print("\n" + "=" * 70)
        print("开始参数单因子敏感性分析")
        print("=" * 70)
        print(f"分析分区: {zone_id}, 历时: {duration_hours}h")
        print("参数定义: r=主峰位置(peak_position), p=主峰增强系数(peak_multiplier)")
        print("敏感性指数 = (ΔH/H_base) / (ΔP/P_base)")
        print("=" * 70)

        # 备份原始参数
        original_r = self.rainfall_generator.region_param['peak_position']

        for level in rainfall_levels:
            print(f"\n{'=' * 60}")
            print(f"【降雨情境】{level}")
            print(f"{'=' * 60}")

            # 备份该等级的原始p
            original_p = self.rainfall_generator.level_pattern_params[level]['peak_multiplier']
            base_r = original_r
            base_p = original_p

            # ---------- 基准运行 ----------
            print(f"\n[1/5] 基准参数: r={base_r:.4f}, p={base_p:.2f}")
            self.flow_delay_queues = {}
            self.rainfall_generator._rainfall_cache.clear()
            result_base = self.simulate_zone_enhanced(zone_id, level, duration_hours)

            if result_base is None or 'final_hazard' not in result_base:
                print(f"⚠ {level} 基准模拟失败，跳过")
                continue
            valid_mask = result_base['valid_mask']
            if not np.any(valid_mask):
                print(f"⚠ {level} 无有效区域，跳过")
                continue
            h_base = float(np.mean(result_base['final_hazard'][valid_mask]))
            print(f"   基准危险性指数均值 H_base = {h_base:.4f}")

            # ---------- r = 90% ----------
            print(f"\n[2/5] r调整为90%: {base_r * 0.9:.4f}")
            self.rainfall_generator.region_param['peak_position'] = base_r * 0.9
            self.flow_delay_queues = {}
            self.rainfall_generator._rainfall_cache.clear()
            result_r90 = self.simulate_zone_enhanced(zone_id, level, duration_hours)
            h_r90 = float(np.mean(result_r90['final_hazard'][result_r90['valid_mask']])) if (
                        result_r90 and np.any(result_r90['valid_mask'])) else 0.0
            print(f"   H(r-90%) = {h_r90:.4f}")

            # ---------- r = 110% ----------
            print(f"\n[3/5] r调整为110%: {base_r * 1.1:.4f}")
            self.rainfall_generator.region_param['peak_position'] = base_r * 1.1
            self.flow_delay_queues = {}
            self.rainfall_generator._rainfall_cache.clear()
            result_r110 = self.simulate_zone_enhanced(zone_id, level, duration_hours)
            h_r110 = float(np.mean(result_r110['final_hazard'][result_r110['valid_mask']])) if (
                        result_r110 and np.any(result_r110['valid_mask'])) else 0.0
            print(f"   H(r-110%) = {h_r110:.4f}")

            # 恢复r
            self.rainfall_generator.region_param['peak_position'] = base_r

            # ---------- p = 90% ----------
            print(f"\n[4/5] p调整为90%: {base_p * 0.9:.2f}")
            self.rainfall_generator.level_pattern_params[level]['peak_multiplier'] = base_p * 0.9
            self.flow_delay_queues = {}
            self.rainfall_generator._rainfall_cache.clear()
            result_p90 = self.simulate_zone_enhanced(zone_id, level, duration_hours)
            h_p90 = float(np.mean(result_p90['final_hazard'][result_p90['valid_mask']])) if (
                        result_p90 and np.any(result_p90['valid_mask'])) else 0.0
            print(f"   H(p-90%) = {h_p90:.4f}")

            # ---------- p = 110% ----------
            print(f"\n[5/5] p调整为110%: {base_p * 1.1:.2f}")
            self.rainfall_generator.level_pattern_params[level]['peak_multiplier'] = base_p * 1.1
            self.flow_delay_queues = {}
            self.rainfall_generator._rainfall_cache.clear()
            result_p110 = self.simulate_zone_enhanced(zone_id, level, duration_hours)
            h_p110 = float(np.mean(result_p110['final_hazard'][result_p110['valid_mask']])) if (
                        result_p110 and np.any(result_p110['valid_mask'])) else 0.0
            print(f"   H(p-110%) = {h_p110:.4f}")

            # 恢复p
            self.rainfall_generator.level_pattern_params[level]['peak_multiplier'] = base_p

            # ---------- 计算敏感性 ----------
            delta_param = 0.2  # (1.1 - 0.9)
            if abs(h_base) > 1e-6:
                sens_r = ((h_r110 - h_r90) / h_base) / delta_param
                sens_p = ((h_p110 - h_p90) / h_base) / delta_param
            else:
                sens_r = 0.0
                sens_p = 0.0
                print(f"⚠ 基准H接近0，敏感性设为0")

            print(f"\n📊 {level} 敏感性指数:")
            print(f"   S_r = {sens_r:.4f}")
            print(f"   S_p = {sens_p:.4f}")

            sens_records.append([
                level, base_r, base_p, h_base,
                h_r90, h_r110, round(sens_r, 4),
                h_p90, h_p110, round(sens_p, 4)
            ])

        # 恢复原始参数
        self.rainfall_generator.region_param['peak_position'] = original_r

        # 输出汇总表
        print("\n" + "=" * 90)
        print("参数单因子敏感性分析汇总结果")
        print("=" * 90)
        header = f"{'情境':<8} {'r_base':<8} {'p_base':<8} {'H_base':<10} {'H_r90':<10} {'H_r110':<10} {'S_r':<10} {'H_p90':<10} {'H_p110':<10} {'S_p':<10}"
        print(header)
        print("-" * 90)
        for rec in sens_records:
            line = f"{rec[0]:<8} {rec[1]:<8.4f} {rec[2]:<8.2f} {rec[3]:<10.4f} {rec[4]:<10.4f} {rec[5]:<10.4f} {rec[6]:<10.4f} {rec[7]:<10.4f} {rec[8]:<10.4f} {rec[9]:<10.4f}"
            print(line)
        print("=" * 90)

        # 保存CSV
        csv_path = os.path.join(self.out_path, 'sensitivity_analysis_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['降雨情境', 'r_base', 'p_base', 'H_base',
                             'H_r90%', 'H_r110%', 'S_r',
                             'H_p90%', 'H_p110%', 'S_p'])
            writer.writerows(sens_records)
        print(f"\n✓ 结果已保存至: {csv_path}")

        return sens_records

    def batch_simulate_enhanced(self, rainfall_levels=None, duration_hours=24, save_dir=None):
        """增强版批量模拟"""

        print("\n📈 生成降雨雨型对比图...")

        self.rainfall_generator.visualize_pattern_comparison(duration_hours=duration_hours)

        if rainfall_levels is None:
            rainfall_levels = list(self.rainfall_levels.keys())
        """self.zone_dems.keys()"""
        for zone_id in [46]:
            zone_results = {}

            for level in rainfall_levels:
                try:
                    result = self.simulate_zone_enhanced(zone_id, level, duration_hours)
                    zone_results[level] = result
                except Exception as e:
                    print(f"✗ 分区{zone_id}-{level}模拟失败: {e}")
                    continue

            # 模拟完成后立即保存该区域的结果
            try:
                print(f"\n💾 正在保存分区{zone_id}的模拟结果...")
                self.save_zone_results(zone_id, zone_results, save_dir)
            except Exception as e:
                print(f"✗ 分区{zone_id}结果保存失败: {e}")

        self.zone_results = 0
        return 0

    def save_zone_results(self, zone_id, zone_results, save_dir=None):
        """保存单个区域的模拟结果"""
        if save_dir is None:
            save_dir = os.path.join(self.out_path, 'simulation_results')
        os.makedirs(save_dir, exist_ok=True)

        # 保存当前区域的逐小时结果（积水和流速）
        self._save_single_zone_hourly_results(save_dir, zone_id, zone_results)

        print(f"✓ 区域{zone_id}结果保存完成!")
        return save_dir

    def _save_single_zone_hourly_results(self, base_dir, zone_id, zone_results):
        """保存单个区域的逐小时积水结果和流速结果"""
        hourly_dir = os.path.join(base_dir, 'hourly')
        os.makedirs(hourly_dir, exist_ok=True)

        total_files = 0

        # 只处理指定区域的结果
        for rainfall_level, result in zone_results.items():
            zone_level_dir = os.path.join(hourly_dir, f'zone_{zone_id}', rainfall_level)
            os.makedirs(zone_level_dir, exist_ok=True)

            zone_data = self.zone_dems[zone_id]
            profile = self._create_output_profile(zone_data)

            # 保存积水深度数据
            for hour, flood_data in enumerate(result['hourly_floods']):
                filename = f"flood_{zone_id}_{rainfall_level}_hour_{hour + 1:02d}.tif"
                filepath = os.path.join(zone_level_dir, filename)

                output = flood_data.copy()
                output[~result['valid_mask']] = -9999.0

                with rasterio.open(filepath, 'w', **profile) as dst:
                    dst.write(output, 1)

                total_files += 1

            # 保存流速数据
            for hour, velocity_data in enumerate(result['hourly_velocities']):
                filename = f"velocity_{zone_id}_{rainfall_level}_hour_{hour + 1:02d}.tif"
                filepath = os.path.join(zone_level_dir, filename)

                output = velocity_data.copy()
                output[~result['valid_mask']] = -9999.0

                with rasterio.open(filepath, 'w', **profile) as dst:
                    dst.write(output, 1)

                total_files += 1

            # 保存危险性指数数据
            if 'hourly_hazards' in result:
                for hour, hazard_data in enumerate(result['hourly_hazards']):
                    filename = f"hazard_{zone_id}_{rainfall_level}_hour_{hour + 1:02d}.tif"
                    filepath = os.path.join(zone_level_dir, filename)

                    output = hazard_data.copy()
                    output[~result['valid_mask']] = -9999.0

                    with rasterio.open(filepath, 'w', **profile) as dst:
                        dst.write(output, 1)

                    total_files += 1

        print(f"  区域{zone_id}逐小时结果: {total_files} 个文件（包含积水、流速和危险性指数）")
        return hourly_dir


class FloodMosaicGenerator:
    def __init__(self, dem_path, shp_path, simulation_results_dir, output_dir, data_type='flood'):
        """初始化拼接生成器

        参数:
        - data_type: 数据类型，'flood'为积水深度，'velocity'为流速，'hazard'为危险性指数
        """
        self.dem_path = dem_path
        self.shp_path = shp_path
        self.simulation_dir = simulation_results_dir
        self.output_dir = output_dir
        self.data_type = data_type  # 'flood' 、'velocity' 或 'hazard'

        # 数据存储
        self.dem_data = None
        self.dem_transform = None
        self.dem_crs = None
        self.dem_profile = None
        self.zones_gdf = None

        # 使用Zone作为分区字段
        self.zone_id_field = 'Zone'

        # 结果目录
        self.hourly_dir = os.path.join(self.simulation_dir, 'hourly')
        self.output_mosaic_dir = os.path.join(self.output_dir, f'hourly_mosaic_{self.data_type}')

        # 创建输出目录
        os.makedirs(self.output_mosaic_dir, exist_ok=True)

    def load_base_data(self):
        """加载基础地理数据"""
        print("=" * 50)
        print(f"加载基础地理数据...")

        try:
            # 加载DEM
            with rasterio.open(self.dem_path) as src:
                self.dem_data = src.read(1)
                self.dem_transform = src.transform
                self.dem_crs = src.crs
                self.dem_profile = src.profile.copy()
                self.dem_profile.update(dtype='float32', nodata=-9999)

            print(f"✓ DEM加载成功: {self.dem_data.shape}")

            # 加载分区矢量
            self.zones_gdf = gpd.read_file(self.shp_path)
            print(f"✓ 分区矢量加载成功: {len(self.zones_gdf)} 个分区")

            # 检查zone_id字段是否存在
            if self.zone_id_field not in self.zones_gdf.columns:
                print(f"✗ 错误: 矢量文件中不存在 {self.zone_id_field} 字段")
                print(f"  可用字段: {list(self.zones_gdf.columns)}")
                # 如果没有zone_id字段，创建一个
                self.zones_gdf['zone_id'] = range(len(self.zones_gdf))
                print(f"  ✅ 已创建zone_id字段")

            # 坐标系统一
            if self.zones_gdf.crs != self.dem_crs:
                print(f"  坐标系统一: {self.zones_gdf.crs} -> {self.dem_crs}")
                self.zones_gdf = self.zones_gdf.to_crs(self.dem_crs)

            print(f"✓ 使用分区字段: {self.zone_id_field}")
            print(f"  分区ID示例: {self.zones_gdf[self.zone_id_field].head().tolist()}")
            print("=" * 50)

        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            raise

    def discover_simulation_results(self):
        """发现可用的模拟结果"""
        print(f"\n发现模拟结果({self.data_type})...")

        if not os.path.exists(self.hourly_dir):
            raise ValueError(f"模拟结果目录不存在: {self.hourly_dir}")

        # 发现分区目录
        zone_dirs = [d for d in os.listdir(self.hourly_dir)
                     if os.path.isdir(os.path.join(self.hourly_dir, d)) and d.startswith('zone_')]

        if not zone_dirs:
            raise ValueError("未找到分区结果目录")

        # 解析分区ID
        zone_ids = []
        for zone_dir in zone_dirs:
            try:
                zone_id = int(zone_dir.split('_')[1])
                zone_ids.append(zone_id)
            except:
                continue

        print(f"✓ 发现 {len(zone_ids)} 个分区的结果")

        # 获取降雨等级
        if zone_ids:
            first_zone_dir = os.path.join(self.hourly_dir, f'zone_{zone_ids[0]}')
            rainfall_levels = [d for d in os.listdir(first_zone_dir)
                               if os.path.isdir(os.path.join(first_zone_dir, d))]
        else:
            rainfall_levels = []

        if not rainfall_levels:
            raise ValueError("未找到降雨等级目录")

        print(f"✓ 发现降雨等级: {rainfall_levels}")

        # 获取小时数
        if zone_ids and rainfall_levels:
            first_rainfall_dir = os.path.join(first_zone_dir, rainfall_levels[0])
            # 根据数据类型筛选文件
            if self.data_type == 'flood':
                hour_files = [f for f in os.listdir(first_rainfall_dir)
                              if f.endswith('.tif') and 'hour_' in f and f.startswith('flood_')]
            elif self.data_type == 'velocity':
                hour_files = [f for f in os.listdir(first_rainfall_dir)
                              if f.endswith('.tif') and 'hour_' in f and f.startswith('velocity_')]
            elif self.data_type == 'hazard':
                hour_files = [f for f in os.listdir(first_rainfall_dir)
                              if f.endswith('.tif') and 'hour_' in f and f.startswith('hazard_')]
            else:
                hour_files = []

            # 解析小时数
            hours = []
            for file in hour_files:
                try:
                    # 从文件名中提取小时，如 flood_1_暴雨_hour_01.tif
                    hour_str = file.split('_')[-1].replace('.tif', '')
                    hour = int(hour_str)
                    hours.append(hour)
                except:
                    continue

            hours = sorted(set(hours))
            n_hours = len(hours)
        else:
            hours = []
            n_hours = 0

        print(f"✓ 发现 {n_hours} 小时的模拟结果: {hours}")

        return {
            'zone_ids': zone_ids,
            'rainfall_levels': rainfall_levels,
            'hours': hours,
            'n_hours': n_hours
        }

    def create_zone_masks(self):
        """为每个分区创建几何掩膜"""
        print(f"\n创建分区掩膜...")

        zone_masks = {}

        with rasterio.open(self.dem_path) as src:
            for _, zone in self.zones_gdf.iterrows():
                zone_id = zone[self.zone_id_field]

                try:
                    # 创建几何掩膜
                    mask_array = geometry_mask(
                        [zone.geometry],
                        out_shape=src.shape,
                        transform=src.transform,
                        invert=True  # True表示几何内为True
                    )

                    zone_masks[zone_id] = mask_array

                except Exception as e:
                    print(f"⚠ 分区 {zone_id} 创建掩膜失败: {e}")

        print(f"✓ 创建了 {len(zone_masks)} 个分区掩膜")
        return zone_masks

    def load_zone_result(self, zone_id, rainfall_level, hour):
        """加载分区结果"""
        # 根据数据类型构建文件名
        if self.data_type == 'flood':
            filename = f"flood_{zone_id}_{rainfall_level}_hour_{hour:02d}.tif"
        elif self.data_type == 'velocity':
            filename = f"velocity_{zone_id}_{rainfall_level}_hour_{hour:02d}.tif"
        elif self.data_type == 'hazard':
            filename = f"hazard_{zone_id}_{rainfall_level}_hour_{hour:02d}.tif"
        else:
            raise ValueError(f"未知的数据类型: {self.data_type}")

        filepath = os.path.join(self.hourly_dir, f'zone_{zone_id}', rainfall_level, filename)

        if not os.path.exists(filepath):
            # 检查是否存在带前导零的文件名
            alt_filename = f"{self.data_type}_{zone_id}_{rainfall_level}_hour_{hour:02d}.tif"
            alt_filepath = os.path.join(self.hourly_dir, f'zone_{zone_id}', rainfall_level, alt_filename)
            if os.path.exists(alt_filepath):
                filepath = alt_filepath
            else:
                return None

        try:
            with rasterio.open(filepath) as src:
                data = src.read(1).astype(np.float32)
                transform = src.transform

                # 处理nodata值
                if src.nodata is not None:
                    data[data == src.nodata] = -9999
                else:
                    # 默认将小于0的值设为nodata
                    data[data < 0] = -9999

                return data, transform

        except Exception as e:
            print(f"✗ 加载分区{zone_id} {self.data_type} 数据失败: {e}")
            return None

    def mosaic_hourly_data(self, rainfall_level, hour, zone_masks):
        """拼接指定小时和降雨等级的数据"""
        print(f"  拼接 {rainfall_level} 第{hour:02d}小时...")

        # 初始化输出网格
        output = np.full_like(self.dem_data, -9999.0, dtype=np.float32)

        # 获取分区ID列表
        discovery = self.discover_simulation_results()
        zone_ids = discovery['zone_ids']

        successful_zones = 0

        for zone_id in zone_ids:
            # 加载分区结果
            result = self.load_zone_result(zone_id, rainfall_level, hour)
            if result is None:
                continue

            zone_data, zone_transform = result

            # 获取分区掩膜
            if zone_id not in zone_masks:
                continue

            zone_mask = zone_masks[zone_id]

            # 计算分区在全局DEM中的位置
            zone_left = zone_transform.c
            zone_top = zone_transform.f
            col_start, row_start = ~self.dem_transform * (zone_left, zone_top)
            col_start, row_start = int(round(col_start)), int(round(row_start))

            h, w = zone_data.shape
            row_end = row_start + h
            col_end = col_start + w

            # 边界检查
            row_start = max(0, row_start)
            col_start = max(0, col_start)
            row_end = min(output.shape[0], row_end)
            col_end = min(output.shape[1], col_end)

            if row_end <= row_start or col_end <= col_start:
                continue

            # 提取数据子集
            data_row_start = max(0, -row_start)
            data_col_start = max(0, -col_start)
            data_row_end = data_row_start + (row_end - row_start)
            data_col_end = data_col_start + (col_end - col_start)

            # 确保索引不越界
            data_row_end = min(data_row_end, zone_data.shape[0])
            data_col_end = min(data_col_end, zone_data.shape[1])

            # 获取数据子集和掩膜子集
            data_subset = zone_data[data_row_start:data_row_end, data_col_start:data_col_end]
            mask_subset = zone_mask[row_start:row_end, col_start:col_end]

            # 确保形状一致
            if data_subset.shape != mask_subset.shape:
                min_rows = min(data_subset.shape[0], mask_subset.shape[0])
                min_cols = min(data_subset.shape[1], mask_subset.shape[1])
                data_subset = data_subset[:min_rows, :min_cols]
                mask_subset = mask_subset[:min_rows, :min_cols]

            # 创建有效数据掩膜
            valid_data_mask = data_subset != -9999

            # 合并掩膜
            write_mask = mask_subset & valid_data_mask

            if np.any(write_mask):
                # 获取输出切片
                output_slice = output[row_start:row_start + write_mask.shape[0],
                col_start:col_start + write_mask.shape[1]]

                # 确保形状一致
                if output_slice.shape != write_mask.shape:
                    min_rows = min(output_slice.shape[0], write_mask.shape[0])
                    min_cols = min(output_slice.shape[1], write_mask.shape[1])
                    output_slice = output_slice[:min_rows, :min_cols]
                    write_mask = write_mask[:min_rows, :min_cols]
                    data_subset = data_subset[:min_rows, :min_cols]

                # 写入数据（取最大值）
                output_slice[write_mask] = np.where(
                    output_slice[write_mask] == -9999,
                    data_subset[write_mask],
                    np.maximum(output_slice[write_mask], data_subset[write_mask])
                )

                successful_zones += 1

        if successful_zones == 0:
            print(f"    ⚠ 无分区数据成功拼接")
            return None

        return output

    def generate_all_mosaics(self):
        """生成所有拼接结果"""
        print("\n" + "=" * 50)
        print(f"开始生成{self.data_type}拼接结果")
        print("=" * 50)

        # 加载基础数据
        self.load_base_data()

        # 创建分区掩膜
        zone_masks = self.create_zone_masks()

        # 发现可用结果
        discovery = self.discover_simulation_results()
        rainfall_levels = discovery['rainfall_levels']
        hours = discovery['hours']

        print(f"\n处理计划: {len(rainfall_levels)}个降雨等级 × {len(hours)}小时")

        # 为每个降雨等级和小时生成拼接
        total_success = 0

        for rainfall_level in rainfall_levels:
            print(f"\n处理降雨等级: {rainfall_level}")

            for hour in hours:
                try:
                    # 拼接数据
                    mosaic_data = self.mosaic_hourly_data(rainfall_level, hour, zone_masks)

                    if mosaic_data is None:
                        continue

                    # 保存结果
                    if self.data_type == 'flood':
                        output_file = os.path.join(self.output_mosaic_dir,
                                                   f'flood_{rainfall_level}_hour_{hour:02d}.tif')
                    elif self.data_type == 'velocity':
                        output_file = os.path.join(self.output_mosaic_dir,
                                                   f'velocity_{rainfall_level}_hour_{hour:02d}.tif')
                    elif self.data_type == 'hazard':
                        output_file = os.path.join(self.output_mosaic_dir,
                                                   f'hazard_{rainfall_level}_hour_{hour:02d}.tif')
                    else:
                        continue

                    profile = self.dem_profile.copy()
                    profile.update(dtype='float32', nodata=-9999, count=1)

                    with rasterio.open(output_file, 'w', **profile) as dst:
                        dst.write(mosaic_data, 1)

                    # 计算统计信息
                    valid_data = mosaic_data[mosaic_data != -9999]
                    if len(valid_data) > 0:
                        if self.data_type == 'flood':
                            max_value = np.max(valid_data)
                            mean_value = np.mean(valid_data)
                            threshold_cells = np.sum(valid_data > 1)  # 1mm阈值
                            unit = "mm"
                        elif self.data_type == 'velocity':
                            max_value = np.max(valid_data)
                            mean_value = np.mean(valid_data)
                            threshold_cells = np.sum(valid_data > 0.01)  # 0.01m/s阈值
                            unit = "m/s"
                        elif self.data_type == 'hazard':
                            max_value = np.max(valid_data)
                            mean_value = np.mean(valid_data)
                            threshold_cells = np.sum(valid_data > 1.0)  # H>1.0视为危险
                            unit = ""

                        print(f"    ✓ 第{hour:02d}小时: 最大{max_value:.2f}{unit}, "
                              f"平均{mean_value:.2f}{unit}, 有效像元{threshold_cells}")

                        total_success += 1

                except Exception as e:
                    print(f"    ✗ 第{hour:02d}小时处理失败: {e}")
                    continue

        print(f"\n" + "=" * 50)
        print(f"{self.data_type}拼接完成! 成功生成 {total_success} 个文件")
        print(f"输出目录: {self.output_mosaic_dir}")
        print("=" * 50)

        return self.output_mosaic_dir


def run_complete_simulation():
    """运行完整的积水模拟（包含逐小时拼接）"""
    print("=" * 60)
    print("改进版SCS-CN洪水模拟系统 - 支持积水、流速和危险性指数三输出")
    print("=" * 60)

    # 初始化模拟器
    simulator = SCSCNFloodSimulator(
        dem_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/DEM/合并DEM.tif',
        shp_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/北京市边界_110000_Shapefile_(poi86.com)/river_canal_zone_beijing.shp',
        landcover_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/LUCC_bj/CLCD_v01_2021_albert_province/CLCD_v01_2021_albert_beijing_pro.tif',
        out_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/output_scs_cn_flood_improved_24',
        soil_type_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/rain_point/水文土壤数据/HYSOGs250mclip.tif',
        zone_id_field='zone_id',
        use_physical_route=True
    )

    # 配置参数
    simulator.williams_use_correction = True

    try:
        # 1. 加载数据
        simulator.load_data()
        simulator.clip_dem_by_zones()

        # 2. 批量模拟
        all_results = simulator.batch_simulate_all_zones(
            duration_hours=24,
            amc='II'
        )

        # 3. 保存结果（包含积水、流速和危险性指数）
        result_dirs = simulator.save_results()

        print("\n" + "=" * 60)
        print("✓ 所有任务完成!")
        print("=" * 60)
        print(f"模拟分区数: {len(all_results)}")
        print(f"结果保存位置: {simulator.out_path}")
        print(f"逐小时分区结果: {result_dirs['hourly_results']}")
        print(f"逐小时拼接结果: {result_dirs['hourly_mosaic_results']}")
        print(f"  包含积水深度(mm)、流速(m/s)和危险性指数三种数据类型")

        return simulator, all_results, result_dirs

    except Exception as e:
        print(f"✗ 模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_mosaic_generator():
    """主函数 - 同时拼接积水深度、流速和危险性指数"""
    print("积水模拟结果拼接模块")
    print("同时拼接积水深度、流速和危险性指数")
    print("=" * 50)

    # 配置路径
    dem_path = 'D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/DEM/合并DEM.tif'
    shp_path = 'D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/北京市边界_110000_Shapefile_(poi86.com)/river_canal_zone_beijing.shp'
    simulation_results_dir = 'D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/output_enhanced_scs_cn_flood_more_enhanced/simulation_results'
    output_dir = 'D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/output_enhanced_scs_cn_flood_more_enhanced'

    try:
        # 先拼接积水深度
        print("\n1. 拼接积水深度结果...")
        flood_mosaic_generator = FloodMosaicGenerator(
            dem_path=dem_path,
            shp_path=shp_path,
            simulation_results_dir=simulation_results_dir,
            output_dir=output_dir,
            data_type='flood'
        )

        flood_mosaic_dir = flood_mosaic_generator.generate_all_mosaics()

        # 再拼接流速结果
        print("\n2. 拼接流速结果...")
        velocity_mosaic_generator = FloodMosaicGenerator(
            dem_path=dem_path,
            shp_path=shp_path,
            simulation_results_dir=simulation_results_dir,
            output_dir=output_dir,
            data_type='velocity'
        )

        velocity_mosaic_dir = velocity_mosaic_generator.generate_all_mosaics()

        # 再拼接危险性指数结果
        print("\n3. 拼接危险性指数结果...")
        hazard_mosaic_generator = FloodMosaicGenerator(
            dem_path=dem_path,
            shp_path=shp_path,
            simulation_results_dir=simulation_results_dir,
            output_dir=output_dir,
            data_type='hazard'
        )

        hazard_mosaic_dir = hazard_mosaic_generator.generate_all_mosaics()

        print(f"\n✓ 所有处理完成!")
        print(f"积水深度结果保存在: {flood_mosaic_dir}")
        print(f"流速结果保存在: {velocity_mosaic_dir}")
        print(f"危险性指数结果保存在: {hazard_mosaic_dir}")

        return {
            'flood_mosaic_dir': flood_mosaic_dir,
            'velocity_mosaic_dir': velocity_mosaic_dir,
            'hazard_mosaic_dir': hazard_mosaic_dir
        }

    except Exception as e:
        print(f"\n✗ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_enhanced_simulation():
    """运行增强版洪水模拟"""
    print("=" * 70)
    print("基于中国暴雨分级的增强版洪水模拟系统")
    print("支持积水、流速和危险性指数三输出")
    print("=" * 70)

    # 初始化增强版模拟器
    simulator = SCSCNFloodSimulatorEnhanced(
        dem_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/DEM/合并DEM.tif',
        shp_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/北京市边界_110000_Shapefile_(poi86.com)/river_canal_zone_beijing.shp',
        landcover_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/LUCC_bj/CLCD_v01_2021_albert_province/CLCD_v01_2021_albert_beijing_pro.tif',
        out_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/output_enhanced_scs_cn_flood_more_enhanced',
        soil_type_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/rain_point/水文土壤数据/HYSOGs250mclip.tif',
        rainfall_region='north_china',  # 华北地区
        rainfall_pattern='auto'  # 自动选择雨型
    )

    # 配置参数
    simulator.williams_use_correction = True

    try:
        # 1. 加载数据
        simulator.load_data()
        simulator.clip_dem_by_zones()
        duration_hours = 24

        # 2. 批量模拟（重点模拟暴雨以上等级）
        print("\n🚀 开始增强版批量模拟...")
        all_results = simulator.batch_simulate_enhanced(
            rainfall_levels=['小雨', '中雨', '大雨', '暴雨', '大暴雨', '特大暴雨'],
            duration_hours=duration_hours
        )

        print("\n" + "=" * 70)
        print("🎉 所有任务完成!")
        print("=" * 70)

        print(f"模拟降雨等级: 小雨、中雨、大雨、暴雨、大暴雨、特大暴雨")
        print(f"地区特征: {simulator.rainfall_region}")
        print(f"结果保存位置: {simulator.out_path}")
        print(f"输出数据类型: 积水深度(mm)、流速(m/s)和危险性指数")

        # 输出汇总统计
        print_summary_statistics(all_results)

        return simulator, all_results

    except Exception as e:
        print(f"✗ 模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def print_summary_statistics(all_results):
    """输出汇总统计信息"""
    print("\n📊 模拟结果汇总统计:")
    print("-" * 50)

    for rainfall_level in ['小雨', '中雨', '大雨', '暴雨', '大暴雨', '特大暴雨']:
        max_depths = []
        max_velocities = []
        total_volumes = []
        max_hazards = []

        for zone_id, zone_results in all_results.items():
            if rainfall_level in zone_results:
                result = zone_results[rainfall_level]
                max_depths.append(result['max_flood_depth'])
                max_velocities.append(result['max_velocity'])
                total_volumes.append(result['total_flood_volume'])
                if 'max_hazard' in result:
                    max_hazards.append(result['max_hazard'])

        if max_depths:
            print(f"{rainfall_level}:")
            print(f"  最大积水深度: {np.max(max_depths):.1f}mm (平均{np.mean(max_depths):.1f}mm)")
            print(f"  最大流速: {np.max(max_velocities):.2f}m/s (平均{np.mean(max_velocities):.2f}m/s)")
            print(f"  总积水量: {np.sum(total_volumes):.0f} m³")
            if max_hazards:
                print(f"  最大危险性指数: {np.max(max_hazards):.2f} (平均{np.mean(max_hazards):.2f})")
            print(f"  影响分区数: {len(max_depths)}")
            print()


def rainfall_simulation():
    # 初始化降雨生成器（华北地区）
    rainfall_gen = AdvancedRainfallPatternGenerator(region='north_china')

    # 定义降雨等级
    rainfall_levels = ['小雨', '中雨', '大雨', '暴雨', '大暴雨', '特大暴雨']
    duration_hours = 24

    print("=" * 70)
    print("华北地区不同降雨等级24小时降雨时间序列（单位：mm/h）")
    print("=" * 70)

    # 存储所有降雨序列
    all_rainfall_series = {}

    for level in rainfall_levels:
        # 生成降雨序列
        rainfall_data = rainfall_gen.generate_comprehensive_rainfall(
            rainfall_level=level,
            duration_hours=duration_hours,
            pattern_type='auto'
        )

        # 提取数据
        hourly_rainfall = rainfall_data['rainfall_per_hour']
        total_rainfall = rainfall_data['total_rainfall']
        stats = rainfall_data['statistics']

        # 存储到字典
        all_rainfall_series[level] = {
            'hourly': hourly_rainfall,
            'total': total_rainfall,
            'stats': stats
        }

        # 打印该等级降雨序列
        print(f"\n{level} (总雨量: {total_rainfall:.1f}mm)")
        print("-" * 60)

        # 按小时输出，每行显示6小时
        for i in range(0, duration_hours, 6):
            line = f"第{i + 1:2d}-{min(i + 6, duration_hours):2d}小时: "
            for h in range(i, min(i + 6, duration_hours)):
                line += f"{hourly_rainfall[h]:5.1f} "
            print(line)

        # 输出统计信息
        print(f"峰值: {stats['peak_intensity']:.1f}mm/h (第{int(stats['peak_time']):2d}小时)")
        print(f"平均: {stats['mean_hourly']:.1f}mm/h, 历时: {stats['rainfall_duration']:.0f}小时")

    # 输出为可直接使用的Python数组
    print("\n" + "=" * 70)
    print("Python数组格式的降雨时间序列")
    print("=" * 70)

    for level in rainfall_levels:
        series = all_rainfall_series[level]['hourly']
        total = all_rainfall_series[level]['total']

        print(f"\n# {level} (总雨量: {total:.1f}mm)")
        print(f"rainfall_{level} = [")

        # 每行显示6个数值
        for i in range(0, duration_hours, 6):
            line = "    "
            for h in range(i, min(i + 6, duration_hours)):
                line += f"{series[h]:6.2f}, "
            print(line)

        print("]")

    # 输出JSON格式
    print("\n" + "=" * 70)
    print("JSON格式的降雨时间序列")
    print("=" * 70)

    rainfall_json = {}
    for level in rainfall_levels:
        series = all_rainfall_series[level]['hourly'].tolist()
        stats = all_rainfall_series[level]['stats']

        rainfall_json[level] = {
            "hourly_rainfall_mm": series,
            "total_rainfall_mm": float(all_rainfall_series[level]['total']),
            "peak_intensity_mm_h": float(stats['peak_intensity']),
            "peak_hour": float(stats['peak_time']),
            "average_intensity_mm_h": float(stats['mean_hourly']),
            "duration_hours": int(stats['rainfall_duration'])
        }

    print(json.dumps(rainfall_json, indent=2, ensure_ascii=False))

    # 输出CSV格式
    print("\n" + "=" * 70)
    print("CSV格式的降雨时间序列")
    print("=" * 70)

    # 表头
    header = "小时"
    for level in rainfall_levels:
        header += f",{level}(mm)"
    print(header)

    # 数据行
    for hour in range(duration_hours):
        row = f"第{hour + 1:2d}小时"
        for level in rainfall_levels:
            rainfall = all_rainfall_series[level]['hourly'][hour]
            row += f",{rainfall:.2f}"
        print(row)

    # 输出汇总统计
    print("\n" + "=" * 70)
    print("降雨特征汇总统计")
    print("=" * 70)

    print(f"{'降雨等级':<8} {'总雨量(mm)':>12} {'峰值(mm/h)':>12} {'峰时(h)':>10} {'平均(mm/h)':>12} {'历时(h)':>8}")
    print("-" * 70)

    for level in rainfall_levels:
        stats = all_rainfall_series[level]['stats']
        print(f"{level:<8} {all_rainfall_series[level]['total']:12.1f} "
              f"{stats['peak_intensity']:12.1f} {stats['peak_time']:10.1f} "
              f"{stats['mean_hourly']:12.2f} {stats['rainfall_duration']:8.0f}")

    # 可视化对比
    print("\n" + "=" * 70)
    print("可视化对比图（字符形式）")
    print("=" * 70)

    import matplotlib.pyplot as plt

    # 创建可视化图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, level in enumerate(rainfall_levels):
        if idx >= len(axes):
            break

        ax = axes[idx]
        series = all_rainfall_series[level]['hourly']

        # 绘制柱状图
        bars = ax.bar(np.arange(duration_hours), series,
                      alpha=0.7,
                      color=rainfall_gen.rainfall_levels_china[level]['color'])

        # 标记峰值
        peak_idx = int(all_rainfall_series[level]['stats']['peak_time'] - 0.5)
        if 0 <= peak_idx < len(bars):
            bars[peak_idx].set_alpha(1.0)
            bars[peak_idx].set_edgecolor('black')

        ax.set_title(f"{level}\n({all_rainfall_series[level]['total']:.0f}mm)", fontsize=11)
        ax.set_xlabel("时间(小时)")
        ax.set_ylabel("降雨量(mm/h)")
        ax.grid(True, alpha=0.3, linestyle='--')

        # 设置y轴范围，使对比更明显
        max_intensity = all_rainfall_series[level]['stats']['peak_intensity']
        ax.set_ylim(0, max_intensity * 1.2)

    plt.suptitle("华北地区不同降雨等级24小时降雨时间序列对比", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rainfall_time_series_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✓ 降雨时间序列已生成并保存为 'rainfall_time_series_comparison.png'")

    # 输出为NumPy数组文件（可选）
    print("\n" + "=" * 70)
    print("NumPy数组文件输出")
    print("=" * 70)

    # 创建一个包含所有序列的字典
    rainfall_dict = {}
    for level in rainfall_levels:
        rainfall_dict[level] = all_rainfall_series[level]['hourly']

    # 保存为.npz文件
    np.savez('rainfall_time_series.npz', **rainfall_dict)
    print("✓ 降雨时间序列已保存为 'rainfall_time_series.npz'")

    # 示例：如何加载使用
    print("\n示例代码 - 加载使用降雨序列:")
    print("```python")
    print("import numpy as np")
    print("")
    print("# 加载降雨序列")
    print("rainfall_data = np.load('rainfall_time_series.npz')")
    print("")
    print("# 获取暴雨序列")
    print("暴雨序列 = rainfall_data['暴雨']")
    print("print(f'暴雨序列: {暴雨序列}')")
    print("print(f'总雨量: {np.sum(暴雨序列):.1f}mm')")
    print("```")


# 主程序入口
if __name__ == '__main__':
    print("=" * 80)
    print("SCS-CN洪水模拟系统 - 完整版")
    print("支持积水深度、流速和危险性指数三输出")
    print("=" * 80)

    print("\n请选择运行模式:")
    print("1. 增强版洪水模拟（推荐）")
    print("2. 拼接模拟结果（积水深度、流速和危险性指数）")
    print("3. 生成降雨时间序列")
    print("4. 运行完整流程（模拟+拼接）")
    print("5. 参数单因子敏感性分析（主峰位置r + 主峰增强p）")

    try:
        choice = input("\n请输入选择 (1-5): ").strip()

        if choice == "1":
            print("\n运行增强版洪水模拟...")
            simulator, results = run_enhanced_simulation()

        elif choice == "2":
            print("\n运行拼接模块...")
            mosaic_results = run_mosaic_generator()

        elif choice == "3":
            print("\n生成降雨时间序列...")
            rainfall_simulation()

        elif choice == "4":
            print("\n运行完整流程...")
            # 先运行模拟
            print("\n=== 第一阶段：洪水模拟 ===")
            simulator, results = run_enhanced_simulation()

            if simulator is not None and results is not None:
                # 再运行拼接
                print("\n=== 第二阶段：结果拼接 ===")
                mosaic_results = run_mosaic_generator()

                print("\n" + "=" * 80)
                print("🎉 完整流程执行完毕!")
                print("=" * 80)
                print(f"模拟分区数: {len(results)}")
                print(f"输出数据类型: 积水深度(mm) + 流速(m/s) + 危险性指数")
                print(f"拼接结果目录:")
                if mosaic_results:
                    print(f"  - 积水深度: {mosaic_results.get('flood_mosaic_dir', 'N/A')}")
                    print(f"  - 流速: {mosaic_results.get('velocity_mosaic_dir', 'N/A')}")
                    print(f"  - 危险性指数: {mosaic_results.get('hazard_mosaic_dir', 'N/A')}")

        elif choice == "5":
            print("\n运行参数单因子敏感性分析...")
            simulator = SCSCNFloodSimulatorEnhanced(
                dem_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/DEM/合并DEM.tif',
                shp_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/北京市边界_110000_Shapefile_(poi86.com)/river_canal_zone_beijing.shp',
                landcover_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/LUCC_bj/CLCD_v01_2021_albert_province/CLCD_v01_2021_albert_beijing_pro.tif',
                out_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/output_enhanced_scs_cn_flood_more_enhanced',
                soil_type_path='D:/BaiduSyncdisk/学习资料/ysjs/开题/论文数据/rain_point/水文土壤数据/HYSOGs250mclip.tif',
                rainfall_region='north_china',
                rainfall_pattern='auto'
            )
            simulator.williams_use_correction = True
            simulator.load_data()
            simulator.clip_dem_by_zones()
            sens_results = simulator.run_sensitivity_analysis(zone_id=42, duration_hours=24)

        else:
            print("无效选择，请输入1-5之间的数字")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback

        traceback.print_exc()
