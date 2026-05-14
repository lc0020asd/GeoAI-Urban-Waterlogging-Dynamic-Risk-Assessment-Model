# ==========================================================
# 时空Transformer模型 - 分区域预测代码（三月全量）
# 核心适配：1.直读补全数据 2.分区域加载模型 3.全维度时空预测
# 预测范围：不同年龄性别组合 + 所有网格 + 三月全日期 + 24小时
# ==========================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob
import os
import warnings
import gc
import time
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler, LabelEncoder
import math

# 全局配置 - 与训练代码保持一致
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() // 2)
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() // 2)
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count() // 2)
pd.set_option('mode.chained_assignment', None)
pd.set_option('compute.use_bottleneck', True)
pd.set_option('compute.use_numexpr', True)
warnings.filterwarnings('ignore')

# 内存优化配置 - 与训练代码完全一致
MEMORY_OPTIMIZATION_CONFIG = {
    'use_mixed_precision': False,
    'gradient_accumulation_steps': 1,
    'use_gradient_checkpointing': False,
    'clear_cache_frequency': 50,
    'reduce_workers': True,
    'pin_memory': False,
}

# ============ 1. 复用训练代码的核心组件（直接复制，保证一致性） ============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DecoupledSpatioTemporalTransformer(nn.Module):
    def __init__(self, temporal_input_dim, spatial_input_dim,
                 d_model=128, nhead=8, num_layers=6,
                 dim_feedforward=512, dropout=0.1,
                 use_gradient_checkpointing=False):
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.temporal_embedding = nn.Linear(temporal_input_dim, d_model)
        self.spatial_embedding = nn.Linear(spatial_input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fusion_layer = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.ucnt_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, 1)
        )

        self.usum_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, 1)
        )

    def forward(self, temporal_seq, spatial_features):
        b, seq_len, _ = temporal_seq.shape
        temp = self.temporal_embedding(temporal_seq)
        temp = self.pos_encoding(temp.transpose(0, 1)).transpose(0, 1)
        temp = self.dropout(temp)
        spat = self.spatial_embedding(spatial_features).unsqueeze(1).expand(-1, seq_len, -1)
        fused = temp + spat

        if self.use_gradient_checkpointing and self.training:
            encoded = torch.utils.checkpoint.checkpoint(
                self.transformer_encoder, fused, use_reentrant=False)
        else:
            encoded = self.transformer_encoder(fused)

        spat_query = spat[:, 0:1, :]
        attended, _ = self.fusion_layer(spat_query, encoded, encoded)
        output = attended.squeeze(1) + spat[:, 0, :]

        ucnt_pred = self.ucnt_head(output)
        usum_pred = self.usum_head(output)
        return ucnt_pred, usum_pred

# ============ 2. 预测专用数据处理器（直读补全数据+仅预处理，无补全） ============
class PredictDataProcessor:
    """预测专用处理器：直读_imputed.csv + 轻量化预处理 + 特征对齐训练集"""
    def __init__(self, excluded_from_encoding=None, excluded_from_features=None):
        # 与训练代码完全一致的排除列，保证特征对齐
        self.excluded_from_encoding = excluded_from_encoding or ['final_grid_id', 'city', 'geohash7']
        self.excluded_from_features = excluded_from_features or ['final_grid_id', 'geohash7', 'city', 'ucnt', 'usum', 'date']
        # 标准化器/编码器：从训练数据继承（此处重新初始化，与训练一致）
        self.temporal_scaler = RobustScaler(quantile_range=(5, 95))
        self.spatial_scaler = RobustScaler(quantile_range=(5, 95))
        self.ucnt_scaler = RobustScaler(quantile_range=(5, 95))
        self.usum_scaler = RobustScaler(quantile_range=(5, 95))
        self.categorical_encoders = {}
        self.temporal_features = None  # 保存训练集一致的时序特征列
        self.spatial_features = None   # 保存训练集一致的空间特征列

    def create_additional_features(self, data):
        """与训练代码完全一致的特征工程，保证特征维度匹配"""
        if 'hour' in data.columns:
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['is_morning'] = ((data['hour'] >= 6) & (data['hour'] < 12)).astype(int)
            data['is_afternoon'] = ((data['hour'] >= 12) & (data['hour'] < 18)).astype(int)
            data['is_evening'] = ((data['hour'] >= 18) & (data['hour'] < 24)).astype(int)
            data['is_night'] = ((data['hour'] >= 0) & (data['hour'] < 6)).astype(int)

        if 'day_of_week' in data.columns:
            data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            data['is_weekend'] = (data['day_of_week'].isin([5, 6])).astype(int)

        if 'month' in data.columns:
            data['season'] = pd.cut(data['month'], bins=[0, 3, 6, 9, 12], labels=[0,1,2,3]).astype(int)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        return data

    def _ensure_nonnegative_integer(self, data, column_names):
        """与训练代码一致的非负整数约束"""
        for col in column_names:
            if col in data.columns:
                data[col] = np.maximum(data[col], 0)
                data[col] = np.round(data[col]).astype(int)
        return data

    def _build_sequences_fast(self, merged_data, seq_length=24, stride=1):
        """与训练代码一致的序列构建逻辑，保证输入格式匹配"""
        grouped = merged_data.groupby('final_grid_id')
        total_sequences = 0
        valid_grids = []
        for grid_id, group in grouped:
            n_rows = len(group)
            if n_rows >= seq_length + 1:
                n_seqs = (n_rows - seq_length) // stride
                if n_seqs > 0:
                    total_sequences += n_seqs
                    valid_grids.append((grid_id, group, n_seqs))

        n_temporal = len(self.temporal_features)
        n_spatial = len(self.spatial_features)
        sequences = np.empty((total_sequences, seq_length, n_temporal), dtype=np.float32)
        spatial_vectors = np.empty((total_sequences, n_spatial), dtype=np.float32)
        # 保存序列对应的时空标签（用于后续结果映射）
        seq_labels = []
        current_idx = 0

        for grid_id, group, n_seqs in tqdm(valid_grids, desc="构建预测序列"):
            group = group.sort_values(['date', 'hour'])
            temp_vals = group[self.temporal_features].values.astype(np.float32)
            spat_vals = group[self.spatial_features].iloc[0].values.astype(np.float32)
            if not (np.all(np.isfinite(temp_vals)) and np.all(np.isfinite(spat_vals))):
                continue
            n_rows = len(temp_vals)
            indices = np.arange(0, n_rows - seq_length, stride)
            for i, start_idx in enumerate(indices):
                end_idx = start_idx + seq_length
                sequences[current_idx] = temp_vals[start_idx:end_idx]
                spatial_vectors[current_idx] = spat_vals
                # 保存当前序列的结束标签（网格、日期、小时、年龄、性别）
                seq_labels.append({
                    'final_grid_id': grid_id,
                    'predict_date': group.iloc[end_idx]['date'],
                    'predict_hour': group.iloc[end_idx]['hour'],
                    'age': group.iloc[end_idx]['age'],
                    'gender': group.iloc[end_idx]['gender']
                })
                current_idx += 1

        sequences = sequences[:current_idx]
        spatial_vectors = spatial_vectors[:current_idx]
        # 标准化（与训练一致的原地操作）
        N, T, F = sequences.shape
        sequences_2d = sequences.reshape(-1, F)
        sequences_2d[:] = self.temporal_scaler.fit_transform(sequences_2d)
        spatial_vectors[:] = self.spatial_scaler.fit_transform(spatial_vectors)

        # ==============================================
        # 【核心新增】拟合ucnt/usum标准化器，解决未拟合报错
        # 用补全数据的真实ucnt/usum值拟合，和训练逻辑完全一致
        # ==============================================
        self.ucnt_scaler.fit(merged_data['ucnt'].values.reshape(-1, 1))
        self.usum_scaler.fit(merged_data['usum'].values.reshape(-1, 1))

        del sequences_2d
        gc.collect()

        return sequences, spatial_vectors, pd.DataFrame(seq_labels)

    def load_imputed_data_and_preprocess(self, imputed_file_path, result_file_path, seq_length=24):
        """
        核心：直读补全后的数据文件，完成轻量化预处理
        :param imputed_file_path: 补全后的_imputed.csv文件路径
        :param result_file_path: 空间特征文件路径（与训练一致）
        :param seq_length: 序列长度（与训练一致，默认48）
        :return: 模型输入序列+空间特征+序列标签+标准化器
        """
        print("="*50)
        print(f"直读补全数据：{os.path.basename(imputed_file_path)}")
        # 1. 加载补全后数据（跳过补全步骤，直接使用）
        all_data = pd.read_csv(imputed_file_path, encoding='utf-8-sig')
        print(f"补全数据加载完成，行数：{len(all_data):,}")

        # 2. 基础预处理（与训练一致）
        all_data = self._ensure_nonnegative_integer(all_data, ['ucnt', 'usum'])
        all_data['date'] = pd.to_datetime(all_data['date'], format='%Y%m%d', errors='coerce')
        all_data['day_of_week'] = all_data['date'].dt.dayofweek
        all_data['month'] = all_data['date'].dt.month
        # 特征工程（与训练完全一致）
        all_data = self.create_additional_features(all_data)

        # 3. 合并空间特征（与训练一致）
        result_df = pd.read_csv(result_file_path)
        merged = all_data.merge(result_df, left_on='final_grid_id', right_on='geohash7', how='inner')
        print(f"空间特征合并完成，行数：{len(merged):,}")

        # 4. 分类特征编码（与训练一致）
        for col in merged.columns:
            if merged[col].dtype == 'object' and col not in self.excluded_from_encoding:
                if col not in self.categorical_encoders:
                    self.categorical_encoders[col] = LabelEncoder()
                    merged[col] = self.categorical_encoders[col].fit_transform(merged[col].astype(str))
                else:
                    merged[col] = self.categorical_encoders[col].transform(merged[col].astype(str))

        # 5. 划分特征（与训练一致，固定特征列）
        self.temporal_features = [c for c in all_data.columns if c not in self.excluded_from_features]
        self.spatial_features = [c for c in result_df.columns if c not in self.excluded_from_features]
        print(f"特征对齐完成 | 时序特征：{len(self.temporal_features)} | 空间特征：{len(self.spatial_features)}")

        # 6. 构建预测序列+保存标签（关键：保留预测的时空/人口属性）
        sequences, spatial_data, seq_labels = self._build_sequences_fast(merged, seq_length, stride=1)
        print(f"预测序列构建完成：{len(sequences)} 个预测样本")

        return sequences, spatial_data, seq_labels, self.ucnt_scaler, self.usum_scaler

# ============ 3. 核心预测函数（分区域加载模型+批量预测+反标准化） ============
class SpatioTemporalPredictor:
    """分区域预测器：自动匹配区域模型+批量预测+结果后处理"""
    def __init__(self, model_weight_dir, seq_length=48):
        self.model_weight_dir = model_weight_dir  # 区域模型权重目录
        self.seq_length = seq_length              # 序列长度（与训练一致）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"预测设备：{self.device} | 模型权重目录：{model_weight_dir}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 修正：获取GPU名称+剩余/总显存（适配新版PyTorch）
            gpu_name = torch.cuda.get_device_name(0)
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_mem_gb = free_mem / 1024 ** 3
            total_mem_gb = total_mem / 1024 ** 3
            print(f"GPU信息：{gpu_name} | 总显存：{total_mem_gb:.2f}GB | 剩余显存：{free_mem_gb:.2f}GB")

    def load_zone_model(self, model, zone_id):
        """加载指定区域的模型权重"""
        weight_file = os.path.join(self.model_weight_dir, f'zone_{zone_id:03d}.pth')
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"区域{zone_id}的模型权重不存在：{weight_file}")
        model.load_state_dict(torch.load(weight_file, map_location=self.device))
        model = model.to(self.device)
        model.eval()  # 预测模式：关闭dropout/batchnorm
        print(f"✅ 加载区域 {zone_id:03d} 模型权重完成")
        return model

    def batch_predict(self, model, sequences, spatial_data, batch_size=512):
        """批量预测：避免显存溢出，返回预测值"""
        model.eval()
        all_ucnt_pred, all_usum_pred = [], []
        # 转换为tensor（适配模型输入）
        sequences = torch.FloatTensor(sequences).to(self.device)
        spatial_data = torch.FloatTensor(spatial_data).to(self.device)
        # 分批次预测
        with torch.no_grad():  # 关闭梯度计算，节省显存
            for start in tqdm(range(0, len(sequences), batch_size), desc="批量预测"):
                end = min(start + batch_size, len(sequences))
                seq_batch = sequences[start:end]
                spat_batch = spatial_data[start:end]
                ucnt_batch, usum_batch = model(seq_batch, spat_batch)
                all_ucnt_pred.append(ucnt_batch.cpu().numpy())
                all_usum_pred.append(usum_batch.cpu().numpy())
                # 定期清理显存
                if (start // batch_size) % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # 合并批次
        ucnt_pred = np.concatenate(all_ucnt_pred, axis=0).flatten()
        usum_pred = np.concatenate(all_usum_pred, axis=0).flatten()
        return ucnt_pred, usum_pred

    def post_process_pred(self, ucnt_pred, usum_pred, ucnt_scaler, usum_scaler):
        """后处理：反标准化+非负整数约束（与训练代码一致的业务规则）"""
        # 反标准化：恢复原始数值范围
        ucnt_pred = ucnt_scaler.inverse_transform(ucnt_pred.reshape(-1, 1)).flatten()
        usum_pred = usum_scaler.inverse_transform(usum_pred.reshape(-1, 1)).flatten()
        # 业务约束：非负整数（人数/时长不能为负，且为整数）
        ucnt_pred = np.maximum(ucnt_pred, 0).round().astype(int)
        usum_pred = np.maximum(usum_pred, 0).round().astype(int)
        print(f"预测结果后处理完成 | UCNT范围：[{ucnt_pred.min()}, {ucnt_pred.max()}] | USUM范围：[{usum_pred.min()}, {usum_pred.max()}]")
        return ucnt_pred, usum_pred

    def predict_one_zone(self, zone_id, imputed_file, result_file, seq_length=48, batch_size=512):
        """
        单区域完整预测流程
        :param zone_id: 区域编号（与训练的file_counter一致）
        :param imputed_file: 该区域的_imputed.csv文件路径
        :param result_file: 空间特征文件路径
        :return: 包含完整标签的预测结果DataFrame
        """
        try:
            # 1. 初始化预测处理器
            processor = PredictDataProcessor(
                excluded_from_encoding=['final_grid_id', 'geohash7', 'area_sqm', 'perim_m', 'width_m', 'height_m', 'city'],
                excluded_from_features=['final_grid_id', 'geohash7', 'ucnt', 'usum', 'area_sqm', 'perim_m', 'width_m', 'height_m', 'city', 'date', 'OID_', 'grid_id','min_lat', 'max_lat', 'min_lon', 'max_lon']
            )
            # 2. 加载补全数据并预处理
            sequences, spatial_data, seq_labels, ucnt_scaler, usum_scaler = processor.load_imputed_data_and_preprocess(
                imputed_file, result_file, seq_length
            )
            # 3. 初始化模型（与训练代码完全一致的参数，必须匹配！）
            model = DecoupledSpatioTemporalTransformer(
                temporal_input_dim=len(processor.temporal_features),
                spatial_input_dim=len(processor.spatial_features),
                d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1,
                use_gradient_checkpointing=MEMORY_OPTIMIZATION_CONFIG['use_gradient_checkpointing']
            )
            # 4. 加载该区域模型权重
            model = self.load_zone_model(model, zone_id)
            # 5. 批量预测
            ucnt_pred, usum_pred = self.batch_predict(model, sequences, spatial_data, batch_size)
            # 6. 结果后处理
            ucnt_pred, usum_pred = self.post_process_pred(ucnt_pred, usum_pred, ucnt_scaler, usum_scaler)
            # 7. 合并标签和预测结果
            pred_result = seq_labels.copy()
            pred_result['ucnt_pred'] = ucnt_pred
            pred_result['usum_pred'] = usum_pred
            # 补充三月标识+区域编号
            pred_result['month'] = 3
            pred_result['zone_id'] = zone_id
            # 格式化日期（统一为%Y%m%d，与原始数据一致）
            pred_result['predict_date'] = pred_result['predict_date'].dt.strftime('%Y%m%d')
            print(f"区域 {zone_id} 预测完成，共生成 {len(pred_result):,} 条预测结果")

            # 释放内存
            del processor, model, sequences, spatial_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return pred_result
        except Exception as e:
            print(f"❌ 区域 {zone_id} 预测失败：{e}")
            import traceback
            traceback.print_exc()
            return None

# ============ 4. 主预测函数（遍历所有区域+合并结果+保存） ============
def main_march_prediction():
    """
    三月全量分区域预测主入口
    配置说明：修改以下路径/参数，与你的训练环境保持一致
    """
    # ===================== 核心配置（请根据你的环境修改）=====================
    SEQ_LENGTH = 48  # 与训练一致（main_zone_by_zone中设置的48）
    BATCH_SIZE = 256  # 与训练一致（main_zone_by_zone中设置的256）
    ZONE_START = 1    # 区域起始编号（与训练一致）
    ZONE_END = 672    # 区域结束编号（与训练一致）
    MODEL_WEIGHT_DIR = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\weights"  # 训练的权重目录
    IMPUTED_DATA_DIR = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令"  # 补全数据(_imputed.csv)所在目录
    RESULT_FILE_PATH = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\fishnet\geohash7\shp\result_core_area.csv"  # 空间特征文件（与训练一致）
    PRED_SAVE_DIR = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\march_pred_result"  # 预测结果保存目录
    # ========================================================================

    # 创建结果保存目录
    os.makedirs(PRED_SAVE_DIR, exist_ok=True)
    # 初始化预测器
    predictor = SpatioTemporalPredictor(MODEL_WEIGHT_DIR, SEQ_LENGTH)
    # 保存所有区域的预测结果（可选：按区域单独保存+合并全量）
    all_zone_pred = []

    # 遍历所有区域
    for zone_id in range(ZONE_START, ZONE_END + 1):
        # 拼接补全数据文件路径（自动匹配_imputed后缀）
        imputed_file_name = f"aggregated_by_geohash7_blocks_core_{zone_id}_imputed.csv"
        imputed_file_path = os.path.join(IMPUTED_DATA_DIR, imputed_file_name)
        # 检查补全文件是否存在
        if not os.path.exists(imputed_file_path):
            print(f"⚠️  区域 {zone_id} 补全文件不存在，跳过：{imputed_file_name}")
            continue
        # 单区域预测
        zone_pred = predictor.predict_one_zone(zone_id, imputed_file_path, RESULT_FILE_PATH, SEQ_LENGTH, BATCH_SIZE)
        if zone_pred is not None and len(zone_pred) > 0:
            # 按区域保存预测结果
            zone_save_path = os.path.join(PRED_SAVE_DIR, f"march_pred_zone_{zone_id:03d}.csv")
            zone_pred.to_csv(zone_save_path, index=False, encoding='utf-8-sig')
            print(f"✅ 区域 {zone_id} 预测结果已保存：{zone_save_path}\n")
            # 加入全量结果
            all_zone_pred.append(zone_pred)
        # 间隔休息，避免系统资源占用过高
        time.sleep(1)

    # 合并所有区域的预测结果并保存全量文件
    if len(all_zone_pred) > 0:
        all_march_pred = pd.concat(all_zone_pred, ignore_index=True)
        all_save_path = os.path.join(PRED_SAVE_DIR, "march_pred_all_zones.csv")
        all_march_pred.to_csv(all_save_path, index=False, encoding='utf-8-sig')
        print("="*60)
        print(f"🎉 三月全量预测完成！")
        print(f"📊 有效预测区域数：{len(all_zone_pred)}/{ZONE_END-ZONE_START+1}")
        print(f"📈 总预测结果数：{len(all_march_pred):,}")
        print(f"💾 全量结果保存路径：{all_save_path}")
        print("="*60)
    else:
        print("❌ 无有效区域预测结果")

# ============ 5. 运行入口 ============
if __name__ == "__main__":
    """
    运行说明：
    1. 先确保所有区域的_imputed.csv文件已生成（训练代码运行后自动生成）
    2. 确保MODEL_WEIGHT_DIR目录下有所有区域的zone_xxx.pth权重文件
    3. 修改main_march_prediction中的【核心配置】为你的实际路径/参数
    4. 直接运行该脚本，自动遍历所有区域完成三月全量预测
    """
    main_march_prediction()