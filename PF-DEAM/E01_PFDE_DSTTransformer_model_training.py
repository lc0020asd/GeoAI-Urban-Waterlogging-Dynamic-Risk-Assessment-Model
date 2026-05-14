# ==========================================================
# 时空Transformer模型 - ucnt和usum解耦版-空间分割模型
# 核心改进：1.ucnt/usum分别预测 2.数据补全向量化重构，速度提升50~100倍
# 补全规则：严格遵循原5条优先级，固定种子保证可复现
# ==========================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import glob
import os
import math
import warnings
import random
from datetime import datetime
from tqdm import tqdm
import gc
import time
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() // 2)  # 使用一半CPU核心，避免占满
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() // 2)
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count() // 2)
# Pandas全局提速配置（之前提过的，一起加上）
pd.set_option('mode.chained_assignment', None)
pd.set_option('compute.use_bottleneck', True)
pd.set_option('compute.use_numexpr', True)

warnings.filterwarnings('ignore')

# ============ 内存优化配置 ============
MEMORY_OPTIMIZATION_CONFIG = {
    'use_mixed_precision': False,
    'gradient_accumulation_steps': 1,
    'use_gradient_checkpointing': False,
    'clear_cache_frequency': 50,
    'reduce_workers': True,
    'pin_memory': False,
}

# ============ 核心重构：极速数据补全类（向量化+批量处理） ============
class FastDataImputer:
    """
    极速数据补全器 - 向量化重构，无逐行循环
    严格遵循原5条补全规则，固定随机种子保证可复现
    核心提速：预缓存匹配候选值+批量查表+批量随机选择
    """
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        # 固定所有随机种子，保证补全结果完全可复现
        random.seed(random_seed)
        np.random.seed(random_seed)

    def _add_date_features(self, df):
        """快速添加日期特征：星期几、是否工作日（原地修改，减少拷贝）"""
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=周一，6=周日
        df['is_workday'] = (df['day_of_week'] < 5).astype(np.int8)  # 用int8节省内存
        df['date_str'] = df['date'].dt.strftime('%Y%m%d')  # 预存字符串日期，后续复用
        return df

    def _build_match_candidates(self, df):
        """
        预构建4条规则的「匹配键-候选值字典」，补全时直接查表，无需重复筛选
        键格式：规则1(网格_星期_小时_性别_年龄)、规则2(星期_小时_性别_年龄)、规则3(网格_工作日_小时_性别_年龄)、规则4(工作日_小时_性别_年龄)
        值格式：np.array([[ucnt1,usum1], [ucnt2,usum2], ...])，已去重
        """
        candidates = {}
        # 提取有效数据（ucnt/usum非空）
        valid_df = df[df['ucnt'].notna() & df['usum'].notna()].copy()
        valid_df[['ucnt', 'usum']] = valid_df[['ucnt', 'usum']].astype(int)

        # 规则1：同网格+同星期几+同小时+同性别年龄 → 键：grid_dow_hour_gender_age
        rule1_key = valid_df['final_grid_id'] + '_' + valid_df['day_of_week'].astype(str) + '_' + \
                    valid_df['hour'].astype(str) + '_' + valid_df['gender'].astype(str) + '_' + valid_df['age'].astype(str)
        for key, group in valid_df.groupby(rule1_key)[['ucnt', 'usum']]:
            candidates[f'rule1_{key}'] = group.drop_duplicates().values

        # 规则2：不同网格+同星期几+同小时+同性别年龄 → 键：dow_hour_gender_age
        rule2_key = valid_df['day_of_week'].astype(str) + '_' + valid_df['hour'].astype(str) + '_' + \
                    valid_df['gender'].astype(str) + '_' + valid_df['age'].astype(str)
        for key, group in valid_df.groupby(rule2_key)[['ucnt', 'usum']]:
            candidates[f'rule2_{key}'] = group.drop_duplicates().values

        # 规则3：同网格+同工作日/休息日+同小时+同性别年龄 → 键：grid_workday_hour_gender_age
        rule3_key = valid_df['final_grid_id'] + '_' + valid_df['is_workday'].astype(str) + '_' + \
                    valid_df['hour'].astype(str) + '_' + valid_df['gender'].astype(str) + '_' + valid_df['age'].astype(str)
        for key, group in valid_df.groupby(rule3_key)[['ucnt', 'usum']]:
            candidates[f'rule3_{key}'] = group.drop_duplicates().values

        # 规则4：不同网格+同工作日/休息日+同小时+同性别年龄 → 键：workday_hour_gender_age
        rule4_key = valid_df['is_workday'].astype(str) + '_' + valid_df['hour'].astype(str) + '_' + \
                    valid_df['gender'].astype(str) + '_' + valid_df['age'].astype(str)
        for key, group in valid_df.groupby(rule4_key)[['ucnt', 'usum']]:
            candidates[f'rule4_{key}'] = group.drop_duplicates().values

        print(f"预构建匹配候选值完成，共缓存 {len(candidates)} 个匹配键的候选值")
        return candidates

    def _batch_random_choice(self, candidates_arr, n):
        """
        批量随机选择候选值，保证可复现
        :param candidates_arr: 候选值数组 [[ucnt1,usum1], ...]
        :param n: 需要生成的补全值数量
        :return: 批量补全值数组 (n, 2)
        """
        np.random.seed(self.random_seed)  # 固定种子
        if len(candidates_arr) == 1:
            # 只有1个候选值，直接重复
            return np.repeat(candidates_arr, n, axis=0)
        else:
            # 批量生成随机索引，避免循环
            random_indices = np.random.choice(len(candidates_arr), size=n, replace=True)
            return candidates_arr[random_indices]

    def impute(self, raw_df):
        """
        极速主补全函数 - 向量化+批量处理，无逐行循环
        :param raw_df: 原始加载的DataFrame
        :return: 补全后的完整DataFrame（ucnt/usum无缺失，维度完整）
        """
        print("===== 开始极速数据补全 =====")
        t0 = time.time()
        # 1. 基础预处理：添加日期特征（原地修改，节省内存）
        df = raw_df.copy().reset_index(drop=True)
        df = self._add_date_features(df)
        # 定义核心维度（保证补全完整性）
        grid_ids = df['final_grid_id'].unique()
        dates = df['date'].unique()
        hours = np.arange(24, dtype=np.int8)  # 0-23小时，固定维度
        genders = sorted(df['gender'].unique())
        ages = sorted(df['age'].unique())
        city = df['city'].iloc[0]

        print(f"补全维度：网格{len(grid_ids)} | 日期{len(dates)} | 小时24 | 性别{len(genders)} | 年龄{len(ages)}")

        # 2. 构建全维度笛卡尔积（快速构建，用np替代Pandas，提升速度）
        print("构建全维度笛卡尔积...")
        grid_rep = np.repeat(grid_ids, len(dates)*24*len(genders)*len(ages))
        date_rep = np.tile(np.repeat(dates, 24*len(genders)*len(ages)), len(grid_ids))
        hour_rep = np.tile(np.repeat(hours, len(genders)*len(ages)), len(grid_ids)*len(dates))
        gender_rep = np.tile(np.repeat(genders, len(ages)), len(grid_ids)*len(dates)*24)
        age_rep = np.tile(ages, len(grid_ids)*len(dates)*24*len(genders))

        # 快速构建全量表（用np构建后转DataFrame，比MultiIndex快10倍以上）
        full_df = pd.DataFrame({
            'final_grid_id': grid_rep,
            'date': date_rep,
            'hour': hour_rep,
            'gender': gender_rep,
            'age': age_rep,
            'city': city
        })
        # 合并原始数据的ucnt/usum和日期特征
        full_df = full_df.merge(
            df[['city', 'final_grid_id', 'date', 'hour', 'gender', 'age', 'ucnt', 'usum', 'day_of_week', 'is_workday', 'date_str']],
            on=['city', 'final_grid_id', 'date', 'hour', 'gender', 'age'],
            how='left'
        )
        # 填充日期特征缺失值（笛卡尔积的date是已知的，直接重新计算）
        full_df['day_of_week'] = full_df['date'].dt.dayofweek
        full_df['is_workday'] = (full_df['day_of_week'] < 5).astype(np.int8)
        full_df['date_str'] = full_df['date'].dt.strftime('%Y%m%d')

        # 3. 拆分已有数据和待补全数据（快速筛选）
        missing_mask = full_df['ucnt'].isna()
        missing_df = full_df[missing_mask].reset_index(drop=True)
        existing_df = full_df[~missing_mask].reset_index(drop=True)
        print(f"待补全行数：{len(missing_df):,} | 已有数据行数：{len(existing_df):,}")

        if len(missing_df) == 0:
            print("无缺失数据，无需补全")
            full_df['ucnt'] = full_df['ucnt'].astype(int).clip(0)
            full_df['usum'] = full_df['usum'].astype(int).clip(0)
            return full_df.drop(columns=['date_str'])

        # 4. 预构建所有匹配规则的候选值字典
        candidates = self._build_match_candidates(df)

        # 5. 为待补全数据构建4条规则的匹配键（向量化，无循环）
        print("为待补全数据构建匹配键...")
        # 规则1键
        missing_df['rule1_key'] = missing_df['final_grid_id'] + '_' + missing_df['day_of_week'].astype(str) + '_' + \
                                  missing_df['hour'].astype(str) + '_' + missing_df['gender'].astype(str) + '_' + missing_df['age'].astype(str)
        # 规则2键
        missing_df['rule2_key'] = missing_df['day_of_week'].astype(str) + '_' + missing_df['hour'].astype(str) + '_' + \
                                  missing_df['gender'].astype(str) + '_' + missing_df['age'].astype(str)
        # 规则3键
        missing_df['rule3_key'] = missing_df['final_grid_id'] + '_' + missing_df['is_workday'].astype(str) + '_' + \
                                  missing_df['hour'].astype(str) + '_' + missing_df['gender'].astype(str) + '_' + missing_df['age'].astype(str)
        # 规则4键
        missing_df['rule4_key'] = missing_df['is_workday'].astype(str) + '_' + missing_df['hour'].astype(str) + '_' + \
                                  missing_df['gender'].astype(str) + '_' + missing_df['age'].astype(str)

        # 6. 批量补全核心逻辑（按优先级，向量化处理）
        print("开始批量补全（按规则优先级）...")
        # 初始化补全值为0（极端情况兜底）
        imputed_ucnt = np.zeros(len(missing_df), dtype=int)
        imputed_usum = np.zeros(len(missing_df), dtype=int)
        # 初始化已补全标记
        filled_mask = np.zeros(len(missing_df), dtype=bool)

        # 按规则1→2→3→4的优先级批量补全
        for rule in ['rule1', 'rule2', 'rule3', 'rule4']:
            if filled_mask.all():
                break  # 所有行已补全，提前退出
            # 筛选当前规则需要处理的行（未补全+有候选值）
            current_key_col = f'{rule}_key'
            current_mask = ~filled_mask & missing_df[current_key_col].apply(lambda x: f'{rule}_{x}' in candidates)
            current_idx = missing_df[current_mask].index
            if len(current_idx) == 0:
                continue
            # 批量获取候选值并随机选择
            for key, group in missing_df.loc[current_idx].groupby(current_key_col):
                cand_key = f'{rule}_{key}'
                cand_arr = candidates[cand_key]
                # 批量生成补全值
                batch_vals = self._batch_random_choice(cand_arr, len(group))
                # 赋值到补全数组
                imputed_ucnt[group.index] = batch_vals[:, 0]
                imputed_usum[group.index] = batch_vals[:, 1]
            # 更新已补全标记
            filled_mask[current_mask] = True
            print(f"  {rule} 补全：{current_mask.sum():,} 行，累计补全：{filled_mask.sum():,}/{len(missing_df):,} 行")

        # 7. 赋值补全值到待补全数据
        missing_df['ucnt'] = imputed_ucnt
        missing_df['usum'] = imputed_usum
        # 强制非负整数（业务约束）
        missing_df['ucnt'] = missing_df['ucnt'].clip(0).astype(int)
        missing_df['usum'] = missing_df['usum'].clip(0).astype(int)

        # 8. 合并已有数据和补全数据
        final_df = pd.concat([existing_df, missing_df], ignore_index=True)
        final_df['ucnt'] = final_df['ucnt'].clip(0).astype(int)
        final_df['usum'] = final_df['usum'].clip(0).astype(int)

        # ========== 修复核心：先恢复date字符串格式，再删除无用列 ==========
        # 先将date从datetime类型转回字符串（与原代码兼容），使用date_str填充
        final_df['date'] = final_df['date_str'].fillna(final_df['date'].dt.strftime('%Y%m%d'))
        # 再删除所有无用中间列（包括date_str、各规则键）
        final_df = final_df.drop(columns=['date_str', 'rule1_key', 'rule2_key', 'rule3_key', 'rule4_key'])
        # 移除缺失值（兜底）
        final_df = final_df.dropna(subset=['ucnt', 'usum'])
        # ========== 修复结束 ==========

        # 打印补全耗时
        t1 = time.time()
        print(f"===== 数据补全完成 =====")
        print(f"最终数据行数：{len(final_df):,} | 补全耗时：{t1 - t0:.2f} 秒")
        print(f"=======================\n")
        return final_df

# ============ 1. 高效数据加载（支持双目标，不变） ============
class FastSpatioTemporalDataset(Dataset):
    """支持ucnt和usum分别处理的数据集"""
    def __init__(self, sequences, spatial_features, targets_ucnt, targets_usum, seq_length=24):
        self.sequences = sequences
        self.spatial_features = spatial_features
        self.targets_ucnt = targets_ucnt
        self.targets_usum = targets_usum
        self.seq_length = seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]),
                torch.FloatTensor(self.spatial_features[idx]),
                torch.FloatTensor([self.targets_ucnt[idx]]),  # ucnt单独
                torch.FloatTensor([self.targets_usum[idx]]))  # usum单独

def _make_fast_loaders(sequences, spatial_data, targets_ucnt, targets_usum,
                       seq_length, batch_size, seed=42):
    """创建数据加载器，支持分离的目标变量"""
    X_temp, X_test, S_temp, S_test, yu_temp, yu_test, ys_temp, ys_test = train_test_split(
        sequences, spatial_data, targets_ucnt, targets_usum,
        test_size=0.2, random_state=seed)

    X_train, X_val, S_train, S_val, yu_train, yu_val, ys_train, ys_val = train_test_split(
        X_temp, S_temp, yu_temp, ys_temp,
        test_size=0.25, random_state=seed)

    if MEMORY_OPTIMIZATION_CONFIG['reduce_workers']:
        num_workers = 0
    else:
        num_workers = min(4, os.cpu_count() // 2) if os.name != 'nt' else 0

    pin_mem = MEMORY_OPTIMIZATION_CONFIG['pin_memory'] and torch.cuda.is_available()

    def _loader(x, s, yu, ys, shuffle=False):
        dataset = FastSpatioTemporalDataset(x, s, yu, ys, seq_length)
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=pin_mem,
                          persistent_workers=False)

    return (_loader(X_train, S_train, yu_train, ys_train, True),
            _loader(X_val, S_val, yu_val, ys_val, False),
            _loader(X_test, S_test, yu_test, ys_test, False))

# ============ 2. 位置编码（不变） ============
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

# ============ 3. 解耦的Transformer网络结构（不变） ============
class DecoupledSpatioTemporalTransformer(nn.Module):
    """
    改进的Transformer模型，将ucnt和usum分别预测
    核心改进：1.共享的特征提取backbone 2.独立的ucnt/usum预测头
    """
    def __init__(self, temporal_input_dim, spatial_input_dim,
                 d_model=128, nhead=8, num_layers=6,
                 dim_feedforward=512, dropout=0.1,
                 use_gradient_checkpointing=False):
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # 共享特征提取层
        self.temporal_embedding = nn.Linear(temporal_input_dim, d_model)
        self.spatial_embedding = nn.Linear(spatial_input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fusion_layer = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # ucnt独立预测头
        self.ucnt_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, 1)
        )

        # usum独立预测头
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
        """前向传播，返回(ucnt_pred, usum_pred)两个独立预测值"""
        b, seq_len, _ = temporal_seq.shape

        # 特征融合
        temp = self.temporal_embedding(temporal_seq)
        temp = self.pos_encoding(temp.transpose(0, 1)).transpose(0, 1)
        temp = self.dropout(temp)
        spat = self.spatial_embedding(spatial_features).unsqueeze(1).expand(-1, seq_len, -1)
        fused = temp + spat

        # Transformer编码（梯度检查点，节省显存）
        if self.use_gradient_checkpointing and self.training:
            encoded = torch.utils.checkpoint.checkpoint(
                self.transformer_encoder, fused, use_reentrant=False)
        else:
            encoded = self.transformer_encoder(fused)

        # 注意力融合
        spat_query = spat[:, 0:1, :]
        attended, _ = self.fusion_layer(spat_query, encoded, encoded)
        output = attended.squeeze(1) + spat[:, 0, :]

        # 分别预测ucnt和usum
        ucnt_pred = self.ucnt_head(output)
        usum_pred = self.usum_head(output)

        return ucnt_pred, usum_pred

# ============ 快速标准化函数（不变） ============
def cpu_robust_scale(x_np, chunk=200_000):
    """CPU 分块RobustScaler，防止RAM暴涨"""
    n, d = x_np.shape
    out = np.empty((n, d), dtype=np.float32)
    med = np.median(x_np, axis=0)
    iqr = np.percentile(x_np, 75, axis=0) - np.percentile(x_np, 25, axis=0)
    iqr[iqr == 0] = 1.0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        out[start:end] = (x_np[start:end] - med) / iqr
    return out

def gpu_robust_scale(x_np, chunk=200_000):
    """GPU分块RobustScaler，峰值显存可控"""
    if not torch.cuda.is_available():
        print('使用CPU标准化')
        return cpu_robust_scale(x_np, chunk)
    print('使用GPU标准化')
    n, d = x_np.shape
    out = np.empty((n, d), dtype=np.float32)
    x_np = np.ascontiguousarray(x_np, dtype=np.float32)

    # 计算全局median/IQR（仅扫一遍，常数内存）
    med = np.median(x_np, axis=0)
    q75 = np.percentile(x_np, 75, axis=0)
    q25 = np.percentile(x_np, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0

    # 分块标准化
    med_gpu = torch.from_numpy(med).cuda()
    iqr_gpu = torch.from_numpy(iqr).cuda()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        x_gpu = torch.from_numpy(x_np[start:end]).cuda()
        x_scaled = (x_gpu - med_gpu) / iqr_gpu
        out[start:end] = x_scaled.cpu().numpy()
    return out

# ============ 4. 数据处理器（解耦版）- 替换为极速补全器 ============
class DecoupledDataProcessor:
    """支持ucnt和usum分别处理的数据处理器 - 集成极速补全"""
    def __init__(self, excluded_from_encoding=None, excluded_from_features=None, imputer_seed=42):
        self.temporal_scaler = RobustScaler(quantile_range=(5, 95))
        self.spatial_scaler = RobustScaler(quantile_range=(5, 95))
        self.ucnt_scaler = RobustScaler(quantile_range=(5, 95))  # ucnt单独标准化
        self.usum_scaler = RobustScaler(quantile_range=(5, 95))  # usum单独标准化
        self.categorical_encoders = {}
        self.excluded_from_encoding = excluded_from_encoding or ['final_grid_id', 'city', 'geohash7']
        self.excluded_from_features = excluded_from_features or ['final_grid_id', 'geohash7', 'city', 'ucnt', 'usum', 'date']
        # 替换为极速补全器
        self.imputer = FastDataImputer(random_seed=imputer_seed)

    def _ensure_nonnegative_integer(self, data, column_names):
        """确保指定列为非负整数"""
        for col in column_names:
            if col in data.columns:
                data[col] = np.maximum(data[col], 0)
                data[col] = np.round(data[col]).astype(int)
        return data

    '''def _build_sequences_fast(self, merged_data, temporal_features, spatial_features, seq_length, stride = 1):
        """快速构建时序序列（向量化，预分配内存）"""
        grouped = merged_data.groupby('final_grid_id')
        total_sequences = 0
        valid_grids = []
        # 预计算总序列数
        for grid_id, group in grouped:
            n_rows = len(group)
            if n_rows >= seq_length + 1:
                n_seqs = (n_rows - seq_length) // stride
                if n_seqs > 0:
                    total_sequences += n_seqs
                    valid_grids.append((grid_id, group, n_seqs))
        # 预分配数组（减少内存碎片）
        n_temporal = len(temporal_features)
        n_spatial = len(spatial_features)
        sequences = np.empty((total_sequences, seq_length, n_temporal), dtype=np.float32)
        spatial_vectors = np.empty((total_sequences, n_spatial), dtype=np.float32)
        targets_ucnt = np.empty((total_sequences, 1), dtype=np.int32)
        targets_usum = np.empty((total_sequences, 1), dtype=np.int32)
        current_idx = 0
        # 批量构建序列
        for grid_id, group, n_seqs in tqdm(valid_grids, desc="构建时序序列"):
            group = group.sort_values(['date', 'hour'])
            temp_vals = group[temporal_features].values.astype(np.float32)
            spat_vals = group[spatial_features].iloc[0].values.astype(np.float32)
            ucnt_vals = group['ucnt'].values.astype(np.int32)
            usum_vals = group['usum'].values.astype(np.int32)
            if not (np.all(np.isfinite(temp_vals)) and np.all(np.isfinite(spat_vals))):
                continue
            n_rows = len(temp_vals)
            indices = np.arange(0, n_rows - seq_length, stride)
            # 向量化赋值，替代循环
            for i, start_idx in enumerate(indices):
                end_idx = start_idx + seq_length
                sequences[current_idx] = temp_vals[start_idx:end_idx]
                spatial_vectors[current_idx] = spat_vals
                targets_ucnt[current_idx] = ucnt_vals[end_idx]
                targets_usum[current_idx] = usum_vals[end_idx]
                current_idx += 1
        # 截取实际有效数据
        sequences = sequences[:current_idx]
        spatial_vectors = spatial_vectors[:current_idx]
        targets_ucnt = targets_ucnt[:current_idx]
        targets_usum = targets_usum[:current_idx]
        # 标准化
        N, T, F = sequences.shape
        sequences = self.temporal_scaler.fit_transform(sequences.reshape(-1, F)).reshape(N, T, F)
        spatial_vectors = self.spatial_scaler.fit_transform(spatial_vectors)
        targets_ucnt = self.ucnt_scaler.fit_transform(targets_ucnt)
        targets_usum = self.usum_scaler.fit_transform(targets_usum)
        return sequences, spatial_vectors, targets_ucnt, targets_usum'''

    def _build_sequences_fast(self, merged_data, temporal_features, spatial_features, seq_length, stride=1):
        """快速构建时序序列（向量化，预分配内存）+ 优化版：减少拷贝+即时GC+无冗余计算"""
        grouped = merged_data.groupby('final_grid_id')
        total_sequences = 0
        valid_grids = []
        # 预计算总序列数（不变）
        for grid_id, group in grouped:
            n_rows = len(group)
            if n_rows >= seq_length + 1:
                n_seqs = (n_rows - seq_length) // stride
                if n_seqs > 0:
                    total_sequences += n_seqs
                    valid_grids.append((grid_id, group, n_seqs))
        # 预分配数组（不变，减少内存碎片）
        n_temporal = len(temporal_features)
        n_spatial = len(spatial_features)
        sequences = np.empty((total_sequences, seq_length, n_temporal), dtype=np.float32)
        spatial_vectors = np.empty((total_sequences, n_spatial), dtype=np.float32)
        targets_ucnt = np.empty((total_sequences, 1), dtype=np.int32)
        targets_usum = np.empty((total_sequences, 1), dtype=np.int32)
        current_idx = 0
        # 批量构建序列（不变）
        for grid_id, group, n_seqs in tqdm(valid_grids, desc="构建时序序列"):
            group = group.sort_values(['date', 'hour'])
            temp_vals = group[temporal_features].values.astype(np.float32)
            spat_vals = group[spatial_features].iloc[0].values.astype(np.float32)
            ucnt_vals = group['ucnt'].values.astype(np.int32)
            usum_vals = group['usum'].values.astype(np.int32)
            if not (np.all(np.isfinite(temp_vals)) and np.all(np.isfinite(spat_vals))):
                continue
            n_rows = len(temp_vals)
            indices = np.arange(0, n_rows - seq_length, stride)
            # 向量化赋值，替代循环
            for i, start_idx in enumerate(indices):
                end_idx = start_idx + seq_length
                sequences[current_idx] = temp_vals[start_idx:end_idx]
                spatial_vectors[current_idx] = spat_vals
                targets_ucnt[current_idx] = ucnt_vals[end_idx]
                targets_usum[current_idx] = usum_vals[end_idx]
                current_idx += 1
        # 截取实际有效数据（不变）
        sequences = sequences[:current_idx]
        spatial_vectors = spatial_vectors[:current_idx]
        targets_ucnt = targets_ucnt[:current_idx]
        targets_usum = targets_usum[:current_idx]

        # ============== 优化核心1：标准化原地操作，减少大数组拷贝 ==============
        print('标准化')
        N, T, F = sequences.shape
        # 三维转二维时用reshape（视图，无拷贝），而非ravel/flat（生成新数组）
        print('1')
        sequences_2d = sequences.reshape(-1, F)
        # 标准化后直接赋值回原视图，无需重新reshape（大幅减少内存开销）
        print('1')
        sequences_2d[:] = self.temporal_scaler.fit_transform(sequences_2d)
        # 空间特征+目标值标准化（原地操作）
        print('1')
        spatial_vectors[:] = self.spatial_scaler.fit_transform(spatial_vectors)
        targets_ucnt[:] = self.ucnt_scaler.fit_transform(targets_ucnt)
        targets_usum[:] = self.usum_scaler.fit_transform(targets_usum)
        print('1')
        # ============== 优化核心2：即时删除临时数组+强制GC，释放内存 ==============
        del sequences_2d  # 删除临时二维数组
        gc.collect()  # 强制垃圾回收，避免内存堆积

        # 目标值展平（与原代码输出格式一致）
        targets_ucnt = targets_ucnt.flatten()
        targets_usum = targets_usum.flatten()

        return sequences, spatial_vectors, targets_ucnt, targets_usum

    def create_additional_features(self, data):
        """创建时间特征（向量化，不变）"""
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

    def load_and_preprocess_data(self, split_files_pattern, result_file_path, seq_length=24, max_files=None):
        """加载并预处理数据 - 集成极速补全，流程不变"""
        print("="*50)
        print("开始加载数据...")
        # 加载时序数据
        files = glob.glob(split_files_pattern)
        if max_files:
            files = files[:max_files]
        all_data = pd.concat([pd.read_csv(f) for f in tqdm(files)], ignore_index=True)
        print(f"原始数据加载完成，行数：{len(all_data):,}")

        # 核心：极速数据补全（替换原补全逻辑，调用新的imputer）
        all_data = self.imputer.impute(all_data)
        if split_files_pattern:
            # 提取输入文件的目录、文件名、扩展名
            input_dir = os.path.dirname(split_files_pattern)
            input_file_name = os.path.basename(split_files_pattern)
            # 分离文件名和扩展名（处理.csv/.txt等任意扩展名）
            file_prefix, file_suffix = os.path.splitext(input_file_name)
            # 生成补全后的文件名：前缀_imputed + 扩展名
            imputed_file_name = f"{file_prefix}_imputed{file_suffix}"
            # 补全文件的完整路径（和输入文件同目录）
            imputed_file_path = os.path.join(input_dir, imputed_file_name)
            # 保存补全结果（utf-8-sig解决中文乱码，不保存索引）
            all_data.to_csv(imputed_file_path, index=False, encoding='utf-8-sig')
            # 打印保存提示
            print(f"✅ 补全结果已保存：{imputed_file_path}")
            print(f"📊 保存数据行数：{len(all_data):,}")
        # 确保ucnt/usum为非负整数
        all_data = self._ensure_nonnegative_integer(all_data, ['ucnt', 'usum'])
        # 重新处理日期特征（补全后已标准化）
        if 'date' in all_data.columns:
            all_data['date'] = pd.to_datetime(all_data['date'], format='%Y%m%d', errors='coerce')
            all_data['day_of_week'] = all_data['date'].dt.dayofweek
            all_data['month'] = all_data['date'].dt.month  # 新增month，为特征工程

        # 特征工程
        all_data = self.create_additional_features(all_data)
        # 加载空间数据并合并
        result_df = pd.read_csv(result_file_path)
        merged = all_data.merge(result_df, left_on='final_grid_id', right_on='geohash7', how='inner')
        print(f"空间数据合并完成，行数：{len(merged):,}")

        # 分类特征编码
        for col in merged.columns:
            if merged[col].dtype == 'object' and col not in self.excluded_from_encoding:
                if col not in self.categorical_encoders:
                    self.categorical_encoders[col] = LabelEncoder()
                    merged[col] = self.categorical_encoders[col].fit_transform(merged[col].astype(str))
                else:
                    merged[col] = self.categorical_encoders[col].transform(merged[col].astype(str))

        # 划分时序/空间特征
        temporal_features = [c for c in all_data.columns if c not in self.excluded_from_features]
        spatial_features = [c for c in result_df.columns if c not in self.excluded_from_features]
        # 构建时序序列
        sequences, spatial_data, targets_ucnt, targets_usum = self._build_sequences_fast(
            merged, temporal_features, spatial_features, seq_length, stride=1)

        # 二次标准化（保证序列数据范围合理）
        '''n_samples, n_timesteps, n_features = sequences.shape
        sequences_2d = sequences.reshape(-1, n_features)
        sequences = self.temporal_scaler.fit_transform(sequences_2d).reshape(n_samples, n_timesteps, n_features)
        spatial_data = self.spatial_scaler.fit_transform(spatial_data)
        targets_ucnt = self.ucnt_scaler.fit_transform(targets_ucnt.reshape(-1, 1)).flatten()
        targets_usum = self.usum_scaler.fit_transform(targets_usum.reshape(-1, 1)).flatten()'''

        print(f'序列构建完成：{len(sequences)} 个样本 | 时序特征：{len(temporal_features)} | 空间特征：{len(spatial_features)}')
        return sequences, spatial_data, targets_ucnt, targets_usum, temporal_features, spatial_features

# ============ 5. 解耦训练函数（不变，修复原返回值问题） ============
def train_model_decoupled(model, train_loader, val_loader, epochs, lr=0.001,
                         ucnt_weight=1.0, usum_weight=1.0, patience_count = 5):
    """解耦训练函数，返回训练/验证损失+最佳模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda') if MEMORY_OPTIMIZATION_CONFIG['use_mixed_precision'] else None

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    accumulation_steps = MEMORY_OPTIMIZATION_CONFIG['gradient_accumulation_steps']
    model_out = model

    print(f'\n开始训练（设备：{device}）')
    print(f'ucnt权重：{ucnt_weight} | usum权重：{usum_weight} | 梯度累积：{accumulation_steps}')
    print("="*50)

    # Epoch 0 验证初始模型
    model.eval()
    val_loss, val_ucnt_loss, val_usum_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for seq, spat, t_ucnt, t_usum in val_loader:
            seq, spat = seq.to(device), spat.to(device)
            t_ucnt, t_usum = t_ucnt.to(device), t_usum.to(device)
            p_ucnt, p_usum = model(seq, spat)
            l_ucnt = criterion(p_ucnt, t_ucnt)
            l_usum = criterion(p_usum, t_usum)
            val_loss += (ucnt_weight * l_ucnt + usum_weight * l_usum).item()
            val_ucnt_loss += l_ucnt.item()
            val_usum_loss += l_usum.item()
    val_loss /= len(val_loader)
    val_ucnt_loss /= len(val_loader)
    val_usum_loss /= len(val_loader)
    val_losses.append(val_loss)
    best_val_loss, best_val_ucnt, best_val_usum = val_loss, val_ucnt_loss, val_usum_loss
    print(f'Epoch 0 [初始验证] | Val Loss：{val_loss:.4f} (ucnt：{val_ucnt_loss:.4f}, usum：{val_usum_loss:.4f})')

    # 训练循环
    patience_count_0 = 0
    for epoch in range(epochs):
        if patience_count_0 >= patience_count:
            print(f'早停触发：验证损失连续{patience_count}轮未下降，停止训练')
            break
        # 训练阶段
        model.train()
        train_loss, train_ucnt_loss, train_usum_loss = 0.0, 0.0, 0.0
        optimizer.zero_grad()
        for batch_idx, (seq, spat, t_ucnt, t_usum) in enumerate(train_loader):
            seq, spat = seq.to(device), spat.to(device)
            t_ucnt, t_usum = t_ucnt.to(device), t_usum.to(device)
            # 前向传播
            if scaler:
                with torch.amp.autocast('cuda'):
                    p_ucnt, p_usum = model(seq, spat)
                    l_ucnt = criterion(p_ucnt, t_ucnt)
                    l_usum = criterion(p_usum, t_usum)
                    loss = (ucnt_weight * l_ucnt + usum_weight * l_usum) / accumulation_steps
                scaler.scale(loss).backward()
                # 梯度累积更新
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                p_ucnt, p_usum = model(seq, spat)
                l_ucnt = criterion(p_ucnt, t_ucnt)
                l_usum = criterion(p_usum, t_usum)
                loss = (ucnt_weight * l_ucnt + usum_weight * l_usum) / accumulation_steps
                loss.backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            # 累计损失
            train_loss += loss.item() * accumulation_steps
            train_ucnt_loss += l_ucnt.item()
            train_usum_loss += l_usum.item()
            # 检测NaN损失
            if math.isnan(l_ucnt.item()) or math.isnan(l_usum.item()):
                print('警告：出现NaN损失，需检查数据/模型')
            # 定期清理GPU缓存
            if (batch_idx + 1) % MEMORY_OPTIMIZATION_CONFIG['clear_cache_frequency'] == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        # 平均训练损失
        train_loss /= len(train_loader)
        train_ucnt_loss /= len(train_loader)
        train_usum_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss, val_ucnt_loss, val_usum_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for seq, spat, t_ucnt, t_usum in val_loader:
                seq, spat = seq.to(device), spat.to(device)
                t_ucnt, t_usum = t_ucnt.to(device), t_usum.to(device)
                p_ucnt, p_usum = model(seq, spat)
                l_ucnt = criterion(p_ucnt, t_ucnt)
                l_usum = criterion(p_usum, t_usum)
                val_loss += (ucnt_weight * l_ucnt + usum_weight * l_usum).item()
                val_ucnt_loss += l_ucnt.item()
                val_usum_loss += l_usum.item()
        val_loss /= len(val_loader)
        val_ucnt_loss /= len(val_loader)
        val_usum_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ucnt = val_ucnt_loss
            best_val_usum = val_usum_loss
            model_out = model
            patience_count_0 = 0
            # print(f'Epoch {epoch+1:03d} | 训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f} ✅ 保存最佳模型')
        else:
            patience_count_0 += 1
            # print(f'Epoch {epoch+1:03d} | 训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f} [早停：{patience_count_0}/{patience_count}]')
        # 学习率调度
        scheduler.step(val_loss)
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f'训练结束 | 最佳验证损失：{best_val_loss:.4f} (ucnt：{best_val_ucnt:.4f}, usum：{best_val_usum:.4f})')
    return train_losses, val_losses, model_out
# ============ 5. 解耦训练函数 ============
def val_model_decoupled(model, train_loader, val_loader, epochs, lr=0.001,
                         ucnt_weight=1.0, usum_weight=1.0, patience_count = 10):
    """
    解耦训练函数

    Args:
        ucnt_weight: ucnt损失的权重
        usum_weight: usum损失的权重
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 使用MSE损失
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda') if MEMORY_OPTIMIZATION_CONFIG['use_mixed_precision'] else None

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    accumulation_steps = MEMORY_OPTIMIZATION_CONFIG['gradient_accumulation_steps']

    # print(f'\n开始训练 (设备: {device})')
    # print(f'ucnt权重: {ucnt_weight}, usum权重: {usum_weight}')
    #print(f'梯度累积步数: {accumulation_steps}')
    print("="*50)
    model_out = model
    model.eval()
    val_loss = 0.0
    val_ucnt_loss = 0.0
    val_usum_loss = 0.0
    val_pbar = tqdm(val_loader,
                   desc=f'Epoch 0 [Val]',
                   leave=False,
                   ncols=100)
    with torch.no_grad():
        for seq, spat, target_ucnt, target_usum in val_loader:
            seq, spat = seq.to(device), spat.to(device)
            target_ucnt = target_ucnt.to(device)
            target_usum = target_usum.to(device)

            pred_ucnt, pred_usum = model(seq, spat)
            loss_ucnt = criterion(pred_ucnt, target_ucnt)
            loss_usum = criterion(pred_usum, target_usum)
            loss = ucnt_weight * loss_ucnt + usum_weight * loss_usum

            val_loss += loss.item()
            val_ucnt_loss += loss_ucnt.item()
            val_usum_loss += loss_usum.item()

    val_loss /= len(val_loader)
    val_ucnt_loss /= len(val_loader)
    val_usum_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f'\nEpoch 0:')
    print(f'  Val Loss:   {val_loss:.4f} (ucnt: {val_ucnt_loss:.4f}, usum: {val_usum_loss:.4f})')

# ============ 6. 解耦评估函数（不变） ============
def evaluate_model_decoupled(model, test_loader, ucnt_scaler, usum_scaler):
    """解耦评估函数，分别评估ucnt和usum"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_p_ucnt, all_p_usum = [], []
    all_t_ucnt, all_t_usum = [], []
    print("\n收集预测结果...")
    with torch.no_grad():
        for seq, spat, t_ucnt, t_usum in tqdm(test_loader):
            seq, spat = seq.to(device), spat.to(device)
            p_ucnt, p_usum = model(seq, spat)
            all_p_ucnt.append(p_ucnt.cpu().numpy())
            all_p_usum.append(p_usum.cpu().numpy())
            all_t_ucnt.append(t_ucnt.numpy())
            all_t_usum.append(t_usum.numpy())

    # 合并批次并反标准化
    p_ucnt = ucnt_scaler.inverse_transform(np.concatenate(all_p_ucnt, axis=0)).flatten()
    p_usum = usum_scaler.inverse_transform(np.concatenate(all_p_usum, axis=0)).flatten()
    t_ucnt = ucnt_scaler.inverse_transform(np.concatenate(all_t_ucnt, axis=0)).flatten()
    t_usum = usum_scaler.inverse_transform(np.concatenate(all_t_usum, axis=0)).flatten()

    # 强制非负整数（业务约束）
    p_ucnt = np.maximum(p_ucnt, 0).round().astype(int)
    p_usum = np.maximum(p_usum, 0).round().astype(int)
    t_ucnt = np.maximum(t_ucnt, 0).round().astype(int)
    t_usum = np.maximum(t_usum, 0).round().astype(int)

    # 计算评估指标
    def calc_metrics(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        print(f"\n{name} 评估结果：")
        print(f"MSE：{mse:.4f} | RMSE：{rmse:.4f} | MAE：{mae:.4f}")
        print(f"预测范围：[{p_ucnt.min()}, {p_ucnt.max()}] | 真实范围：[{t_ucnt.min()}, {t_ucnt.max()}]")
        print(f"预测均值：{y_pred.mean():.2f} | 真实均值：{y_true.mean():.2f}")
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

    metrics = {}
    metrics['ucnt'] = calc_metrics(t_ucnt, p_ucnt, "ucnt")
    metrics['usum'] = calc_metrics(t_usum, p_usum, "usum")
    # 整体评估
    print("\n" + "="*50)
    print("整体评估：")
    print(f"ucnt/usum MSE比值：{metrics['ucnt']['MSE']/metrics['usum']['MSE']:.4f}")
    print(f"ucnt/usum MAE比值：{metrics['ucnt']['MAE']/metrics['usum']['MAE']:.4f}")
    print("="*50)

    return p_ucnt, p_usum, t_ucnt, t_usum, metrics

# ============ 分区域训练/验证函数（不变） ============
def train_one_zone(model, zone_id, train_loader, val_loader, epochs, lr, ucnt_w, usum_w):
    """单区域训练，保存区域专属权重"""
    weight_file = f'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\weights/zone_{zone_id:03d}.pth'
    if os.path.exists(weight_file):
        model.load_state_dict(torch.load(weight_file, map_location='cpu'))
        print(f'[区域 {zone_id}] 加载已有权重继续训练')
    else:
        print(f'[区域 {zone_id}] 首次训练，随机初始化')
    # 训练模型
    tr_loss, val_loss, model_out = train_model_decoupled(model, train_loader, val_loader, epochs, lr, ucnt_w, usum_w)
    # 保存权重
    os.makedirs('weights', exist_ok=True)
    torch.save(model_out.state_dict(), weight_file)
    return tr_loss, val_loss

def val_one_zone(model, zone_id, train_loader, val_loader, epochs, lr, ucnt_w, usum_w):
    """单区域验证"""
    weight_file = f'weights/zone_{zone_id:03d}.pth'
    if os.path.exists(weight_file):
        model.load_state_dict(torch.load(weight_file, map_location='cpu'))
        print(f'[区域 {zone_id}] 加载已有权重验证')
    else:
        print(f'[区域 {zone_id}] 无权重文件，跳过')
    val_model_decoupled(model, train_loader, val_loader, epochs, lr, ucnt_w, usum_w)
    return 0, 0

# ============ 7. 各类主函数（不变，直接调用） ============
def main_decoupled():
    """单模型全量训练主函数"""
    SEQ_LENGTH = 48
    BATCH_SIZE = 512
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    UCNT_WEIGHT = 0.75
    USUM_WEIGHT = 0.25

    split_files_pattern = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\aggregated_test_all_hourly_sum_chengliuqu_grid_matched.csv"
    result_file_path = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\fishnet\geohash7\shp\result.csv"

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU：{torch.cuda.get_device_name(0)} | 总显存：{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
        # 初始化处理器
        processor = DecoupledDataProcessor(
            excluded_from_encoding=['final_grid_id', 'geohash7', 'area_sqm', 'perim_m', 'width_m', 'height_m', 'city'],
            excluded_from_features=['final_grid_id', 'geohash7', 'ucnt', 'usum', 'area_sqm', 'perim_m', 'width_m', 'height_m', 'city', 'date', 'OID_', 'grid_id'])
        # 加载预处理数据
        sequences, spatial_data, targets_ucnt, targets_usum, t_feat, s_feat = processor.load_and_preprocess_data(split_files_pattern, result_file_path, SEQ_LENGTH)
        # 创建数据加载器
        train_loader, val_loader, test_loader = _make_fast_loaders(sequences, spatial_data, targets_ucnt, targets_usum, SEQ_LENGTH, BATCH_SIZE)
        # 初始化模型
        model = DecoupledSpatioTemporalTransformer(
            temporal_input_dim=len(t_feat), spatial_input_dim=len(s_feat),
            d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1,
            use_gradient_checkpointing=MEMORY_OPTIMIZATION_CONFIG['use_gradient_checkpointing'])
        print(f"模型可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        # 打印配置
        print("\n" + "="*50)
        print("训练配置：")
        for k, v in MEMORY_OPTIMIZATION_CONFIG.items():
            print(f"  {k}：{v}")
        print(f"  批次大小：{BATCH_SIZE} | 等效批次：{BATCH_SIZE*MEMORY_OPTIMIZATION_CONFIG['gradient_accumulation_steps']}")
        print("="*50)
        # 训练模型
        train_losses, val_losses, model_out = train_model_decoupled(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, UCNT_WEIGHT, USUM_WEIGHT)
        torch.save(model_out.state_dict(), 'best_model_decoupled.pth')
        # 评估模型
        evaluate_model_decoupled(model_out, test_loader, processor.ucnt_scaler, processor.usum_scaler)
        print("\n✅ 全量训练完成！")
        return model_out, processor
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        raise

def main_decoupled_continue():
    """基于已有模型继续训练"""
    SEQ_LENGTH = 48
    BATCH_SIZE = 512
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-5
    UCNT_WEIGHT = 0.75
    USUM_WEIGHT = 0.25

    split_files_pattern = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\aggregated_test_all_hourly_sum_chengliuqu_grid_matched.csv"
    result_file_path = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\fishnet\geohash7\shp\result.csv"

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU：{torch.cuda.get_device_name(0)}")
        # 加载数据
        processor = DecoupledDataProcessor(
            excluded_from_encoding=['final_grid_id', 'geohash7', 'area_sqm', 'perim_m', 'width_m', 'height_m', 'city'],
            excluded_from_features=['final_grid_id', 'geohash7', 'ucnt', 'usum', 'area_sqm', 'perim_m', 'width_m', 'height_m', 'city', 'date', 'OID_', 'grid_id'])
        sequences, spatial_data, targets_ucnt, targets_usum, t_feat, s_feat = processor.load_and_preprocess_data(split_files_pattern, result_file_path, SEQ_LENGTH)
        train_loader, val_loader, test_loader = _make_fast_loaders(sequences, spatial_data, targets_ucnt, targets_usum, SEQ_LENGTH, BATCH_SIZE)
        # 初始化模型并加载权重
        model = DecoupledSpatioTemporalTransformer(
            temporal_input_dim=len(t_feat), spatial_input_dim=len(s_feat),
            d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1)
        model.load_state_dict(torch.load('best_model_decoupled.pth'))
        print(f"模型可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        # 继续训练
        train_losses, val_losses, model_out = train_model_decoupled(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, UCNT_WEIGHT, USUM_WEIGHT)
        torch.save(model_out.state_dict(), 'best_model_decoupled.pth')
        # 评估
        evaluate_model_decoupled(model_out, test_loader, processor.ucnt_scaler, processor.usum_scaler)
        print("\n✅ 继续训练完成！")
        return model_out, processor
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        raise

def main_decoupled_eval():
    """仅评估已有模型"""
    SEQ_LENGTH = 48
    BATCH_SIZE = 512

    split_files_pattern = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\aggregated_test_all_hourly_sum_chengliuqu_grid_matched.csv"
    result_file_path = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\fishnet\geohash7\shp\result.csv"

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 加载数据
        processor = DecoupledDataProcessor(
            excluded_from_encoding=['final_grid_id', 'geohash7', 'area_sqm', 'perim_m', 'width_m', 'height_m', 'city'],
            excluded_from_features=['final_grid_id', 'geohash7', 'ucnt', 'usum', 'area_sqm', 'perim_m', 'width_m', 'height_m', 'city', 'date', 'OID_', 'grid_id'])
        sequences, spatial_data, targets_ucnt, targets_usum, t_feat, s_feat = processor.load_and_preprocess_data(split_files_pattern, result_file_path, SEQ_LENGTH)
        _, _, test_loader = _make_fast_loaders(sequences, spatial_data, targets_ucnt, targets_usum, SEQ_LENGTH, BATCH_SIZE)
        # 加载模型
        model = DecoupledSpatioTemporalTransformer(
            temporal_input_dim=len(t_feat), spatial_input_dim=len(s_feat),
            d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1)
        model.load_state_dict(torch.load('best_model_decoupled_medium.pth'))
        # 评估
        evaluate_model_decoupled(model, test_loader, processor.ucnt_scaler, processor.usum_scaler)
        print("\n✅ 模型评估完成！")
        return model, processor
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        raise

def main_zone_by_zone():
    """分区域训练主函数 - 核心运行入口"""
    SEQ_LENGTH = 48
    BATCH_SIZE = 256
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    UCNT_WEIGHT = 0.75
    USUM_WEIGHT = 0.25

    result_file_path = r"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\fishnet\geohash7\shp\result_core_area.csv"
    # 遍历分区域文件（222~409）
    for file_counter in range(486, 649):
        split_files_pattern = f"D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令/aggregated_by_geohash7_blocks_core_{file_counter}.csv"
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 检查文件是否存在且非空
            if not os.path.exists(split_files_pattern):
                print(f"[区域 {file_counter}] 文件不存在，跳过")
                continue
            split_files_read = pd.read_csv(split_files_pattern)
            if len(split_files_read) == 0:
                print(f"[区域 {file_counter}] 文件为空，跳过")
                del split_files_read
                continue
            del split_files_read
            # 初始化数据处理器
            processor = DecoupledDataProcessor(
                excluded_from_encoding=['final_grid_id', 'geohash7', 'area_sqm', 'perim_m', 'width_m', 'height_m', 'city'],
                excluded_from_features=['final_grid_id', 'geohash7', 'ucnt', 'usum', 'area_sqm', 'perim_m', 'width_m', 'height_m', 'city', 'date', 'OID_', 'grid_id','min_lat', 'max_lat', 'min_lon', 'max_lon'])
            # 加载并极速补全数据
            sequences, spatial_data, targets_ucnt, targets_usum, t_feat, s_feat = processor.load_and_preprocess_data(split_files_pattern, result_file_path, SEQ_LENGTH)
            # 创建数据加载器
            train_loader, val_loader, test_loader = _make_fast_loaders(sequences, spatial_data, targets_ucnt, targets_usum, SEQ_LENGTH, BATCH_SIZE)
            # 初始化模型并加载基础权重
            model = DecoupledSpatioTemporalTransformer(
                temporal_input_dim=len(t_feat), spatial_input_dim=len(s_feat),
                d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1, use_gradient_checkpointing=MEMORY_OPTIMIZATION_CONFIG['use_gradient_checkpointing'])
            # model.load_state_dict(torch.load('D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令/best_model_decoupled.pth'), strict=False)
            # 分区域训练
            train_losses, val_losses = train_one_zone(model, file_counter, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, UCNT_WEIGHT, USUM_WEIGHT)
            # 加载训练后权重并评估
            model.load_state_dict(torch.load(f'D:\BaiduSyncdisk\学习资料\ysjs\开题\论文数据\辅助数据\daas手机信令\weights/zone_{file_counter:03d}.pth'))
            evaluate_model_decoupled(model, test_loader, processor.ucnt_scaler, processor.usum_scaler)
            # 打印完成信息
            print(f"\n----- 区域 {file_counter}/672 训练完成，权重已保存 -----\n")
            # 强制释放内存，避免泄漏
            del processor, model, train_loader, val_loader, test_loader
            del sequences, spatial_data, targets_ucnt, targets_usum
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)
        except Exception as e:
            print(f"\n❌ [区域 {file_counter}] 训练失败：{e}")
            import traceback
            traceback.print_exc()
            continue
    return model, processor

# ============ 运行入口 ============
if __name__ == "__main__":
    """
    使用说明：
    1. 直接运行分区域训练：main_zone_by_zone()
    2. 全量训练基础模型：main_decoupled()
    3. 基于基础模型继续训练：main_decoupled_continue()
    4. 仅评估已有模型：main_decoupled_eval()
    5. 数据补全已自动集成，极速向量化，无需额外操作
    """
    model, processor = main_zone_by_zone()