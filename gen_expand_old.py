"""将旧数据集（data/simulator，202×48 = 16 host × 3 维）扩充为
新数据集格式（16 host × 7 维，202×112）。

策略：cpu/ram/disk 三列**原样保留**（分布、时间结构、异常事件 100% 一致），
只新增 4 列读写通道（固定系数缩放主列）：
  [cpu, ram, ram_r, ram_w, disk, disk_r, disk_w] × 16 host
- 读写列 = 主列 × 每 host 固定系数 → 读写列的 98 分位阈值与主列同步，
  不产生独立异常事件，标签与旧数据 form_test_dataset（全局 p98）一致
- schedule_series 原样复制（202×16×16 one-hot，完整共置信息）
- 行 0 保持全 0（初始化行）

这是"把旧数据集扩充为新格式"的最忠实做法：模型看到的 cpu/ram/disk
分布与旧数据集完全相同，只有输入宽度从 48 → 112（含 64 列冗余缩放列）。

用法：
  EXPAND_OUT=recovery/PreGANSrc/data/expand_old/ python gen_expand_old.py
"""
import os
import numpy as np

OUT = os.environ.get('EXPAND_OUT', 'recovery/PreGANSrc/data/expand_old/')
SRC = os.environ.get('EXPAND_SRC', 'recovery/PreGANSrc/data/simulator/')

old_ts = np.load(os.path.join(SRC, 'time_series.npy'))      # (202, 48)
old_ss = np.load(os.path.join(SRC, 'schedule_series.npy'))  # (202, 16, 16)
T, D = old_ts.shape
N = D // 3  # 16

cpu = old_ts[:, 0::3]   # (T, 16)
ram = old_ts[:, 1::3]
disk = old_ts[:, 2::3]

# 读写列：每 host 固定系数（同步缩放，不引入独立异常）
rng = np.random.default_rng(42)
ram_c = rng.uniform(0.0005, 0.002, N)
ram_wc = rng.uniform(0.0002, 0.001, N)
disk_c = rng.uniform(0.0001, 0.001, N)
disk_wc = rng.uniform(0.00005, 0.0005, N)
ram_r = ram * ram_c
ram_w = ram * ram_wc
disk_r = disk * disk_c
disk_w = disk * disk_wc

cols = []
for h in range(N):
    cols += [cpu[:, h], ram[:, h], ram_r[:, h], ram_w[:, h],
             disk[:, h], disk_r[:, h], disk_w[:, h]]
time_series = np.stack(cols, axis=1)  # (T, 112)

os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, 'time_series.npy'), time_series)
np.save(os.path.join(OUT, 'schedule_series.npy'), old_ss)
print(f'written: {OUT}time_series.npy {time_series.shape}')
print(f'written: {OUT}schedule_series.npy {old_ss.shape}')

# 分布检查
print(f'cpu==100: {(cpu == 100).mean():.3f} (原 0.594)')
print(f'全 0 host-step: '
      f'{((time_series.reshape(T, N, 7)[:, :, :3] == 0).all(axis=2)).mean():.3f} (原 0.385)')
print(f'占用 host/行: {((old_ss.sum(axis=1)) > 0).sum(axis=1).mean():.2f} (原 10.5)')
