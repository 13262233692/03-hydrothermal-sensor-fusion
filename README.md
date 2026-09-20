# ROV 深海热液羽流多传感器数据融合系统

面向深海热液探测 ROV 的多传感器数据融合管线：pH / H₂S / 温度 / 浊度
四类传感器以各自频率经串口回传 CSV 流，系统完成时间对齐、异常值剔除、
卡尔曼滤波融合，并用普通克里金（Ordinary Kriging）重建热液羽流的三维
浓度剖面，输出 CF-1.8 约定的 NetCDF-4 产品供科研使用。

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `hydrothermal_fusion/serial_reader.py` | 串口/文件 CSV 流读取、解析、多线程多口采集 |
| `hydrothermal_fusion/kalman_fusion.py` | Hampel 预筛 + 卡方新息门控的恒速卡尔曼滤波，统一时间网格对齐 |
| `hydrothermal_fusion/profile_builder.py` | 地理配准散点 → 局部邻域普通克里金三维插值 |
| `hydrothermal_fusion/netcdf_writer.py` | CF-1.8 NetCDF-4 产品输出（三维场 + 克里金方差 + 融合时间序列） |
| `hydrothermal_fusion/sensor_simulator.py` | 无硬件端到端测试用的高斯羽流 + 割草机航迹模拟器 |
| `main.py` | 命令行编排（`simulate` / `fuse`） |

## CSV 流格式

每行一条记录：`channel,epoch_seconds,v1[,v2,...]`，`#` 开头为注释：

```
h2s,1712345678.125,143.2
nav,1712345678.050,1201.4,502.9,-1480.2
```

`nav` 通道携带 ROV 的 x/y/z 位置（局部测量坐标系，米）。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 无硬件端到端验证：生成模拟 CSV 流（含噪声与尖峰异常值）
python3 main.py simulate --out data/sim --duration 600

# 2) 融合并重建三维剖面，输出 NetCDF
python3 main.py fuse --input data/sim --out plume_profile.nc

# 实航模式：直接监听串口
python3 main.py fuse --ports /dev/ttyUSB0 /dev/ttyUSB1 \
    --listen-seconds 600 --out plume_profile.nc
```

## 处理流程

1. **采集**：每个串口一个守护线程，样本进入共享队列后按通道归集。
2. **异常值剔除**：两级防护——Hampel 滚动中位数/MAD 预筛，
   卡尔曼更新时再做新息卡方门控（默认 χ²=9，约 3σ）。
3. **时间对齐**：各通道在自身（异步）采样时刻滤波，随后内插到统一
   1 Hz 时间网格；超过 `max-gap` 秒的数据空缺不外推，置 NaN。
4. **剖面重建**：融合序列与导航位置配准为 (x, y, z, value) 散点，
   采用指数变差函数的局部邻域普通克里金插值到规则三维网格，
   同时输出克里金方差作为不确定性度量。
5. **输出**：NetCDF-4，含每个通道的三维场、克里金方差、融合时间
   序列及 ROV 轨迹，遵循 CF-1.8 元数据约定。

## 调参

各通道的过程噪声 / 量测噪声 / 门控阈值见
`hydrothermal_fusion/kalman_fusion.py` 中的 `DEFAULT_CONFIGS`；
变差函数变程默认取测量区域尺度的 1/4，可在
`profile_builder.build_profile` 中显式指定。
