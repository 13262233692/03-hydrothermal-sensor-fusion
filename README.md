# Hydrothermal Sensor Fusion

深海热液探测 ROV 多传感器数据融合系统：融合 pH、H₂S、温度、浊度四类
异频采样传感器，经卡尔曼滤波去噪与时间对齐后，用普通克里金（Ordinary
Kriging）重建热液羽流三维浓度剖面，输出 CF 风格 NetCDF 供科研使用。

## 模块结构

| 模块 | 职责 |
|---|---|
| `hydrofusion/serial_reader.py` | 串口/文件 CSV 流读取与解析（`SerialCSVReader`、`FileCSVReader`） |
| `hydrofusion/kalman_fusion.py` | 每通道恒速卡尔曼滤波、多速率时间对齐、NIS 卡方门限异常值剔除、加权异常指数融合 |
| `hydrofusion/profile_builder.py` | ROV 轨迹插值定位 + 普通克里金三维剖面插值（指数变差函数、KD 树局部邻域） |
| `hydrofusion/netcdf_writer.py` | 网格化剖面写入 NetCDF4（含克里金方差、CF-1.8 元数据） |
| `main.py` | 端到端 CLI 流程 |
| `scripts/simulate_rov_stream.py` | 合成 ROV 测量流生成器（测试/演示用） |
| `tests/test_pipeline.py` | 单元测试 + 端到端测试 |

## CSV 流格式

每行一个样本，位置可选：

```
timestamp,sensor,value[,x,y,z]
1726.500,h2s,12.300000,1.000,2.000,-100.000
2026-09-20T10:00:01Z,ph,5.1
```

`timestamp` 支持 UNIX 秒或 ISO-8601；`sensor ∈ {ph, h2s, temperature, turbidity}`；
`x,y,z` 为 ROV 位置（米，z 向上为正）。

## 快速开始

```bash
pip install -r requirements.txt

# 生成 5 分钟合成测量流（含 1% 毛刺）
python scripts/simulate_rov_stream.py --output data/simulated_stream.csv --duration 300

# 回放文件 -> 融合 -> 克里金 -> NetCDF
python main.py --input data/simulated_stream.csv --output data/plume_profile.nc --grid 20 20 14

# 实时串口模式
python main.py --port /dev/ttyUSB0 --baud 115200 --duration 120 --output plume.nc
```

## 处理流程

1. **读取**：串口 CSV 流逐行解析，坏行计数跳过。
2. **滤波**：每个传感器一个恒速模型卡尔曼滤波器（状态 `[值, 变化率]`，
   连续白噪声加速度过程模型），按实际 `dt` 预测，天然支持异步多速率采样。
3. **异常值剔除**：新息卡方门限（NIS = y²/S > 9，约 3σ）判为毛刺丢弃；
   连续拒绝超过 5 次自动放行一次，防止真实工况跳变导致滤波器锁死。
4. **时间对齐**：在公共时间网格上回放各通道已接受样本，预测到网格历元，
   输出同步多通道序列及滤波方差；并生成加权 z-score 热液异常指数
   （pH 取负号——热液流体偏酸）。
5. **剖面重建**：ROV 位置线性插值到网格历元，普通克里金（指数变差函数，
   最多 32 个近邻局部求解）把各通道插值到规则三维网格，同时输出克里金方差。
6. **输出**：NetCDF4，维度 `(z, y, x)`，每通道一个变量 + `<name>_variance`，
   NaN 映射为 `_FillValue`，附 CF-1.8 全局属性。

## 输出验证（合成数据）

对 300 s 合成测量流（羽流中心真值 `(0, 0, -95)` m）：

- 四通道重建场与真值相关系数 ≈ **0.935**
- 温度场峰值定位 `(1.2, 2.6, -99.3)` m
- 毛刺剔除率与注入的 1% 毛刺率一致

## 测试

```bash
python -m unittest discover -s tests -v
```

## 调参

- 各通道噪声参数见 `kalman_fusion.DEFAULT_SENSOR_CONFIG`
  （`process_noise` 控制跟踪梯度的敏捷度，`measurement_noise` 为观测方差）。
- 克里金参数（变程、块金、近邻数）见 `OrdinaryKriging3D`；
  CLI 可用 `--kriging-range`、`--gate`、`--dt`、`--grid` 调整。
