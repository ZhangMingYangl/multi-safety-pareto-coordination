# LibMTL集成

`mspc.libmtl.MSPC`实现了LibMTL的`AbsWeighting`接口，可作为共享参数多任务训练中的weighting方法使用。

## 前提

1. 安装或克隆LibMTL，并确保`import LibMTL`可以成功执行。
2. 安装本项目：`python -m pip install -e .`。
3. 使用共享参数梯度模式，设置`rep_grad=False`。

## 注册

在LibMTL项目的weighting注册位置导入：

```python
from mspc.libmtl import MSPC
```

然后把`MSPC`加入项目自身的weighting映射。不同LibMTL版本的注册文件位置可能不同，因此本仓库不直接修改第三方源码。

## 关键参数

- `mspc_risks`：显式提供长度等于任务数的风险向量；
- `mspc_risk_lower`、`mspc_risk_upper`：未显式提供风险时，用于把任务损失归一化到`[0, 1]`；
- `mspc_risk_thresholds`：历史欠优化判断阈值；
- `mspc_grid_step`：Pareto候选网格步长；
- `mspc_diagnostics_path`：可选的JSONL诊断输出路径；
- `mspc_log_interval`：诊断写入间隔。

正式实验不应使用默认的损失归一化边界`[0, 1]`。应先通过预实验估计每个任务的上下界，冻结后再运行对比实验。

LibMTL适配器只处理核心模式。需要更新后安全验证和事务回滚时，应在平台训练循环中直接使用`FullMSPCRunner`。
