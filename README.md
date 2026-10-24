# Multi-Safety Pareto Coordination (MSPC)

MSPC 是面向多模态大模型多安全模块的共享参数梯度冲突消解算法实现，对应专利说明书《一种面向多模态大模型的多安全模块参数冲突消解方法》。项目提供可安装的 PyTorch 算法包、单元测试、最小示例和平台接入说明。

> 发布状态：v0.1.0，供课题测试与学术交流使用。

## 主要功能

- `core`：采集同一参数版本上的多模块梯度，构建梯度冲突矩阵，执行尺度协调、顺序冲突投影、三目标 Pareto 权重求解和统一梯度融合。
- `full`：在更新后执行风险、可用性与稳定性验证；失败时支持反馈调权、学习率衰减，以及模型、优化器、调度器、MSPC 历史和随机状态回滚。
- `MSPC-Hybrid`：用于消融实验的工程增强配置，提供保尺度投影和 MGDA 公共下降保护；该配置必须与专利核心模式分别报告。
- LibMTL：提供兼容 `AbsWeighting` 的 `MSPC.backward()` 适配器。

## 项目结构

```text
multi-safety-pareto-coordination/
├── mspc/                    # MSPC算法包
├── tests/                   # 单元测试
├── examples/                # 最小可运行示例
├── docs/                    # 算法和平台接入说明
├── integrations/libmtl/     # LibMTL接入说明
├── pyproject.toml           # Python安装配置
├── THIRD_PARTY_NOTICES.md   # 第三方来源与许可证说明
└── OPEN_SOURCE_CHECKLIST.md # 公开发布前检查清单
```

## 环境与安装

要求 Python 3.10+、PyTorch 2.0+。

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

如果需要 CUDA，请先按照目标服务器的 CUDA 版本安装匹配的 PyTorch，再执行 `python -m pip install -e .`。

## 测试和最小示例

```powershell
python -m unittest discover -s tests -v
python -m examples.toy_training
```

示例运行后会输出每一步的模块权重 `alpha`、均衡因子 `q` 和冲突对比例。

## 核心模式用法

```python
import torch
from mspc import MSPCConfig, MSPCCoordinator

coordinator = MSPCCoordinator(
    MSPCConfig(
        mode="core",
        grid_step=0.05,
        risk_thresholds=[0.5, 0.5, 0.5],
    )
)

optimizer.zero_grad(set_to_none=True)
losses = [loss_reasoning, loss_hallucination, loss_alignment]
risks = torch.tensor([0.7, 0.5, 0.4], device=losses[0].device)

result = coordinator.coordinate_losses(
    losses,
    model.shared_parameters(),
    risks,
)
optimizer.step()

print(result.alpha)
print(result.q)
print(result.diagnostics.to_dict())
```

所有损失必须由同一模型参数版本完成前向计算。`coordinate_losses()` 只把融合梯度写入共享参数的 `.grad`，参数更新仍由调用方统一执行 `optimizer.step()`。

如果已经取得外部梯度矩阵，可调用：

```python
result = coordinator.coordinate(gradients, risks, active_masks)
```

- `gradients`：形状为 `[模块数, 共享梯度维数]`。
- `risks`：形状为 `[模块数]`，每个值必须位于 `[0, 1]`。
- `active_masks`：可选，形状与 `gradients` 相同，用于区分“参数未参与当前模块”和“参数参与但梯度为零”。

返回值包含 `alpha`、`fused_gradient`、`q`、投影后梯度、冲突矩阵及诊断信息。

## 完整模式

`FullMSPCRunner` 要求调用方提供：

- `compute_losses()`：从当前参数版本重新前向并返回多个安全模块损失；
- `evaluate()`：在独立验证数据上返回 `SafetyEvaluation(risks, utility, stability)`。

更新未通过安全验证时，完整模式会恢复事务起点，并按配置进行反馈重算和学习率衰减。达到重试上限后，模型参数、优化器状态、调度器状态、MSPC历史和随机状态都会回滚。

## 文档

- [算法流程与接口](docs/ALGORITHM.md)
- [平台集成指南](docs/INTEGRATION.md)
- [LibMTL集成指南](integrations/libmtl/README.md)
- [第三方来源说明](THIRD_PARTY_NOTICES.md)
- [公开发布前检查清单](OPEN_SOURCE_CHECKLIST.md)

## 实验边界

通用多任务数据集只能验证共享参数上的梯度协调机制，不能单独证明真实安全风险下降。正式验证至少应接入两个真实安全模块，并用独立验证集报告各模块风险、模型可用性、训练稳定性、额外计算时间和显存开销。

## 许可证

本仓库暂未附带开源许可证，仅供课题测试与学术交流使用。如需在其他项目中使用，请联系作者。
