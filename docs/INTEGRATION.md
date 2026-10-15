# 平台集成指南

## 1. 推荐集成边界

平台侧负责模型、数据、各安全模块损失、风险评估和优化器；MSPC只负责共享参数梯度协调。这样可以保持算法包与具体多模态模型解耦。

```text
训练平台
  ├─ 计算多个安全模块损失
  ├─ 生成归一化风险向量
  ├─ 调用MSPCCoordinator.coordinate_losses()
  ├─ 执行optimizer.step()
  └─ 保存MSPC状态和诊断信息
```

## 2. 初始化

```python
from mspc import MSPCConfig, MSPCCoordinator

coordinator = MSPCCoordinator(
    MSPCConfig(
        mode="core",
        risk_thresholds=[0.4, 0.4, 0.4],
        history_size=5,
        grid_step=0.05,
    )
)
```

`risk_thresholds`的长度必须与安全模块数一致。风险归一化上下界应由训练集或验证集预实验确定，并在正式实验前冻结。

## 3. 嵌入训练循环

```python
optimizer.zero_grad(set_to_none=True)

outputs = model(batch)
losses = [
    trusted_reasoning_loss(outputs, batch),
    hallucination_loss(outputs, batch),
    alignment_loss(outputs, batch),
]
risks = risk_adapter.evaluate(outputs, batch)

result = coordinator.coordinate_losses(
    losses=losses,
    shared_parameters=model.shared_parameters(),
    risks=risks,
)
optimizer.step()
```

接入时必须满足：

- 所有损失来自同一次参数更新前的模型版本；
- `shared_parameters`顺序在训练和恢复过程中保持一致；
- 风险值有限且位于 `[0, 1]`；
- 分布式训练时，每个模块的梯度和风险需要采用一致的聚合策略；
- 混合精度训练应在梯度反缩放后执行冲突计算。

## 4. 状态保存与恢复

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "mspc": coordinator.state_dict(),
}

coordinator.load_state_dict(checkpoint["mspc"])
```

也可以使用 `save_full_checkpoint()`和 `load_full_checkpoint()`统一保存模型、优化器、调度器、MSPC历史和随机状态。

## 5. 服务接口建议

如果未来平台通过服务调用算法，建议服务层只传递以下数据：

```json
{
  "model_version": "string",
  "gradients": "Tensor[m,d]",
  "risks": "Tensor[m]",
  "active_masks": "Tensor[m,d] or null"
}
```

返回：

```json
{
  "alpha": "Tensor[m]",
  "fused_gradient": "Tensor[d]",
  "q": "Tensor[m]",
  "conflict_matrix": "Tensor[m,m]",
  "diagnostics": "dict"
}
```

服务端必须验证同一轮请求的 `model_version`一致，避免融合来自不同参数版本的梯度。生产系统不建议直接用JSON传输大规模梯度，可使用共享内存、对象存储或张量通信框架，只在控制接口中传递引用和元数据。

## 6. 上线前验证

至少验证以下内容：

1. 单元测试全部通过；
2. 两个及以上真实安全模块能够完成端到端更新；
3. 失败验证确实恢复模型、优化器、调度器和MSPC历史；
4. 风险阈值和归一化边界来自固定验证集；
5. 对比等权、PCGrad等基线，并报告安全指标、通用能力、耗时和显存；
6. 检查点恢复后权重和历史状态连续。
