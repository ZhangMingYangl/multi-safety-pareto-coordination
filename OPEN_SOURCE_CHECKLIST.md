# MSPC公开发布前检查清单

## 已完成

- [x] 核心算法代码独立放在`mspc/`。
- [x] 单元测试独立放在`tests/`。
- [x] 提供最小可运行示例。
- [x] 提供算法接口和平台集成文档。
- [x] 提供第三方代码来源及许可证说明。
- [x] `.gitignore`排除虚拟环境、IDE配置、数据集、权重、检查点和实验输出。
- [x] Git仓库根目录与本地父目录`Github_MSPC`分离。

## 公开前必须完成

- [ ] 确认专利已经提交并取得可核验的申请信息或受理材料。
- [ ] 由指导教师、项目负责人或学校知识产权管理部门确认代码权属和可公开范围。
- [ ] 确认是否允许通过许可证授予专利实施权。
- [ ] 根据确认结果添加正式`LICENSE`，不要只在README中写许可证名称。
- [ ] 确认作者、单位、联系邮箱和项目资助信息。
- [ ] 对全部文件执行密钥、个人信息、服务器地址和内部数据路径检查。
- [ ] 在全新Python环境执行安装、测试和最小示例。
- [ ] 先推送到GitHub私有仓库，由导师检查后再改为Public。
- [ ] 公开后把真实仓库地址补入`CITATION.cff`和`pyproject.toml`。
- [ ] 创建`v0.1.0`标签和GitHub Release。

## 建议的仓库信息

- Repository name：`multi-safety-pareto-coordination`
- Description：`PyTorch implementation of Multi-Safety Pareto Coordination for shared-parameter gradient conflict resolution.`
- Topics：`pytorch`、`ai-safety`、`multi-task-learning`、`multi-objective-optimization`、`gradient-conflict`

## 不应提交

- 未公开的数据集、模型权重、训练检查点和实验日志；
- API Key、Token、密码、私钥和远程服务器配置；
- 未经许可复制的第三方仓库源码；
- 专利申请内部沟通记录、身份证明、盖章文件和未公开申请材料；
- 含有个人信息或平台内部地址的截图与配置文件。
