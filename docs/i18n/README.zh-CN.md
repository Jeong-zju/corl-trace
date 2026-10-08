<div align="center">

# TRACE

### Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation

**CoRL 2026 · 主会论文**

Zihao Li · Ranpeng Qiu · Yincong Chen · Guoqiang Ren · Weiming Zhi  
The University of Sydney · Zeno AI

[项目主页](https://jeong-zju.github.io/trace/) · [快速开始](#快速开始) · [训练](../training.md) · [评估](../evaluation.md) · [引用](#引用)

[English](../../README.md) · **简体中文** · [日本語](README.ja.md) · [한국어](README.ko.md)

</div>

![TRACE 保存早期线索，在该线索消失后支持后续决策。](../assets/teaser.png)

**记住线索，在需要时做出决策。** TRACE 为视觉运动策略提供固定容量的因果记忆，
用于早期线索在后续决策前已经消失的任务。路径签名概括机器人已经执行的状态轨迹，
引导视觉和状态证据的写入与检索；轻量适配器将记忆接入 ACT 和 Diffusion Policy。

## 包含内容

- **完整 TRACE 主方法**，支持回归式 ACT 和扩散式策略两种接口。
- 因果前缀数据处理、签名缓存、训练及离线／仿真评估。
- 使用合成观测和真实路径签名的 CPU 示例，无需下载预训练权重。
- 参考配置、针对性测试和四语文档。

此开源版只包含主方法，不包含消融实现、对比实验和私有机器人部署配置。
数据集、训练权重及论文各次运行的完整产物**不随源码提供**；参考配置是使用起点，
不代表已经精确复现论文结果。详见[发布范围](../release-scope.md)。

## 快速开始

使用 Linux 和 Python 3.12，在解压后的仓库根目录运行以下命令。
构建 Signatory 前需安装系统 FFmpeg 和 C++ 编译器。
GPU 训练需要与显卡驱动兼容的 PyTorch／CUDA 环境。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e depends/signatory --no-build-isolation
python examples/smoke_trace.py
```

示例在 CPU 上运行随机初始化的小模型，检查完整记忆路径是否输出有限值动作，
不测量任务成功率。环境与编译说明见[安装指南](../installation.md)。

## 使用自己的数据训练

准备包含同步 RGB 观测、机器人状态和动作的本地 LeRobot v3 数据集。
将 `data/local/task` 替换为实际目录。预处理写入独立输出目录，保留原始输入。

```bash
python data/process_dataset.py data/local/task \
  --output-dir data/local/task_trace --signature-depth 3

python scripts/train_policy.py --policy streaming_act \
  --defaults-path configs/trace_act.yaml --dataset data/local/task_trace

python scripts/train_policy.py --policy prism_diffusion \
  --defaults-path configs/trace_diffusion.yaml --dataset data/local/task_trace
```

两份配置使用完整 TRACE 计算流程、固定随机种子，并默认关闭实验追踪。
数据划分、归一化和 checkpoint 说明见[数据准备](../data.md)与[训练指南](../training.md)。

## 评估 checkpoint

```bash
python scripts/eval_policy.py --policy streaming_act \
  --policy-path outputs/train/trace-act/RUN/checkpoints/STEP/pretrained_model \
  --dataset data/local/task_trace
```

将 checkpoint 路径替换为实际保存目录。离线动作误差不等同于机器人任务成功率。
仿真评估需要安装相应环境和资产，详见[评估指南](../evaluation.md)。

## 代码导航

| 路径 | 用途 |
| --- | --- |
| `policy/lerobot_policy_streaming_act/src/` | TRACE 记忆与 ACT 接口 |
| `policy/lerobot_policy_prism_diffusion/src/` | TRACE 记忆与 Diffusion Policy 接口 |
| `configs/` | 公开的主方法训练配置 |
| `scripts/` | 训练、评估及数据工具 |
| `data/` | LeRobot 格式转换与路径签名预处理 |
| `examples/` | 可运行的 CPU 示例 |
| `tests/` | 主方法、数据及发布检查 |
| `docs/` | 安装、方法、训练、评估与翻译文档 |
| `depends/signatory/` | 签名计算库源码及其原始许可 |

`streaming_act` 和 `prism_diffusion` 是两种 TRACE 接口沿用的序列化策略名称，
用于保持完整主方法 checkpoint 参数名的稳定性。
对应关系及兼容条件见[架构说明](../architecture.md)。

## 开发

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python scripts/check_release.py
python -m pytest
```

如只需运行不依赖模型环境的轻量 CPU 工具检查，安装 `requirements-test.txt` 后运行
`python -m pytest --lightweight`。开发流程见[贡献指南](../../CONTRIBUTING.md)。

## 引用

```bibtex
@inproceedings{li2026trace,
  title     = {TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation},
  author    = {Li, Zihao and Qiu, Ranpeng and Chen, Yincong and Ren, Guoqiang and Zhi, Weiming},
  booktitle = {Conference on Robot Learning},
  year      = {2026},
  url       = {https://jeong-zju.github.io/trace/}
}
```

## 许可与致谢

TRACE 原创贡献采用 [Apache-2.0](../../LICENSE) 许可。
本项目基于 LeRobot、ACT、Diffusion Policy 和 Signatory，
原始署名及许可保留在[第三方声明](../../THIRD_PARTY_NOTICES.md)中。
外部数据与权重遵循各自的许可。
