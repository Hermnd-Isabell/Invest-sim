# Invest Sim：投资组合模拟与回测套件

Invest Sim 是一个集 **前瞻性 Monte Carlo 模拟**、**历史回测**、**可视化报表**、**交互式仪表板** 于一体的量化研究工具箱。项目既可作为 Python 包集成到你的量化研究流程，也可以通过 CLI / Streamlit UI 直接运行。

## 🎯 核心能力总览

- **双引擎**：`ForwardSimulator` 负责 Monte Carlo 前瞻模拟，`Backtester` 负责历史行情回放，两套算法共用一套策略接口。
- **策略工厂**：`strategies.py` 定义固定权重、目标风险、自适应三类策略，可通过配置文件热切换。
- **强校验配置**：所有配置均通过 `pydantic` 数据模型验证，提前发现输入错误。
- **可解释输出**：`report.py` 将结果转为 Rich 表格/Matplotlib 图；`output/` 收集的图片和 CSV 可直接分享。
- **可视化体验**：`app/app.py` 基于 Streamlit，提供面向高净值客户的深色系交互面板。

## 🆕 最新进展（50ETF 期权）

- **真实 50ETF 合约回放**：仅展示/使用具备真实价格的可回放合约（至少 1 条非 NaN 收盘价），防止“空合约”误选。
- **下拉框一致性**：到期日/行权价直接来源于可回放合约集，避免缓存导致的错误选项与误配合约。
- **策略腿符号正确性**：`OptionLeg` 使用人类可读 `symbol`，价格查询走 `order_book_id`，确保定价与展示一致。
- **历史回放校验**：在运行前强制检查所选合约的历史价格列非空，否则给出清晰错误提示。
- **缓存与性能**：`OptionDataStore` 单例缓存，过期日/行权价/HV/IV 结果缓存，大幅减少重复加载与计算。
- **调试降噪**：移除多余的 `print`/`st.write` 调试输出，保留必要的错误与提示信息。

> 数据未随仓库提交，需按下方“复现步骤”放置到 `invest-sim/data/50ETF/`。数据结构与字段详见 `50ETF_DATA_README.md`。

## 🧱 模块与代码职责

| 模块 | 用途 | 实现要点 |
| --- | --- | --- |
| `invest_sim/data_models.py` | 定义资产、定投、策略、模拟/回测配置的 Pydantic 模型 | 自动归一化权重、校验期限、提供 `normalized_weights` 等便捷属性 |
| `invest_sim/config.py` | 统一从 JSON/YAML 读取配置 | `load_config`/`load_backtest_config` 自动选择模型并抛出清晰错误 |
| `invest_sim/strategies.py` | 策略抽象基类与三种实现 | 通过 `_normalize` 保证权重合法，`TargetRiskStrategy` 依据波动率缩放权重，`AdaptiveRebalanceStrategy` 只在偏离阈值时调仓 |
| `invest_sim/forward_simulator.py` | Monte Carlo 引擎 | 将年化收益/波动转为月度参数，注入定投金额，按再平衡频率调用策略；结果对象提供分位数、终值分布、VaR/CVaR、最大回撤等接口 |
| `invest_sim/backtester.py` | 历史回测引擎 | 读取价格序列，按期注入现金、应用真实收益，再由策略适配器完成再平衡；结果对象可计算 Sharpe、波动率、最大回撤等指标并输出权重/净值序列 |
| `invest_sim/data_loader.py` | 历史数据加载与收益率计算 | 支持 CSV/Excel/Parquet，自动解析日期、排序并抽取数值列，可切换简单/对数收益率 |
| `invest_sim/report.py` | 结果汇总与图表生成 | Rich 表格展示关键指标，Matplotlib 输出组合价值、权重、收益分布、回撤等图像 |
| `invest_sim/cli.py` | 命令行入口 | 基于 argparse，提供 `forward` 与 `backtest` 子命令；集成配置加载、数据校验、结果渲染与图表导出 |
| `app/app.py` | Streamlit 豪华风仪表板 | 通过 Plotly 绘制历史与 Monte Carlo 扇形图，展示 Sharpe/波动/回撤等核心指标，含自定义 CSS 主题 |
| `scripts/*.py` | 实用脚本 | `generate_sample_data.py` 快速造数，`run_backtest_demo.py` 一键运行示例回测 |
| `tests/test_simulator.py` | 单测示例 | 覆盖模拟器核心流程，可作为扩展测试的模板 |

> 📄 更多背景与设计思路：`docs/ENVIRONMENT_SETUP.md`、`docs/BACKTEST_DEMO_GUIDE.md`、`docs/BACKTEST_FRAMEWORK.md`、`SETUP_QUICKSTART.md`

## 🗂️ 目录结构

```
invest-sim/
├── app/                     # Streamlit 前端
├── docs/                    # 文字文档
├── examples/                # JSON 配置样例
├── invest_sim/              # 核心 Python 包
├── output/                  # 样例图表与报告
├── scripts/                 # 工具脚本
├── tests/                   # 单元测试
├── data/sample_prices.csv   # 演示数据
└── README.md
```

## ⚙️ 安装与环境

### 1. 创建虚拟环境

**Windows (PowerShell)**
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

> 完整环境排障指南请看 `docs/ENVIRONMENT_SETUP.md`

## 🚀 快速体验

```bash
# 运行 Monte Carlo 前瞻模拟
invest-sim forward \
  --config examples/balanced.json \
  --seed 42 \
  --quantiles 0.1,0.5,0.9 \
  --chart output/forward_balanced.png

# 运行历史回测
invest-sim backtest \
  --config examples/backtest_balanced.json \
  --data data/sample_prices.csv \
  --risk-free-rate 0.02 \
  --chart output/backtest_balanced.png

# 运行单元测试
pytest
```

### Streamlit 仪表板

```bash
cd app
streamlit run app.py
```

选择不同策略和杠杆，仪表板会实时刷新历史/未来曲线、风险指标及 Monte Carlo 区间。

### 复现 50ETF 期权功能（含历史回放）

1. 准备数据（本仓库未包含）：将 `Filtered_OptionInstruments_510050.pkl` 与 `Filtered_OptionPrice_2020_2022.feather` 放入 `invest-sim/data/50ETF/` 目录，字段说明见 `50ETF_DATA_README.md`。
2. 安装依赖：在仓库根目录执行 `pip install -r requirements-dev.txt`，再 `pip install -e .`。
3. 运行 UI：`cd app && streamlit run app.py`，进入 Market View / Derivatives Lab。
4. 选择合约：到期日、Call/Put、行权价下拉框仅展示可回放合约；若选中无价格的合约会提示错误。
5. 历史回放：选择“Historical Replay (50ETF)”即可复用真实价格路径；若无数据会自动阻止运行并提示。

## 🧾 配置文件详解

- `examples/*.json` 给出不同风险偏好的资产组合（保守/平衡/进取/自适应等）。
- 所有字段定义、默认值及校验逻辑位于 `invest_sim/data_models.py`，通过 `load_config()` / `load_backtest_config()` 读入。
- 重点字段：
  - `assets`：每个资产包含 `expected_return`、`volatility`、`weight`
  - `contribution_plan`：`annual_contribution` + `frequency` 自动换算为 `periodic_contribution`
  - `strategy`：`name` + `target_volatility` / `rebalance_threshold` 定义策略行为
  - `rebalance_frequency`：在模拟中默认按月，在回测中按交易日

## 📊 结果输出与可视化

- CLI 运行后会在终端输出 Rich 表格，并在 `output/`（或指定路径）生成：
  - `portfolio_overview.png`：资产价值走势 + 权重变化
  - `returns_distribution.png`：收益率直方图
  - `drawdown.png`：回撤填充图
  - `strategy_comparison.csv`：多策略关键指标对比
- `report.py` 还提供 `render_*` 方法，可在外部 Notebook/脚本中复用。

## 🧪 测试与扩展

- `pytest` 将运行 `tests/` 下的用例，可作为构建更多策略、风控指标时的模板。
- 如果需要引入新策略：
  1. 在 `strategies.py` 中继承 `Strategy` 并实现 `rebalance`
  2. 在 `build_strategy` 中注册名称
  3. 在配置文件中引用该名称

## 🤝 贡献指南

1. Fork & Clone
2. 创建特性分支
3. 补充/更新测试与文档
4. 提交 PR，附上运行指令和截图（若包含可视化）

## 📌 TODO

- [x] 接入历史数据回测
- [x] 生成示例图表和报告
- [ ] 增加更多风险指标（如 Sortino、Calmar）
- [ ] 输出交互式 HTML 报告

---

> 💬 有任何想法或想接入真实券商 API？欢迎开启 Issue！
