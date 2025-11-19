# Invest Sim

Invest Sim 是一个可配置的投资组合模拟工具，支持多资产类别、定投计划以及与目标风险的比较。

## 特性

- 📈 支持股票、债券、现金等多种资产类别
- 🧠 内置多种策略：固定权重、目标风险、自适应再平衡
- 🎯 Monte Carlo 模拟支持，输出收益区间
- 🛠️ 基于 `pydantic` 的配置验证，配置更安全
- 🖥️ CLI 入口，快速运行模拟并输出报告

## 快速开始

### 1. 设置虚拟环境

**Windows (PowerShell):**
```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\Activate.ps1

# 升级 pip
python -m pip install --upgrade pip

# 安装所有依赖（包括开发工具）
pip install -r requirements-dev.txt

# 以可编辑模式安装项目
pip install -e .
```

**Linux / macOS:**
```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 升级 pip
python -m pip install --upgrade pip

# 安装所有依赖（包括开发工具）
pip install -r requirements-dev.txt

# 以可编辑模式安装项目
pip install -e .
```

> 📖 详细的环境设置说明请参考 [环境设置指南](docs/ENVIRONMENT_SETUP.md)

### 2. 运行示例

```bash
# 运行前瞻性模拟
invest-sim forward --config examples/balanced.json

# 运行历史回测
invest-sim backtest --config examples/backtest_balanced.json --data data/sample_prices.csv

# 运行测试
pytest
```

## 配置说明

配置文件使用 JSON 或 YAML（需手动安装 `pyyaml`）描述资产与策略，具体字段见 `invest_sim/config.py` 中的模型定义。

## 目录结构

```
invest-sim/
├── invest_sim/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── data_models.py
│   ├── report.py
│   ├── simulator.py
│   └── strategies.py
├── examples/
│   └── balanced.json
├── tests/
│   └── test_simulator.py
├── pyproject.toml
└── README.md
```

## 📚 文档

- [环境设置指南](docs/ENVIRONMENT_SETUP.md) - 详细的虚拟环境和依赖安装说明
- [回测演示指南](docs/BACKTEST_DEMO_GUIDE.md) - 如何使用回测功能
- [回测框架设计](docs/BACKTEST_FRAMEWORK.md) - 回测框架的技术设计文档

## 📦 依赖管理

项目使用 `requirements.txt` 和 `requirements-dev.txt` 管理依赖：

- **requirements.txt**: 基础运行时依赖
- **requirements-dev.txt**: 包含开发工具（测试框架等）

所有依赖也定义在 `pyproject.toml` 中，支持通过 `pip install -e .[dev]` 安装。

## TODO

- [x] 接入历史数据回测
- [ ] 增加更多风险指标
- [ ] 输出 HTML 报告
