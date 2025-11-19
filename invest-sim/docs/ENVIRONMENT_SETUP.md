# 环境设置指南

本指南将帮助您为 invest-sim 项目设置 Python 虚拟环境，并安装所有必要的依赖。

## 📋 前置要求

- Python 3.10 或更高版本
- pip（Python 包管理器，通常随 Python 一起安装）

## 🚀 快速开始

### Windows (PowerShell)

```powershell
# 1. 进入项目目录
cd invest-sim

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
.venv\Scripts\Activate.ps1

# 4. 升级 pip
python -m pip install --upgrade pip

# 5. 安装所有依赖（包括开发工具）
pip install -r requirements-dev.txt

# 6. 以可编辑模式安装项目本身
pip install -e .
```

### Windows (CMD)

```cmd
# 1. 进入项目目录
cd invest-sim

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
.venv\Scripts\activate.bat

# 4. 升级 pip
python -m pip install --upgrade pip

# 5. 安装所有依赖（包括开发工具）
pip install -r requirements-dev.txt

# 6. 以可编辑模式安装项目本身
pip install -e .
```

### Linux / macOS

```bash
# 1. 进入项目目录
cd invest-sim

# 2. 创建虚拟环境
python3 -m venv .venv

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 升级 pip
python -m pip install --upgrade pip

# 5. 安装所有依赖（包括开发工具）
pip install -r requirements-dev.txt

# 6. 以可编辑模式安装项目本身
pip install -e .
```

## 📦 依赖说明

### 基础依赖 (`requirements.txt`)

- **pydantic>=2.5**: 数据验证和配置管理
- **numpy>=1.26**: 数值计算
- **pandas>=2.1**: 数据处理和分析
- **matplotlib>=3.8**: 数据可视化
- **rich>=13.6**: 终端美化输出

### 开发依赖 (`requirements-dev.txt`)

包含所有基础依赖，外加：
- **pytest>=7.4**: 单元测试框架
- **pytest-cov>=4.1**: 测试覆盖率工具

## ✅ 验证安装

安装完成后，您可以运行以下命令验证安装是否成功：

```bash
# 检查 invest-sim 命令是否可用
invest-sim --help

# 运行测试
pytest

# 检查已安装的包
pip list
```

## 🔄 日常使用

### 激活虚拟环境

每次使用项目前，需要先激活虚拟环境：

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

激活后，您的终端提示符前会显示 `(.venv)`，表示虚拟环境已激活。

### 停用虚拟环境

当您完成工作后，可以停用虚拟环境：

```bash
deactivate
```

## 📝 使用 requirements.txt 安装依赖

如果您只需要安装基础依赖（不包含开发工具），可以使用：

```bash
pip install -r requirements.txt
```

如果您需要开发工具（测试等），使用：

```bash
pip install -r requirements-dev.txt
```

## 🔧 常见问题

### 1. PowerShell 执行策略错误

如果遇到 "无法加载文件，因为在此系统上禁止运行脚本" 的错误，可以运行：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. 虚拟环境已存在

如果 `.venv` 目录已存在，您可以：
- 删除旧环境：`Remove-Item -Recurse -Force .venv` (PowerShell) 或 `rm -rf .venv` (Linux/macOS)
- 然后重新创建虚拟环境

### 3. Python 版本不匹配

确保使用 Python 3.10 或更高版本：

```bash
python --version
```

如果版本过低，请升级 Python 或使用 `python3` 命令。

### 4. pip 安装失败

如果遇到网络问题，可以尝试使用国内镜像源：

```bash
pip install -r requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 📚 相关文档

- [项目 README](../README.md)
- [回测演示指南](BACKTEST_DEMO_GUIDE.md)
- [回测框架设计](BACKTEST_FRAMEWORK.md)

## 💡 提示

- 虚拟环境目录 `.venv` 已添加到 `.gitignore`，不会被提交到版本控制
- 建议每次开始工作前都激活虚拟环境
- 如果添加了新的依赖，记得更新 `requirements.txt` 或 `requirements-dev.txt`
- 可以使用 `pip freeze > requirements.txt` 导出当前环境的精确版本（但建议手动维护版本范围）




