# 快速设置指南

## 🚀 一键设置（Windows PowerShell）

```powershell
# 进入项目目录
cd invest-sim

# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1

# 安装所有依赖
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

## ✅ 验证安装

```powershell
# 检查命令是否可用
invest-sim --help

# 检查包导入
python -c "import invest_sim; print('✅ 安装成功')"
```

## 📝 日常使用

**每次使用前激活虚拟环境：**
```powershell
.venv\Scripts\Activate.ps1
```

**使用完成后停用：**
```powershell
deactivate
```

## 📚 更多信息

- 详细设置说明：[docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md)
- 项目 README：[README.md](README.md)




