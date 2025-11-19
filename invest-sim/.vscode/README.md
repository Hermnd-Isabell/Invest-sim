# Cursor/VS Code 配置说明

本目录包含项目的 Cursor/VS Code 配置文件。

## 📁 配置文件说明

### `settings.json`
- **Python 解释器**: 自动指向项目的虚拟环境 `.venv`
- **测试框架**: 配置为使用 pytest
- **终端**: 自动激活虚拟环境
- **代码分析**: 启用基本类型检查

### `launch.json`
提供以下调试配置：
1. **Python: 当前文件** - 调试当前打开的 Python 文件
2. **invest-sim: Forward Simulation** - 运行前瞻性模拟
3. **invest-sim: Backtest** - 运行历史回测
4. **Python: Pytest** - 运行测试套件

### `extensions.json`
推荐安装的扩展：
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Debugpy (ms-python.debugpy)

## 🚀 使用方法

### 1. 选择 Python 解释器
1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 "Python: Select Interpreter"
3. 选择 `.venv\Scripts\python.exe`

或者直接使用已配置的默认解释器（已在 `settings.json` 中设置）。

### 2. 使用调试配置
1. 按 `F5` 或点击左侧调试图标
2. 从下拉菜单中选择要运行的配置
3. 点击运行按钮

### 3. 运行测试
- 在测试文件中点击测试上方的 "Run Test" 按钮
- 或使用调试配置 "Python: Pytest"

## ⚙️ 注意事项

- 这些配置文件在 `.gitignore` 中，不会被提交到 git
- 如果团队成员需要相同配置，可以手动复制这些文件
- 确保虚拟环境已创建并安装了所有依赖




