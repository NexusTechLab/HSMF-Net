# ENVI-HSI Toolkit

高光谱图像分析与批量处理桌面应用（PySide6 + NumPy/SciPy）。

![splash](docs/images/splash.png) <!-- TODO: 替换为真实截图 -->

---

## ✨ 主要特性

* **多格式数据加载**：支持 ENVI (`.hdr`+`.raw`)、TIFF、BMP 与批量文件夹导入。
* **交互式光谱查看**：点击像素即显示光谱曲线，支持光谱平滑/求导/归一化/标准化。
* **智能分割**：一键 Otsu / Watershed，对单粒籽粒自动分割、编号及统计。
* **批量导出**：ROI 光谱 (`CSV/MAT/NPY`)、分割掩膜 (`PNG`)、单粒 ENVI、论文级光谱图等。
* **科研分析**：健康等级预测、数据质量诊断、2D-COS、分组对比等常用科研图表。
* **服务层架构**：业务逻辑与 UI 解耦，方便二次开发与单元测试。
* **一键打包**：`scripts/build_exe.py` 基于 PyInstaller 输出单文件 `.exe`；Inno Setup 脚本生成安装包。

---

## 📂 目录结构（简化）

```text
ENVI-HSI-Toolkit/
├─ src/hsi_toolkit/        # 应用源代码
│  ├─ main.py              # GUI 入口
│  ├─ ui/                  # 界面组件
│  ├─ services/            # 业务逻辑
│  ├─ modules/             # 图像/光谱处理算法
│  └─ utils/               # 工具函数、日志等
├─ resources/              # 图标、QSS 等静态资源
├─ scripts/                # 打包 & 运维脚本
├─ tests/                  # pytest 单元测试
├─ docs/                   # 用户文档
└─ README.md               # 当前文件
```

完整结构请见仓库根目录的 `project_layout`。开发期间建议遵循 [src-layout](https://blog.jaraco.com/you-should-use-src-layout/) 以避免命名冲突。

---

## 🚀 快速开始

1. 克隆仓库并创建虚拟环境：
   ```bash
   git clone https://github.com/yourname/envi-hsi-toolkit.git
   cd envi-hsi-toolkit
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行：
   ```bash
   python -m hsi_toolkit.main  # 或 python src/hsi_toolkit/main.py
   ```

---

## 🏗️ 构建与打包

### 1. 仅生成可执行文件
```bash
python scripts/build_exe.py   # Windows 下生成 dist/ENVI_HSI.exe
```

### 2. 生成安装包 (Inno Setup)
1. 安装 [Inno Setup](https://jrsoftware.org/isinfo.php)。
2. 打开 `scripts/build_installer.iss`，检查 `AppVersion`、`Source` 路径。
3. 编译 (`Ctrl+F9`) 即输出 `Output/ENVI_HSI_Installer.exe`。

### 3. 代码签名（可选）
```powershell
signtool sign /fd sha256 /a /tr http://timestamp.digicert.com dist/ENVI_HSI.exe
signtool sign /fd sha256 /a /tr http://timestamp.digicert.com Output/ENVI_HSI_Installer.exe
```

---

## ⚙️ 配置
`config.py` 暴露以下关键参数：
* `APP_NAME`, `APP_VERSION`
* `UI["window_title"]`, `UI["window_size"]`
* `ensure_dirs()` —— 创建 `~/.cache/hsi_toolkit/`、日志目录等。

---

## 🧪 测试
```bash
pytest -q
```

---

## 🤝 贡献指南
1. Fork & PR，分支命名 `feature/xxx` / `bugfix/xxx`。
2. 提交前请运行 `ruff --fix . && black .`，保证代码格式。
3. 附带单元测试覆盖新功能。

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
