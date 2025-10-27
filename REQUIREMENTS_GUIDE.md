# Requirements 文件说明

本项目提供了三个不同的 requirements 文件，适用于不同场景。

## 📦 文件说明

### 1. `requirements.txt` - 标准版（推荐）
**用途**：日常训练和部署
**内容**：核心依赖 + 常用工具

```bash
pip install -r requirements.txt
```

**包含的主要包：**
- PyTorch 1.12.1 + TorchVision 0.13.1
- NumPy, SciPy, Scikit-learn
- OpenCV, Pillow, Scikit-image
- TensorBoard（训练监控）
- Click, tqdm（命令行工具）

---

### 2. `requirements_minimal.txt` - 最小版
**用途**：只进行训练，不需要额外功能
**内容**：仅核心训练依赖

```bash
pip install -r requirements_minimal.txt
```

**适合场景：**
- 空间有限的环境
- 只需要基本训练功能
- Docker 镜像（减小体积）

---

### 3. `requirements_full.txt` - 完整版
**用途**：包含所有依赖（与 conda 环境一致）
**内容**：所有已安装的包

```bash
pip install -r requirements_full.txt
```

**额外包含：**
- Hugging Face Hub（模型上传）
- Google Auth（云存储）
- Dask（分布式计算）
- 所有间接依赖

---

## 🚀 快速开始

### 方法 1：使用 Conda（推荐）

```bash
# 创建新环境
conda create -n simplenet python=3.8

# 激活环境
conda activate simplenet

# 安装 PyTorch（带 CUDA 11.3）
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch

# 安装其他依赖
pip install -r requirements.txt
```

### 方法 2：仅使用 pip

```bash
# 创建虚拟环境
python3.8 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装所有依赖
pip install -r requirements.txt
```

### 方法 3：在现有环境安装

```bash
# 直接安装
pip install -r requirements.txt
```

---

## 📋 从 Conda 环境导出 Requirements

### 导出当前环境的所有包

**方法 1：使用 conda（推荐用于 conda 环境）**
```bash
# 导出完整环境（包括 conda 和 pip 包）
conda env export > environment.yml

# 只导出手动安装的包（更简洁）
conda env export --from-history > environment_minimal.yml

# 跨平台兼容
conda env export --no-builds > environment_cross_platform.yml
```

**方法 2：使用 pip**
```bash
# 导出所有 pip 包（包括 conda 安装的）
pip freeze > requirements_freeze.txt

# 只导出顶层包（不含依赖）
pip list --not-required --format=freeze > requirements_top_level.txt
```

**方法 3：手动筛选（推荐）**
```bash
# 查看所有包
conda list

# 或
pip list

# 手动创建 requirements.txt，只包含必要的包
# 让 pip 自动解析依赖关系
```

---

## 🎯 推荐做法

### 场景 1：新建项目
1. 手动编写 `requirements.txt`，只列出主要依赖
2. 不要用 `pip freeze`（会包含所有依赖）
3. 不指定过于严格的版本号

```txt
# 好的做法
torch>=1.12.0,<2.0
numpy>=1.22.0
scikit-learn

# 避免
torch==1.12.1+cu113  # 过于具体
```

### 场景 2：复现环境
1. 使用 `pip freeze` 导出精确版本
2. 或使用 `conda env export`

```bash
# 精确复现
pip freeze > requirements_exact.txt
conda env export > environment_exact.yml
```

### 场景 3：跨平台部署
1. 不包含平台特定的构建信息
2. 使用版本范围而不是精确版本

```bash
# Conda 跨平台导出
conda env export --no-builds > environment.yml
```

---

## 🔍 版本对照表

基于你的 conda 环境，这里是主要包的版本：

| 包名 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.8.15 | 运行环境 |
| **torch** | 1.12.1 | 深度学习框架 |
| **torchvision** | 0.13.1 | 图像处理 |
| **cudatoolkit** | 11.3.1 | GPU 加速 |
| **numpy** | 1.22.4 | 数值计算 |
| **scipy** | 1.9.1 | 科学计算 |
| **scikit-learn** | 1.3.2 | 机器学习 |
| **scikit-image** | 0.20.0 | 图像处理 |
| **opencv** | 4.5.1 | 计算机视觉 |
| **pillow** | 10.4.0 | 图像 I/O |
| **pandas** | 2.0.3 | 数据处理 |
| **matplotlib** | 3.7.3 | 可视化 |
| **tensorboard** | 2.11.2 | 训练监控 |
| **timm** | 1.0.11 | 预训练模型 |
| **click** | 8.1.7 | CLI 工具 |
| **tqdm** | 4.67.1 | 进度条 |

---

## ⚠️ 注意事项

### CUDA 版本
当前环境使用 **CUDA 11.3**。如果你的 GPU 支持不同的 CUDA 版本：

```bash
# 查看系统 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
# 访问：https://pytorch.org/get-started/locally/

# 例如 CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# 例如 CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### OpenCV 版本
- Conda 环境使用 `opencv=4.5.1`（包含 GUI 支持）
- pip 环境应使用 `opencv-python==4.5.1`（无 GUI）
- 如需完整功能：`opencv-contrib-python==4.5.1`

### Python 版本兼容性
- 推荐：Python 3.8 - 3.10
- 不支持：Python 3.11+（部分包不兼容）

---

## 🔧 故障排查

### 问题 1: CUDA 版本不匹配
```bash
# 错误: CUDA driver version is insufficient
# 解决: 安装 CPU 版本或更新 CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 问题 2: OpenCV 导入错误
```bash
# 错误: ImportError: libGL.so.1
# 解决: 安装系统依赖
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### 问题 3: NumPy 版本冲突
```bash
# 错误: numpy version mismatch
# 解决: 重新安装 numpy
pip install --force-reinstall numpy==1.22.4
```

### 问题 4: 包安装失败
```bash
# 尝试升级 pip
pip install --upgrade pip setuptools wheel

# 清理缓存
pip cache purge

# 使用国内镜像（中国用户）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 📝 如何维护 Requirements

### 添加新包
1. 先在环境中测试安装
2. 确认可用后添加到 `requirements.txt`
3. 只添加主要包，不添加依赖

```bash
# 测试安装
pip install new-package

# 确认版本
pip show new-package

# 添加到 requirements.txt
echo "new-package==x.y.z" >> requirements.txt
```

### 更新包版本
```bash
# 查看可更新的包
pip list --outdated

# 更新特定包
pip install --upgrade package-name

# 更新 requirements.txt
pip freeze | grep package-name
```

### 定期清理
```bash
# 查看未被依赖的包
pip list --not-required

# 移除不需要的包
pip uninstall package-name
```

---

## 🌐 国内用户加速

### 使用清华镜像
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 配置默认镜像
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Conda 镜像配置
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

---

## 📚 相关资源

- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [pip requirements 文档](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
- [Conda 环境管理](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

---

最后更新：2025-10-25
