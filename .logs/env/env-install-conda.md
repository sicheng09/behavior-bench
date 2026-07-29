# BehaviorBench 环境安装文档（conda）

本文档面向 BehaviorBench 仓库根目录。不同机器上的绝对路径可能不同，下面命令建议先进入仓库根目录，再用 `REPO_ROOT="$(pwd)"` 派生数据路径：

```bash
cd /path/to/behavior-bench
export REPO_ROOT="$(pwd)"
```

项目是基于 PufferDrive/PufferLib 的 Python 包，包含 C/C++ 扩展和 PyTorch 扩展。当前机器有 NVIDIA H20 GPU，`nvidia-smi` 显示驱动版本 `570.133.20`，最高支持 CUDA `12.8`。本文档使用 Conda/Miniforge 管理 Python 解释器和基础环境，再在 Conda 环境内用 `python -m pip` 安装 PyTorch、PyG 和本项目依赖。

## 1. 环境概览

- Python 版本：`pyproject.toml` 要求 `>=3.9`。默认推荐 `Python 3.11`；如果 `waymo-open-dataset-tf-2-11-0` 必须安装在同一类流程中，建议单独创建 `Python 3.10` 转换环境。
- 包管理器：主流程使用 `conda` 创建和激活环境；Python 包安装阶段使用 `python -m pip`，避免在同一个环境中让 Conda 和 pip 同时解析同一批 Python 依赖。
- GPU 信息：`NVIDIA H20`，显存约 `97871 MiB`，驱动 `570.133.20`，`nvidia-smi` 报告 CUDA `12.8`。
- PyTorch CUDA wheel：当前驱动可以优先尝试 `cu128` wheel；如不匹配，再退到 `cu126` 或 `cu121`。
- 本地扩展：
  - `pufferlib._C`：PyTorch C++/CUDA 扩展。
  - `pufferlib/ocean/drive/binding*.so`：Drive 环境 C 扩展。
- 重要环境变量：
  - `DRIVE_BINARIES_DATA_ROOT`：训练/评估用地图二进制文件根目录。
  - `NUPLAN_DATA_ROOT`、`NUPLAN_MAPS_ROOT`：仅 nuPlan 集成需要。
- 可选构建开关：
  - `NO_OCEAN=1`：跳过 Ocean/Drive C 扩展。
  - `NO_TRAIN=1`：跳过 PyTorch 训练扩展。
  - `DEBUG=1`：用调试参数构建扩展。

## 2. 安装系统依赖

当前机器是 Linux。若你有 `sudo` 权限，先安装编译工具、Git、下载工具和常见图形/链接库。

### AlmaLinux / RHEL / CentOS

```bash
sudo dnf install -y \
  gcc gcc-c++ make git curl wget tar unzip which \
  python3-devel libgomp \
  mesa-libGL mesa-libGL-devel \
  libX11-devel libXrandr-devel libXinerama-devel libXi-devel libXcursor-devel
```

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential git curl wget tar unzip python3-dev \
  libgomp1 libgl1 libgl1-mesa-dev \
  libx11-dev libxrandr-dev libxinerama-dev libxi-dev libxcursor-dev
```

如果你没有 `sudo` 权限，请确认集群或容器镜像中至少已有：

```bash
gcc --version
g++ --version
make --version
git --version
```

## 3. 安装 Conda / Miniforge

如果系统中已经有可用的 `conda`，可以跳过本节。推荐使用 Miniforge，因为默认走 `conda-forge`，包比较新。

```bash
cd /path/to/behavior-bench

wget -O Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda --version
```

如果希望以后打开 shell 自动支持 `conda activate`：

```bash
conda init bash
```

执行 `conda init` 后需要重新打开 shell，或者先临时执行：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

## 4. 创建 Conda 环境

进入项目目录：

```bash
cd /path/to/behavior-bench
```

创建 Python 3.11 环境：

```bash
conda create -n behavior-bench python=3.11 pip -y
conda activate behavior-bench
python -V
which python
```

配置推荐 channel：

```bash
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
```

安装基础构建依赖。建议这一步用 Conda 安装，之后 Python 包主要交给 pip：

```bash
conda install -y pip setuptools wheel cython "numpy<2.0"
python -m pip install -U pip setuptools wheel
```

如果需要让本项目的 CUDA extension 也用 `nvcc` 编译，可以额外准备 CUDA Toolkit。若系统或集群模块已经提供 `nvcc`，优先使用系统版本；否则可尝试 Conda 的 CUDA Toolkit：

```bash
# 可选：仅在需要 nvcc 编译 CUDA extension 时执行
conda install -y -c nvidia cuda-toolkit=12.8
which nvcc || true
nvcc --version || true
```

说明：PyTorch CUDA wheel 自带运行时库，但通常不提供 `nvcc`。没有 `nvcc` 时仍可使用 GPU 训练，只是本项目的 CUDA extension 可能不会编译为 CUDA 版本。

## 5. 安装 PyTorch

根据机器环境选择 CPU 或 CUDA 版本。二选一即可。

### 方案 A：CPU 环境

适合没有 NVIDIA GPU、没有 CUDA 驱动，或只做轻量验证。

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 方案 B：CUDA 环境

先确认驱动和 CUDA 编译器状态：

```bash
nvidia-smi
which nvcc || true
```

当前 `nvidia-smi` 显示驱动支持 CUDA `12.8`，优先使用 PyTorch 的 CUDA 12.8 wheel：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

如果 `cu128` wheel 与你选择的 PyTorch 版本不匹配，再退一步使用 `cu126` 或 `cu121`：

```bash
# 备选 1：CUDA 12.6 wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

# 备选 2：CUDA 12.1 wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

验证 PyTorch：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
PY
```

## 6. 预装 PyG 相关依赖

项目依赖 `torch-geometric==2.7.0` 和 `torch-cluster==1.6.3`。`torch-cluster` 强依赖当前 PyTorch/CUDA 组合，建议先从 PyG wheel 源安装，避免本地源码编译失败。

```bash
TORCH_MM=$(python - <<'PY'
import torch
v = torch.__version__.split("+")[0].split(".")
print(f"{v[0]}.{v[1]}.0")
PY
)

CUDA_TAG=$(python - <<'PY'
import torch
print("cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", ""))
PY
)

echo "PyG wheel index: https://data.pyg.org/whl/torch-${TORCH_MM}+${CUDA_TAG}.html"
python -m pip install "torch-cluster==1.6.3" -f "https://data.pyg.org/whl/torch-${TORCH_MM}+${CUDA_TAG}.html"
python -m pip install "torch-geometric==2.7.0"
```

如果这里提示没有匹配 wheel，打印当前 PyTorch 信息后，到 `https://data.pyg.org/whl/` 选择相同 PyTorch 主次版本与 CPU/CUDA 标签：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

## 7. 安装 BehaviorBench / PufferLib

推荐使用 editable install，便于后续修改源码后立即生效。注意：当前 `setup.py` 的依赖元数据存在一个已知冲突：

- `pufferlib==3.0.0` 写死依赖 `tensorflow==2.20.0`
- 同时又依赖 `waymo-open-dataset-tf-2-11-0==1.6.1`
- `waymo-open-dataset-tf-2-11-0==1.6.1` 写死依赖 `tensorflow==2.11`

因此不要直接执行：

```bash
python -m pip install -e . --no-build-isolation
```

否则 pip resolver 会因为 TensorFlow/Waymo 版本约束报依赖不可满足。主训练/评估流程通常消费已经转换好的 `.bin` 地图，并不需要在主环境中安装 Waymo TFRecord 包。推荐做法是先手动安装可共存依赖，再用 `--no-deps` 安装本地项目：

```bash
python -m pip install \
  "setuptools" \
  "numpy<2.0" \
  "shimmy[gym-v21]" \
  "gym==0.23" \
  "gymnasium==0.29.1" \
  "pettingzoo==1.24.1" \
  "tensorflow==2.20.0" \
  "scipy==1.17.0" \
  "pillow==11.3.0" \
  "msgpack==1.1.2" \
  "psutil" \
  "nvidia-ml-py" \
  "rich" \
  "rich_argparse" \
  "imageio" \
  "pyro-ppl" \
  "heavyball<2.0.0" \
  "neptune" \
  "wandb" \
  "matplotlib"

python -m pip install -e . --no-build-isolation --no-deps
```

如果你需要 CARLA 相关数据生成工具，初始化 submodule 并安装开发依赖：

```bash
git submodule update --init --recursive
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` 当前只包含：

```text
-e external/pyxodr
```

如果你确实需要处理原始 Waymo TFRecord，建议创建单独的转换环境，不要和主训练环境混装：

```bash
cd /path/to/behavior-bench

conda create -n behavior-bench-waymo python=3.10 pip -y
conda activate behavior-bench-waymo
python -m pip install "tensorflow==2.11.*" "waymo-open-dataset-tf-2-11-0==1.6.1"
```

转换出 `.bin` 后，回到主环境运行训练/评估：

```bash
conda activate behavior-bench
export DRIVE_BINARIES_DATA_ROOT=/path/to/binaries
```

## 8. 编译本地扩展

安装后建议显式重新编译一次扩展，确保 `.so` 与当前 Conda 环境匹配。

```bash
conda activate behavior-bench
cd /path/to/behavior-bench
python setup.py build_ext --inplace --force
```

构建过程会按需下载并解压：

- `raylib-5.5_linux_amd64`
- `raylib-5.5_webassembly`
- `box2d-linux-amd64`
- `box2d-web`
- `inih-r62`

如果只想跳过 Drive/Ocean C 扩展：

```bash
NO_OCEAN=1 python -m pip install -e . --no-build-isolation --no-deps
NO_OCEAN=1 python setup.py build_ext --inplace --force
```

如果只想跳过训练侧 PyTorch 扩展：

```bash
NO_TRAIN=1 python -m pip install -e . --no-build-isolation --no-deps
NO_TRAIN=1 python setup.py build_ext --inplace --force
```

调试构建：

```bash
DEBUG=1 python setup.py build_ext --inplace --force
```

## 9. 基础验证

先验证 Python 包和核心依赖：

```bash
python - <<'PY'
import sys
import numpy
import torch
import pufferlib

print("python:", sys.version)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("pufferlib import: ok")
PY
```

验证 PyTorch 扩展：

```bash
python - <<'PY'
import torch
import pufferlib._C
print("pufferlib._C import: ok")
PY
```

说明：`pufferlib._C` 链接了 PyTorch 自带的 `libc10.so`、`libtorch_cpu.so` 等动态库。先 `import torch` 会加载这些库；如果你需要在不先导入 `torch` 的情况下直接导入 `pufferlib._C`，请设置下方 13.4 节中的 `LD_LIBRARY_PATH`。

验证 Drive C binding：

```bash
python - <<'PY'
from pufferlib.ocean.drive import binding
print("drive binding import: ok")
print("MAX_OBS_PARTNERS:", binding.MAX_OBS_PARTNERS)
PY
```

验证 CLI：

```bash
puffer --help
```

## 10. 准备训练/评估数据

项目训练和评估依赖地图二进制文件。评估脚本要求：

```bash
export DRIVE_BINARIES_DATA_ROOT=/path/to/binaries
```

通常目录结构类似：

```text
$DRIVE_BINARIES_DATA_ROOT/
  training/
    map_000.bin
    map_001.bin
  validation/
    map_000.bin
  pufferhard/
    map_000.bin
```

### 10.1 安装 Hugging Face CLI

```bash
conda activate behavior-bench
python -m pip install -U "huggingface_hub[cli]"

# 如果数据集需要登录或你遇到权限错误，先执行：
# hf auth login
```

### 10.2 下载 10k WOMD training 数据

```bash
cd /path/to/behavior-bench
mkdir -p data/processed/training

hf download daphne-cornelisse/pufferdrive_womd_train \
  --repo-type dataset \
  --local-dir data/processed/training

# 下载结果是 training.zip；zip 内部路径已经是 data/processed/training/*.json
unzip -q -n data/processed/training/training.zip -d .

# 检查 JSON 数量，预期约 10000
python - <<'PY'
from pathlib import Path
print(len(list(Path("data/processed/training").rglob("*.json"))))
PY
```

### 10.3 下载 10k WOMD validation 数据

```bash
cd /path/to/behavior-bench
mkdir -p data/processed/validation

hf download daphne-cornelisse/pufferdrive_womd_val \
  --repo-type dataset \
  --local-dir data/processed/validation

# 下载结果是 pufferdrive_val.zip；zip 内部路径已经是 data/processed/validation/*.json
unzip -q -n data/processed/validation/pufferdrive_val.zip -d .

# 检查 JSON 数量，预期约 10000
python - <<'PY'
from pathlib import Path
print(len(list(Path("data/processed/validation").rglob("*.json"))))
PY
```

### 10.4 下载 WOMD + CARLA mixed 数据

如果你要跑混合数据训练，可下载 mixed 数据集。建议放到单独 split 名称下，避免覆盖 `training`。

```bash
cd /path/to/behavior-bench
mkdir -p data/processed/mixed

hf download daphne-cornelisse/pufferdrive_womd_train_carla_mixed \
  --repo-type dataset \
  --local-dir data/processed/mixed

# mixed 的 zip 内部路径是 data/processed/training/*.json。
# 为了保留 mixed split 名称，先解压到临时目录，再移动到 data/processed/mixed。
rm -rf /tmp/pufferdrive_mixed_extract
mkdir -p /tmp/pufferdrive_mixed_extract
unzip -q -n data/processed/mixed/training.zip -d /tmp/pufferdrive_mixed_extract

python - <<'PY'
from pathlib import Path
import shutil

src = Path("/tmp/pufferdrive_mixed_extract/data/processed/training")
dst = Path("data/processed/mixed")
dst.mkdir(parents=True, exist_ok=True)
for path in src.rglob("*.json"):
    shutil.move(str(path), dst / path.name)
print(len(list(dst.rglob("*.json"))))
PY

rm -rf /tmp/pufferdrive_mixed_extract
```

训练时指定 split：

```bash
puffer train puffer_drive --env.split mixed
```

### 10.5 下载 GPUDrive mini 或 full 数据

如果你想用 README 中提到的 GPUDrive mini/full 数据：

```bash
# Mini: 约 1000 training + 300 validation/testing
mkdir -p data/gpudrive_mini
hf download EMERGE-lab/GPUDrive_mini \
  --repo-type dataset \
  --local-dir data/gpudrive_mini

# Full: 约 100000 scenes，体积更大，确认磁盘空间后再下载
mkdir -p data/gpudrive_full
hf download EMERGE-lab/GPUDrive \
  --repo-type dataset \
  --local-dir data/gpudrive_full
```

仓库也提供了分组下载脚本，默认从 `EMERGE-lab/GPUDrive` 下载到 `data/processed`，并限制 training 约 10000 个文件：

```bash
python data_utils/womd/download_womd_data.py
```

### 10.6 将 JSON 转为 Drive 二进制地图

训练/评估读取的是 `.bin`，不是原始 JSON。当前 `pufferlib/ocean/drive/drive.py` 的入口会读取：

- `DRIVE_DATA_ROOT`：原始 JSON 根目录，下面应有 `training/`、`validation/`、`mixed/` 等 split 子目录。
- `DRIVE_BINARIES_DATA_ROOT`：输出 `.bin` 根目录，脚本会按 split 写入对应子目录。

推荐命令：

```bash
cd /path/to/behavior-bench
conda activate behavior-bench
export REPO_ROOT="$(pwd)"

export DRIVE_DATA_ROOT="$REPO_ROOT/data/processed"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"

python pufferlib/ocean/drive/drive.py
```

转换完成后检查：

```bash
find "$DRIVE_BINARIES_DATA_ROOT" -maxdepth 2 -type f -name "*.bin" | head
ls "$DRIVE_BINARIES_DATA_ROOT"
```

### 10.7 用 validation binaries 生成 1000-map benchmark split

如果已经有完整 validation binaries，并想生成 1000-map benchmark split：

```bash
cd /path/to/behavior-bench
export REPO_ROOT="$(pwd)"
mkdir -p data/eval_splits

# 生成 interactive1k
python scripts/remap_files.py \
  --source-path "$DRIVE_BINARIES_DATA_ROOT/validation" \
  --csv-path pufferlib/resources/splits/interactive1k.csv \
  --target-path data/eval_splits

# 生成 random1k
python scripts/remap_files.py \
  --source-path "$DRIVE_BINARIES_DATA_ROOT/validation" \
  --csv-path pufferlib/resources/splits/random1k.csv \
  --target-path data/eval_splits
```

使用 1k split 评估：

```bash
cd /path/to/behavior-bench
export REPO_ROOT="$(pwd)"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"

python pufferlib/ocean/benchmark/eval.py \
  --eval.split interactive1k \
  --map-ids all
```

### 10.8 保存常用数据环境变量

建议把常用环境变量写入本地 shell 配置或项目内自用脚本，例如：

```bash
cat > .env.local <<'EOF'
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DRIVE_DATA_ROOT="$REPO_ROOT/data/processed"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
EOF

source .env.local
```

注意：`.env.local` 不一定被 `.gitignore` 忽略，提交前请确认不要提交个人路径。

## 11. 运行 smoke test

如果已有地图二进制文件，可运行一个训练命令验证环境：

```bash
conda activate behavior-bench
export DRIVE_BINARIES_DATA_ROOT=/path/to/binaries
puffer train puffer_drive
```

如果没有 GPU 或想先用 CPU 跑通，可覆盖训练设备：

```bash
puffer train puffer_drive --train.device cpu
```

评估 smoke test：

```bash
export DRIVE_BINARIES_DATA_ROOT=/path/to/binaries
python pufferlib/ocean/benchmark/eval.py --map-ids 0-10
```

如果 PPO、SMART 或 Hybrid planner 默认使用 CUDA，而当前没有 GPU，可按 planner 类型指定 CPU，例如：

```bash
python pufferlib/ocean/benchmark/eval.py \
  --planner.type smart \
  --planner.smart.weights-path weights/smart_1M_epoch_029.pt \
  --planner.smart.device cpu \
  --map-ids 0-10
```

## 12. 可选：nuPlan 集成

nuPlan 不是标准 WOMD 训练/评估必需项。只有需要跑 nuPlan 闭环仿真时再安装。建议在 `behavior-bench` Conda 环境中安装，或按 nuPlan 自身要求创建单独环境。

```bash
conda activate behavior-bench
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
python -m pip install -e .
export NUPLAN_DEVKIT_ROOT=$(pwd)
export NUPLAN_DATA_ROOT=/path/to/nuplan/dataset
export NUPLAN_MAPS_ROOT=/path/to/nuplan/dataset/maps
```

回到 BehaviorBench 项目后注册 planner 配置：

```bash
cd /path/to/behavior-bench
cp pufferlib/nuplan_integration/config/simulation/planner/*.yaml \
  "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/config/simulation/planner/"
```

## 13. 常见问题与处理

### 13.1 `pip install -e .` 报 TensorFlow / Waymo 依赖冲突

如果看到类似错误：

```text
Because waymo-open-dataset-tf-2-11-0==1.6.1 depends on tensorflow==2.11 and
pufferlib==3.0.0 depends on tensorflow==2.20.0 ...
requirements are unsatisfiable.
```

根因是项目当前 `setup.py` 中同时写死了两个互斥依赖。主环境不要让 resolver 自动解析这组依赖，改用：

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

但执行这条命令前，请先按第 5、6、7 节安装 PyTorch、PyG 和可共存的运行依赖。若确实需要 `waymo-open-dataset-tf-2-11-0`，请使用第 7 节中的 `behavior-bench-waymo` 独立转换环境。

### 13.2 `torch` 或 `torch-cluster` 安装失败

优先确认 PyTorch 版本、CUDA 标签和 PyG wheel URL 是否匹配：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

然后重新安装匹配版本的 `torch-cluster`：

```bash
python -m pip uninstall -y torch-cluster
python -m pip install "torch-cluster==1.6.3" -f "https://data.pyg.org/whl/torch-${TORCH_MM}+${CUDA_TAG}.html"
```

如果 `TORCH_MM` 或 `CUDA_TAG` 已经丢失，重新执行第 6 节中的变量计算命令。

### 13.3 NumPy 版本冲突

项目显式要求 `numpy<2.0`。若 resolver 升级到了 NumPy 2.x，重新固定：

```bash
python -m pip install --force-reinstall "numpy<2.0"
python setup.py build_ext --inplace --force
```

### 13.4 `pufferlib.ocean.drive.binding` 导入失败

通常是 C 扩展没有编译成功，或 `.so` 与当前 Python/Conda 环境不匹配：

```bash
conda activate behavior-bench
cd /path/to/behavior-bench
python setup.py build_ext --inplace --force
python - <<'PY'
from pufferlib.ocean.drive import binding
print("ok")
PY
```

### 13.5 `pufferlib._C` 导入失败

如果报错是：

```text
ImportError: libc10.so: cannot open shared object file: No such file or directory
```

根因通常不是 `libc10.so` 缺失，而是动态链接器没有搜索到 PyTorch 的库目录。先确认：

```bash
conda activate behavior-bench
python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
ldd pufferlib/_C*.so
```

临时修复方式：

```bash
export LD_LIBRARY_PATH="$(python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
):${LD_LIBRARY_PATH:-}"

python - <<'PY'
import pufferlib._C
print("ok")
PY
```

如果只是做验证，也可以先导入 `torch`：

```bash
python - <<'PY'
import torch
import pufferlib._C
print("ok")
PY
```

如果仍然失败，再重新编译 PyTorch 扩展：

```bash
python setup.py build_ext --inplace --force
```

如果只是临时做非训练任务，可用：

```bash
NO_TRAIN=1 python setup.py build_ext --inplace --force
```

### 13.6 `DRIVE_BINARIES_DATA_ROOT` 未设置或地图缺失

训练和评估依赖二进制地图：

```bash
export DRIVE_BINARIES_DATA_ROOT=/path/to/binaries
ls "$DRIVE_BINARIES_DATA_ROOT"
```

如果目录下没有 `.bin` 文件，请先下载/转换数据。

### 13.7 `python pufferlib/ocean/drive/drive.py` 显示 `Found 0 json files`

如果转换时看到：

```text
Found 0 json files
Processing maps: 0map
```

说明 `DRIVE_DATA_ROOT` 下的 split 目录里没有解压后的 `.json` 文件。`hf download` 下载 `pufferdrive_womd_train` / `pufferdrive_womd_val` 时拿到的是 zip 包，需要先解压：

```bash
cd /path/to/behavior-bench
export REPO_ROOT="$(pwd)"

# training
unzip -q -n data/processed/training/training.zip -d .

# validation
unzip -q -n data/processed/validation/pufferdrive_val.zip -d .

# 检查 JSON 数量
python - <<'PY'
from pathlib import Path
for split in ["training", "validation", "mixed"]:
    p = Path("data/processed") / split
    print(split, len(list(p.rglob("*.json"))))
PY
```

确认有 JSON 后再运行：

```bash
export REPO_ROOT="$(pwd)"
export DRIVE_DATA_ROOT="$REPO_ROOT/data/processed"
export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/resources/drive/binaries"
python pufferlib/ocean/drive/drive.py
```

### 13.8 构建时下载 Raylib/Box2D 失败

`setup.py` 构建 Ocean/Drive 扩展时会从 GitHub 下载 Raylib、Box2D、inih。若机器无法访问外网：

- 在可联网机器下载相同归档并放到项目根目录后重试。
- 或临时跳过 Ocean 构建：

```bash
NO_OCEAN=1 python -m pip install -e . --no-build-isolation --no-deps
NO_OCEAN=1 python setup.py build_ext --inplace --force
```

### 13.9 CUDA 扩展没有编译

`setup.py` 只有在 `nvcc` 存在时才编译 CUDA extension，否则会编译 C++ CPU extension：

```bash
which nvcc
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

如果需要 CUDA extension，请使用带 CUDA Toolkit 的环境或容器，并确保 `nvcc` 在 PATH 中。

如果 CUDA Toolkit 由 Conda 安装，但构建脚本没有找到它，可以显式设置：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
python setup.py build_ext --inplace --force
```

## 14. 一键命令清单（推荐完整流程）

下面是一份从零开始的完整命令清单。请根据是否有 GPU 修改 PyTorch 安装那一行。

```bash
cd /path/to/behavior-bench

# 1) 准备 Conda 环境
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -n behavior-bench python=3.11 pip -y
conda activate behavior-bench
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
python -V

# 2) 基础构建依赖
conda install -y pip setuptools wheel cython "numpy<2.0"
python -m pip install -U pip setuptools wheel

# 3) 安装 PyTorch（二选一）
# CPU:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 当前机器 nvidia-smi 显示驱动支持 CUDA 12.8。如果使用 CUDA，请注释掉上面的 CPU 安装命令，改用这一行：
# python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 4) 安装 PyG 相关依赖
TORCH_MM=$(python - <<'PY'
import torch
v = torch.__version__.split("+")[0].split(".")
print(f"{v[0]}.{v[1]}.0")
PY
)
CUDA_TAG=$(python - <<'PY'
import torch
print("cpu" if torch.version.cuda is None else "cu" + torch.version.cuda.replace(".", ""))
PY
)
python -m pip install "torch-cluster==1.6.3" -f "https://data.pyg.org/whl/torch-${TORCH_MM}+${CUDA_TAG}.html"
python -m pip install "torch-geometric==2.7.0"

# 5) 安装项目依赖并 editable 安装
python -m pip install \
  "setuptools" \
  "numpy<2.0" \
  "shimmy[gym-v21]" \
  "gym==0.23" \
  "gymnasium==0.29.1" \
  "pettingzoo==1.24.1" \
  "tensorflow==2.20.0" \
  "scipy==1.17.0" \
  "pillow==11.3.0" \
  "msgpack==1.1.2" \
  "psutil" \
  "nvidia-ml-py" \
  "rich" \
  "rich_argparse" \
  "imageio" \
  "pyro-ppl" \
  "heavyball<2.0.0" \
  "neptune" \
  "wandb" \
  "matplotlib"
python -m pip install -e . --no-build-isolation --no-deps

# 6) 可选：CARLA/pyxodr 开发依赖
git submodule update --init --recursive
python -m pip install -r requirements-dev.txt

# 7) 编译本地扩展
python setup.py build_ext --inplace --force

# 8) 验证安装
python - <<'PY'
import numpy
import torch
import pufferlib
import pufferlib._C
from pufferlib.ocean.drive import binding
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("pufferlib and native extensions: ok")
print("MAX_OBS_PARTNERS:", binding.MAX_OBS_PARTNERS)
PY

# 9) 设置数据路径后再运行训练/评估
export DRIVE_BINARIES_DATA_ROOT=/path/to/binaries
puffer --help
# puffer train puffer_drive --train.device cpu
# python pufferlib/ocean/benchmark/eval.py --map-ids 0-10
```

## 15. 建议的日常使用命令

每次进入项目：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate behavior-bench
cd /path/to/behavior-bench

export LD_LIBRARY_PATH="$(python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
):${LD_LIBRARY_PATH:-}"

export DRIVE_BINARIES_DATA_ROOT=/path/to/binaries
```

拉取代码后，如果 C/Raylib/Drive 源码或 Python 版本变化：

```bash
python setup.py build_ext --inplace --force
```

查看已安装包：

```bash
conda list
python -m pip list
```

重新同步 editable 安装：

```bash
python -m pip install -e . --no-build-isolation --no-deps
```
