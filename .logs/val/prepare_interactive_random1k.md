# PufferDrive Benchmark Splits 准备说明

本文档基本按 `BENCHMARK.md` 原文翻译整理，说明如何复现 **Interactive1k**（兼容名 `pufferinter`）和 **Random1k**（兼容名 `pufferrandom`）两个 benchmark split。不同之处只在“本机补充”中说明：使用独立 conda 环境、独立大文件目录和软链接，避免破坏当前 `behavior-bench` 训练环境与代码库。

两个 split 由仓库中的 CSV manifest 定义：

- `pufferlib/resources/splits/interactive1k.csv`
- `pufferlib/resources/splits/random1k.csv`

每个 manifest 都引用 WOMD validation set 中的 1000 个场景。下面流程会生成与这些 manifest 匹配的二进制场景目录，供 evaluation harness 使用。

---

## 本机补充：目录约定

```bash
cd "$HOME/wsc/behavior-bench"
export REPO_ROOT="$(pwd)"

# 大文件、转换中间产物和最终 benchmark split 均放在 /data2 下
export BENCH_ROOT="/data2/puffer/benchmark_build"
export WAYMO_ROOT="$BENCH_ROOT/waymo"
export GPUDRIVE_JSON_ROOT="$BENCH_ROOT/gpudrive_json"
export ENRICHED_ROOT="$BENCH_ROOT/gpudrive_with_connectivity"
export FULL_BIN_ROOT="$BENCH_ROOT/full_binaries"
export FINAL_SPLIT_ROOT="/data2/puffer/eval_splits"

mkdir -p "$BENCH_ROOT" "$WAYMO_ROOT" "$GPUDRIVE_JSON_ROOT" "$ENRICHED_ROOT" "$FULL_BIN_ROOT" "$FINAL_SPLIT_ROOT"
```

当前仓库的 `data` 是软链接：

```bash
ls -ld "$REPO_ROOT/data"
# 期望类似：
# data -> /data2/puffer/data
```

最终会让：

```text
$REPO_ROOT/data/eval_splits -> /data2/puffer/eval_splits
```

## 本机补充：创建独立转换环境

不要在当前 `behavior-bench` 训练环境里混装 Waymo / TensorFlow / ScenarioMax。创建单独环境：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda create -n behavior-bench-waymo python=3.10 pip -y
conda activate behavior-bench-waymo

python -m pip install -U pip setuptools wheel
python -m pip install "tensorflow==2.11.*" "waymo-open-dataset-tf-2-11-0==1.6.1"
python -m pip install tqdm numpy pandas
```

安装 Google Cloud 下载工具。任选一种：

```bash
# 方案 A：conda 安装 Google Cloud SDK（推荐）
conda install -c conda-forge google-cloud-sdk -y

# 方案 B：如果 conda 包不可用，可尝试 pip 版 gsutil
# python -m pip install gsutil
```

验证：

```bash
which gsutil
python - <<'PY'
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
print("tensorflow:", tf.__version__)
print("waymo scenario proto: ok")
PY
```

---

## Step 1 — Waymo JSONs -> Validation Binaries

完整 WOMD validation set 需要先转换为带 lane connectivity 的 PufferDrive binaries。这个流程需要两个输入：

- **Raw Waymo TFRecords**：只用于抽取 lane connectivity。它来自 Waymo Open Motion Dataset，路径通常在 `uncompressed/scenario/{training,validation,testing}/` 下。
- **GPUDrive-style JSON scenarios**：每个 scenario 一个 JSON，用于构建 binary trajectories 和 road graph。

### 1a. 获取 GPUDrive-style JSONs

为了复现 benchmark，需要 WOMD validation set 中每个 scenario 都有一个 GPUDrive-style JSON。`BENCHMARK.md` 推荐用 [ScenarioMax](https://github.com/valeoai/ScenarioMax) 从原始 Waymo TFRecords 转出 GPUDrive JSON。

`BENCHMARK.md` 原文写的是旧版入口：

```bash
git clone https://github.com/valeoai/ScenarioMax.git
cd ScenarioMax
pip install -e .

# Convert WOMD TFRecords -> GPUDrive JSONs (see ScenarioMax docs for full options)
python -m scenariomax.convert \
  --source waymo \
  --input /path/to/waymo/uncompressed/scenario \
  --output /path/to/gpudrive \
  --format gpudrive
```

当前实测安装的 ScenarioMax 版本没有 `scenariomax.convert` 模块，实际命令行入口是 `scenariomax-convert`，等价 Python 模块入口是 `python -m scenariomax.convert_dataset`。后续请使用本文“本机运行 ScenarioMax 转换”中的命令。

> 原文注意：预构建的 `EMERGE-lab/GPUDrive` HuggingFace 数据集只是 WOMD 子集，不能覆盖完整 validation set，因此不能用来复现 Interactive1k / Random1k。要复现 benchmark，需要用 ScenarioMax 转完整 WOMD validation。

本机建议安装 ScenarioMax 到独立工具目录：

```bash
mkdir -p /data2/puffer/tools
cd /data2/puffer/tools

if [ ! -d ScenarioMax ]; then
  git clone https://github.com/valeoai/ScenarioMax.git
fi

cd ScenarioMax
python -m pip install -e .
```

### 本机补充：下载完整 WOMD validation TFRecords

Waymo Open Motion Dataset 需要接受 Waymo 条款/权限。请先按 Waymo 官网完成访问授权。

常见公开路径示例：

```bash
export WAYMO_GCS_ROOT="gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario"
```

只下载 validation：

```bash
mkdir -p "$WAYMO_ROOT"

gsutil -m cp -r \
  "$WAYMO_GCS_ROOT/validation" \
  "$WAYMO_ROOT/"
```

完成后应类似：

```text
$WAYMO_ROOT/validation/
  validation.tfrecord-00000-of-00150
  validation.tfrecord-00001-of-00150
  ...
```

检查：

```bash
find "$WAYMO_ROOT/validation" -type f | wc -l
ls "$WAYMO_ROOT/validation" | head
```

如果 Waymo 官方当前版本不是 `v_1_2_0`，请以 Waymo 下载页面给出的 `gs://.../uncompressed/scenario/validation` 路径为准。

#### 国内下载补充：OpenDataLab

如果国内环境访问 Waymo 官方 Google Cloud Storage 不稳定，可以优先尝试 OpenDataLab。当前测试结果：

```text
openxlab CLI 已安装
OpenDataLab/Waymo_Motion_Dataset_v1_dot_2 可访问
小文件 README.md 下载成功
数据集元信息中包含 /raw/scenario/validation.tar，大小约 38.40G
```

轻量连通性测试：

```bash
openxlab dataset info --dataset-repo OpenDataLab/Waymo_Motion_Dataset_v1_dot_2

rm -rf /tmp/opendatalab_waymo_test
mkdir -p /tmp/opendatalab_waymo_test
openxlab dataset download \
  --dataset-repo OpenDataLab/Waymo_Motion_Dataset_v1_dot_2 \
  --source-path /README.md \
  --target-path /tmp/opendatalab_waymo_test
```

下载完整 WOMD validation scenario 包：

```bash
cd "$REPO_ROOT"
export OPENDATALAB_ROOT="$BENCH_ROOT/opendatalab_waymo"
mkdir -p "$OPENDATALAB_ROOT"

openxlab dataset download \
  --dataset-repo OpenDataLab/Waymo_Motion_Dataset_v1_dot_2 \
  --source-path /raw/scenario/validation.tar \
  --target-path "$OPENDATALAB_ROOT"
```

下载完成后检查并解压：

```bash
tar -tf "$OPENDATALAB_ROOT/OpenDataLab___Waymo_Motion_Dataset_v1_dot_2/raw/scenario/validation.tar" | head

mkdir -p "$WAYMO_ROOT"

tar -xf "$OPENDATALAB_ROOT/OpenDataLab___Waymo_Motion_Dataset_v1_dot_2/raw/scenario/validation.tar" \
  -C "$WAYMO_ROOT"
```

最终需要整理成：

```text
$WAYMO_ROOT/validation/
  validation.tfrecord-00000-of-00150
  validation.tfrecord-00001-of-00150
  ...
```

然后继续执行下面的 ScenarioMax 转换、`create_training_binaries.py`、`remap_files.py` 步骤。注意：OpenDataLab 页面也声明数据使用需遵循 Waymo 官方协议，请确认你的使用权限。

本机运行 ScenarioMax 转换。当前版本已验证 `scenariomax-convert --help` 可用；若 CLI 参数变化，以当前 `--help` 为准：

```bash
cd /data2/puffer/tools/ScenarioMax

scenariomax-convert \
  --waymo_src "$WAYMO_ROOT/validation" \
  --dst "$GPUDRIVE_JSON_ROOT/validation" \
  --target_format gpudrive \
  --num_workers 8
```

等价备选：

```bash
python -m scenariomax.convert_dataset \
  --waymo_src "$WAYMO_ROOT/validation" \
  --dst "$GPUDRIVE_JSON_ROOT/validation" \
  --target_format gpudrive \
  --num_workers 8
```

期望输出：

```text
$GPUDRIVE_JSON_ROOT/validation/
  *.json
```

检查：

```bash
find "$GPUDRIVE_JSON_ROOT/validation" -type f -name "*.json" | wc -l
find "$GPUDRIVE_JSON_ROOT/validation" -type f -name "*.json" | head
```

如果 ScenarioMax 只转换出部分文件，请先不要继续；`Interactive1k / Random1k` 需要完整 validation 覆盖。

### 1b. JSONs -> Binaries

原文要求运行 `data_utils/womd/create_training_binaries.py`，从 TFRecords 抽取 lane connectivity， enrich JSONs，然后转换为 `.bin`：

```bash
python data_utils/womd/create_training_binaries.py \
  --tfrecord-root /path/to/waymo \
  --json-root /path/to/gpudrive \
  --enriched-root /path/to/gpudrive_with_connectivity \
  --output-root /path/to/binaries \
  --workers 80
```

期望输入布局：

```text
/path/to/waymo/{training,validation,testing}/      # raw Waymo TFRecords
/path/to/gpudrive/{training,validation,testing}/   # GPUDrive JSONs
```

期望输出：

```text
/path/to/binaries/validation/
  map_000000.bin
  map_000001.bin
  ...
```

这里的文件名 `map_NNNNNN.bin` 就是 split manifests 中 `original_filename` 所引用的名字。

本机建议分两段执行：

- Step 1/2（TFRecord connectivity 提取 + enrich JSON）需要 TensorFlow / Waymo 依赖，使用独立 `behavior-bench-waymo` 环境。
- Step 3（enriched JSON -> Drive `.bin`）会 import `pufferlib.ocean.drive.drive`，需要主项目运行依赖（例如 `gymnasium`、Drive C extension），建议切回主 `behavior-bench` 环境执行。

如果从头运行，可先执行前两步：

```bash
cd "$REPO_ROOT"
conda activate behavior-bench-waymo

python data_utils/womd/create_training_binaries.py \
  --tfrecord-root "$WAYMO_ROOT" \
  --json-root "$GPUDRIVE_JSON_ROOT" \
  --enriched-root "$ENRICHED_ROOT" \
  --output-root "$FULL_BIN_ROOT" \
  --workers 80 \
  --splits validation
```

如果它在 Step 3 报错：

```text
ModuleNotFoundError: No module named 'gymnasium'
```

说明 `behavior-bench-waymo` 环境只适合 TensorFlow/Waymo 转换，不适合 import 主项目。此时不要重跑前两步；只要确认 enriched JSON 已经生成：

```bash
find "$ENRICHED_ROOT/validation" -type f -name "*.json" | wc -l
```

然后切回主 `behavior-bench` 环境，从 Step 3 恢复：

```bash
cd "$REPO_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

python - <<'PY'
import os
from pathlib import Path
from pufferlib.ocean.drive.drive import process_all_maps

enriched_root = Path(os.environ["ENRICHED_ROOT"])
full_bin_root = Path(os.environ["FULL_BIN_ROOT"])

process_all_maps(
    json_dir=enriched_root / "validation",
    binary_dir=full_bin_root / "validation",
    max_maps=10_000_000,
    num_workers=80,
)
PY
```

期望输出：

```text
$FULL_BIN_ROOT/validation/
  map_000000.bin
  map_000001.bin
  ...
  map_032035.bin
  ...
```

检查是否覆盖 manifest 需要的源文件：

```bash
cd "$REPO_ROOT"

python - <<'PY'
import csv
from pathlib import Path
src = Path("/data2/puffer/benchmark_build/full_binaries/validation")
for csv_path in [
    Path("pufferlib/resources/splits/interactive1k.csv"),
    Path("pufferlib/resources/splits/random1k.csv"),
]:
    missing = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if not (src / row["original_filename"]).exists():
                missing.append(row["original_filename"])
    print(csv_path.name, "missing", len(missing))
    print("first missing:", missing[:10])
PY
```

必须看到：

```text
interactive1k.csv missing 0
random1k.csv missing 0
```

否则说明完整 validation binaries 仍不完整，不能生成完整 benchmark split。

---

## Step 2 — Validation Binaries -> Benchmark Split

使用 `scripts/remap_files.py` 和其中一个 manifest CSV 生成 split。

原文命令：

```bash
python scripts/remap_files.py \
  --source-path /path/to/binaries/validation \
  --csv-path    pufferlib/resources/splits/interactive1k.csv \
  --target-path /path/to/test_eval
```

脚本会：

1. 读取 `interactive1k.csv`，列包含 `new_filename, original_filename, ego_agent_idx, interactivity_score, map_id`。
2. 对每一行，把 `<source-path>/<original_filename>` 复制到 `<target-path>/<csv_stem>/<new_filename>`。
3. 把 CSV 自身复制到输出目录，命名为 `manifest.csv`。

结果布局：

```text
/path/to/test_eval/interactive1k/
  map_000000.bin       # was map_032035.bin in the full validation set
  map_000001.bin       # was map_005373.bin
  ...
  map_000999.bin
  manifest.csv
```

如果要生成 random split，把 CSV 换成 `random1k.csv`，输出目录会按 CSV stem 命名为 `random1k/`。

本机生成命令：

```bash
cd "$REPO_ROOT"

rm -rf "$FINAL_SPLIT_ROOT/interactive1k" "$FINAL_SPLIT_ROOT/random1k"

python scripts/remap_files.py \
  --source-path "$FULL_BIN_ROOT/validation" \
  --csv-path pufferlib/resources/splits/interactive1k.csv \
  --target-path "$FINAL_SPLIT_ROOT"

python scripts/remap_files.py \
  --source-path "$FULL_BIN_ROOT/validation" \
  --csv-path pufferlib/resources/splits/random1k.csv \
  --target-path "$FINAL_SPLIT_ROOT"
```

检查完整性：

```bash
python - <<'PY'
from pathlib import Path
root = Path("/data2/puffer/eval_splits")
for split in ["interactive1k", "random1k"]:
    p = root / split
    ids = sorted(int(f.stem.split("_")[1]) for f in p.glob("map_*.bin"))
    missing = [i for i in range(1000) if i not in set(ids)]
    print(split)
    print("  count:", len(ids))
    print("  min/max:", (min(ids), max(ids)) if ids else None)
    print("  missing:", missing[:20], "missing_count=", len(missing))
    print("  manifest:", (p / "manifest.csv").exists())
PY
```

期望：

```text
count: 1000
min/max: (0, 999)
missing_count= 0
manifest: True
```

---

## 本机补充：建立当前仓库需要的软链接

当前仓库的 `data` 已经是软链接。这里确保最终结构是：

```text
$REPO_ROOT/data/eval_splits/interactive1k
$REPO_ROOT/data/eval_splits/random1k
$REPO_ROOT/data/eval_splits/pufferinter
$REPO_ROOT/data/eval_splits/pufferrandom
```

命令：

```bash
cd "$REPO_ROOT"

# 确保 data/eval_splits 指向 /data2/puffer/eval_splits
if [ -e "$REPO_ROOT/data/eval_splits" ] && [ ! -L "$REPO_ROOT/data/eval_splits" ]; then
  mv "$REPO_ROOT/data/eval_splits" "$REPO_ROOT/data/eval_splits.backup.$(date +%Y%m%d_%H%M%S)"
fi

ln -sfn /data2/puffer/eval_splits "$REPO_ROOT/data/eval_splits"

# 兼容代码白名单和 BENCHMARK.md 中的命名
ln -sfn interactive1k "$REPO_ROOT/data/eval_splits/pufferinter"
ln -sfn random1k "$REPO_ROOT/data/eval_splits/pufferrandom"
```

检查：

```bash
ls -ld "$REPO_ROOT/data" "$REPO_ROOT/data/eval_splits"
ls -ld "$REPO_ROOT/data/eval_splits/interactive1k" \
       "$REPO_ROOT/data/eval_splits/random1k" \
       "$REPO_ROOT/data/eval_splits/pufferinter" \
       "$REPO_ROOT/data/eval_splits/pufferrandom"
```

---

## Using the Split in Evaluation

原文说明：split 目录存在后，把 `DRIVE_BINARIES_DATA_ROOT` 指向 split 的父目录，然后运行：

```bash
export DRIVE_BINARIES_DATA_ROOT=/path/to/test_eval

python pufferlib/ocean/benchmark/eval.py \
  --eval.split interactive1k --map-ids all
```

传给 `--eval.split` 的 split 名称必须和 Step 2 生成的目录名一致，即 CSV stem。

本机因为 `Drive` 环境白名单当前支持 `pufferinter` / `pufferrandom`，所以优先使用这两个兼容 split 名称：

`Drive` 环境白名单中当前支持 `pufferinter` / `pufferrandom`，所以优先使用这两个兼容 split 名称：

```bash
cd "$REPO_ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate behavior-bench

export DRIVE_BINARIES_DATA_ROOT="$REPO_ROOT/data/eval_splits"

python pufferlib/ocean/benchmark/eval.py \
  --planner.type ppo \
  --planner.ppo.weights-path experiments/puffer_drive_xt5biagd.pt \
  --traffic.type idm \
  --eval.split pufferinter \
  --map-ids all
```

Random1k：

```bash
python pufferlib/ocean/benchmark/eval.py \
  --planner.type ppo \
  --planner.ppo.weights-path experiments/puffer_drive_xt5biagd.pt \
  --traffic.type idm \
  --eval.split pufferrandom \
  --map-ids all
```

---

## 本机补充：不破坏现有环境的原则

本流程遵循：

```text
不改 behavior-bench conda 训练环境
不覆盖 resources/drive/binaries/training
不覆盖 resources/drive/binaries/validation
大文件都写入 /data2/puffer/benchmark_build
最终 benchmark split 写入 /data2/puffer/eval_splits
仓库只通过 data/eval_splits 软链接访问最终结果
```

如果任一步中断，可以重复执行对应步骤。`remap_files.py` 只复制文件；`create_training_binaries.py` 的输出目录可以保留，必要时删除对应 split 后重跑。
