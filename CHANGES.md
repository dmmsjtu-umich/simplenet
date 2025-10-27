# 变更记录（由助手自动维护）

本文件用于记录由我（助手）在仓库中所做的文件级改动，包含简要说明、变更原因与时间戳。未来每次我在仓库中修改文件，会同步在此追加记录。

---

## 2025-10-24  (UTC+8)

- 文件: `run.sh`
  - 变更: 修复变量赋值错误并加入显存碎片优化与默认更低的 batch 大小。
  - 详细:
    - 具体代码变更（原始 -> 替换为）：

      原始 (before):
      ```bash
      datapath= mvtec_anomaly_detection
      
      python3 main.py \
      --gpu 4 \
      --seed 0 \
      ... \
      dataset \
      --batch_size 8 \
      --resize 329 \
      --imagesize 288 "${dataset_flags[@]}" mvtec $datapath
      ```

      替换为 (after):
      ```bash
      datapath=mvtec_anomaly_detection
      # Reduce PyTorch CUDA fragmentation which can cause OOMs
      export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

      python3 main.py \
      --gpu 0 \
      --seed 0 \
      ... \
      dataset \
      --batch_size 2 \
      --resize 329 \
      --imagesize 288 "${dataset_flags[@]}" mvtec $datapath
      ```
      说明：省略号 `...` 代表未修改的其余命令行参数，仅显示与本次变更相关的部分。
  - 原因: 修复导致 `command not found: mvtec_anomaly_detection` 的 bug；减小显存压力，缓解 `CUDA out of memory` 问题。

- 文件: `metrics.py`
  - 变更: 兼容性修复（pandas / numpy 现代版本）。
  - 详细:
    - 具体代码变更（原始 -> 替换为）：

      原始 (before):
      ```python
      df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
      binary_amaps = np.zeros_like(amaps, dtype=np.bool)
      
      ...

      df = df.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

      # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
      df = df[df["fpr"] < 0.3]
      df["fpr"] = df["fpr"] / df["fpr"].max()

      pro_auc = metrics.auc(df["fpr"], df["pro"])
      return pro_auc
      ```

      替换为 (after):
      ```python
      df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
      # use Python bool dtype to avoid deprecated np.bool
      binary_amaps = np.zeros_like(amaps, dtype=bool)
      
      ...

      denom = inverse_masks.sum()
      fpr = fp_pixels / denom if denom > 0 else 0.0

      # handle empty pros (no annotated regions) safely
      mean_pro = float(np.mean(pros)) if len(pros) > 0 else 0.0

      df.loc[len(df)] = {"pro": mean_pro, "fpr": fpr, "threshold": th}

      # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
      df = df[df["fpr"] < 0.3]

      # Need at least two points to compute AUC; handle edge cases gracefully
      if df.empty or len(df["fpr"]) < 2:
          return 0.0

      max_fpr = df["fpr"].max()
      if max_fpr == 0:
          return 0.0

      df["fpr"] = df["fpr"] / max_fpr

      pro_auc = metrics.auc(df["fpr"], df["pro"])
      return pro_auc
      ```
  - 原因: 避免在较新版本的 numpy/pandas 中触发 AttributeError / DeprecationError，从而保证脚本在现代环境中可以运行。

- 文件: `metrics.py` (追加修复)
  - 变更: 增强 compute_pro 的健壮性与边界条件处理。
  - 详细:
    - 对每个阈值，若没有有效的 region-pros，使用 `pro = 0.0`，避免产生 NaN 值。
    - 在计算 FPR 时若分母为 0（没有负样本像素），将 FPR 设为 0.0，避免除零。
    - 在归一化并计算 AUC 前，检查样本点数量；若小于 2 点则返回 0.0，避免 sklearn.metrics.auc 抛出错误。
  - 原因: 在某些数据/阈值下，`df` 可能只包含 0 或 1 个点，导致 sklearn 的 AUC 计算失败；这些修复让 `compute_pro` 在稀疏或极端情况下能安全返回 0 而不是抛异常。

  - 具体代码变更（原始 -> 替换为）：

  原始 (before):
  ```python
  pros = []
  for binary_amap, mask in zip(binary_amaps, masks):
    binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
    for region in measure.regionprops(measure.label(mask)):
      axes0_ids = region.coords[:, 0]
      axes1_ids = region.coords[:, 1]
      tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
      pros.append(tp_pixels / region.area)

  inverse_masks = 1 - masks
  fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
  fpr = fp_pixels / inverse_masks.sum()

  df.loc[len(df)] = {"pro": np.mean(pros), "fpr": fpr, "threshold": th}

  # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
  df = df[df["fpr"] < 0.3]
  df["fpr"] = df["fpr"] / df["fpr"].max()

  pro_auc = metrics.auc(df["fpr"], df["pro"])
  return pro_auc
  ```
  - 其他小改动:
    - 文件: `main.py`
      - 变更: 修复 `--test` 行为，现在 `--test` 会调用 `SimpleNet.test(...)` 来用已存在的 checkpoint 做评估并生成指标。
      - 详细: 之前 `--test` 分支仅打印警告并未调用测试函数；现已改为调用 `SimpleNet.test` 并将 `pro_auroc` 置为 `-1`（`test` 返回 `auroc` 与 `full_pixel_auroc` 两个值）。

  替换为 (after):
  ```python
  pros = []
  for binary_amap, mask in zip(binary_amaps, masks):
    binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
    for region in measure.regionprops(measure.label(mask)):
      axes0_ids = region.coords[:, 0]
      axes1_ids = region.coords[:, 1]
      tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
      pros.append(tp_pixels / region.area)

  inverse_masks = 1 - masks
  fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
  denom = inverse_masks.sum()
  fpr = fp_pixels / denom if denom > 0 else 0.0

  # handle empty pros (no annotated regions) safely
  mean_pro = float(np.mean(pros)) if len(pros) > 0 else 0.0

  df.loc[len(df)] = {"pro": mean_pro, "fpr": fpr, "threshold": th}

  # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
  df = df[df["fpr"] < 0.3]

  # Need at least two points to compute AUC; handle edge cases gracefully
  if df.empty or len(df["fpr"]) < 2:
    return 0.0

  max_fpr = df["fpr"].max()
  if max_fpr == 0:
    return 0.0

  df["fpr"] = df["fpr"] / max_fpr

  pro_auc = metrics.auc(df["fpr"], df["pro"])
  return pro_auc
  ```

---

后续流程

- 每次我对仓库内文件进行非平凡修改（添加/更新源文件、脚本、配置），我将把变更记录追加到本文件，包含：变更日期、文件、关键修改点、原因。

---

如果你希望变更记录采用不同格式（例如更详细的 diff、关联 git commit 或放在 `docs/CHANGELOG.md`），告诉我我会调整。

---

## 2025-10-24  (UTC+8)  — run.sh: restore backbone, increase batch

- 文件: `run.sh`
  - 变更: 将 backbone 恢复为 `wideresnet50` 并将 `--batch_size` 从 2 提升至 4。
  - 详细:
    - 具体代码变更（原始 -> 替换为）:

      原始 (before):
      ```bash
      python3 main.py \
      --gpu 0 \
      --seed 0 \
      ... \
      net \
      -b resnet18 \
      ... \
      dataset \
      --batch_size 2 \
      --resize 329 \
      --imagesize 288 "${dataset_flags[@]}" mvtec $datapath
      ```

      替换为 (after):
      ```bash
      python3 main.py \
      --gpu 0 \
      --seed 0 \
      ... \
      net \
      -b wideresnet50 \
      ... \
      dataset \
      --batch_size 4 \
      --resize 329 \
      --imagesize 288 "${dataset_flags[@]}" mvtec $datapath
      ```
    - 原因: 你要求恢复为原本更强的 backbone (`wideresnet50`) 并将 batch size 设为 4 以提高训练吞吐，同时保留之前为防 OOM 而加入的 CUDA 分配优化。

## 2025-10-25 (UTC+8) — 清理根目录部署副本

- 操作: 删除仓库根目录中重复且已在 `deployment/` 下维护的部署相关文件，避免混淆。
- 删除的文件:
  - `DEPLOY.md`
  - `Dockerfile`
  - `inference_cli.py`
  - `load_test.sh`
  - `serve.py`
  - `requirements.txt`

- 说明: 这些文件的权威副本保存在 `deployment/` 目录下（`deployment/DEPLOY.md`, `deployment/Dockerfile`, 等）。为避免误用或不同步，已将根目录中的重复文件删除。若你希望保留根目录备份，请在删除前告知（本次操作已按你的确认执行）。

---


