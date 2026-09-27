# FTC-Seg

**When Noise Meets Long-Tail: Feature-Threshold Dual Calibration for Robust Pseudo-Labeling**.

## Installation

After downloading this repository, create an environment:

```bash
cd FTC-Seg
conda create -n FTC-Seg python=3.10 -y
conda activate FTC-Seg
python -m pip install -r requirements.txt \
  "numpy<2" "opencv-python<4.12" \
  torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

The NumPy and OpenCV constraints keep the dependencies compatible with this older PyTorch release. See the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) for other CUDA builds.

Run all commands below from the **FTC-Seg repository root**.

## Pretrained Backbone

The provided configurations use **DINOv2-S**. Download the [DINOv2 ViT-S/14 pretrained backbone](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth), rename it to `dinov2_small.pth`, and place it under `pretrained/`:

```text
FTC-Seg/
└── pretrained/
    └── dinov2_small.pth
```

## Dataset Preparation

| Dataset | Download | Preparation |
| --- | --- | --- |
| FSSG | [Baidu Netdisk](https://pan.baidu.com/s/10qCqdCkPZ_o7hk7KlC74nw?pwd=0516) (code: `0516`) | FTC-Seg-prepared data; extract and set `data_root`. |
| FLSMD | **Baidu Netdisk: link and access code to be added.** | FTC-Seg-prepared data; extract and set `data_root`. |
| SUIM | [Official download page](https://irvlab.cs.umn.edu/resources/suim-dataset) | Download the original data and follow the SUIM preparation steps below. |
| ACDC | [Official download page](https://acdc.vision.ee.ethz.ch/download) | Obtain the original data under the official license and follow the ACDC preparation steps below. |

FSSG and FLSMD are provided in the directory and label format used by FTC-Seg. SUIM and ACDC must be downloaded from their original providers.

The SUIM and ACDC data used by this project have the following counts of unique image-mask pairs:

| Dataset | Prepared directory | Training pool | FTC-Seg validation | Stored but excluded from these experiments |
| --- | ---: | ---: | ---: | ---: |
| SUIM | 1,635 | 1,219 | 306 | 110 official test pairs |
| ACDC | 2,006 | 1,600 | 406 | 0 |

For SUIM, the training pool and FTC-Seg validation list partition the official 1,525-pair `train_val/` set. For ACDC, use the official 1,600-pair training set and 406-pair validation set. Always use the provided validation lists, rather than generating a new random validation split. Matching these counts does not reconstruct the private labeled/unlabeled partitions, which are not distributed.

### Sources, Attribution, and Licenses

- **FSSG:** Released by Ping Guo et al. with [CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels](https://arxiv.org/abs/2603.21071), CVPR 2026. Please cite that paper and acknowledge the [original FSSG release](https://github.com/pingggg516/CTFS). The data are for **non-commercial use only**.
- **FLSMD:** Images and segmentation annotations originate from Deepak Singh and Matias Valdenegro-Toro, [The Marine Debris Dataset for Forward-Looking Sonar Semantic Segmentation](https://arxiv.org/abs/2108.06800), ICCV Workshops 2021. Please also cite Matias Valdenegro-Toro et al., [The Marine Debris Forward-Looking Sonar Datasets](https://arxiv.org/abs/2503.22880), OCEANS 2025. The [author's data release](https://zenodo.org/records/15101686) uses **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**, which also applies to the prepared data. This copy reorganizes the data into `Images/` and `Masks/` with class-ID PNG masks for FTC-Seg.
- **SUIM:** Cite Md Jahidul Islam et al., [Semantic Segmentation of Underwater Imagery: Dataset and Benchmark](https://arxiv.org/abs/2004.01241), IROS 2020. Obtain the data from the official page and follow the original provider's terms.
- **ACDC:** Cite Christos Sakaridis, Dengxin Dai, and Luc Van Gool, *ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding*, ICCV 2021; see the [official citation page](https://acdc.vision.ee.ethz.ch/citation). Download and use the data under the [official license](https://acdc.vision.ee.ethz.ch/license).

### Required Directory and Split Format

Each dataset has its own root containing `Images/` and `Masks/`. Set that root in the corresponding YAML configuration:

```yaml
dataset: FSSG
data_root: /path/to/datasets/FSSG
```

| Dataset | Configuration | Number of classes | Class IDs | Validation list |
| --- | --- | ---: | --- | --- |
| FLSMD | `configs/FLSMD.yaml` | 12 | `0–11` | `splits/FLSMD/val.txt` |
| FSSG | `configs/FSSG.yaml` | 11 | `0–10` | `splits/FSSG/val.txt` |
| SUIM | `configs/SUIM.yaml` | 8 | `0–7` | `splits/SUIM/val.txt` |
| ACDC | `configs/ACDC.yaml` | 19 | `0–18` | `splits/ACDC/val.txt` |

Masks must be single-channel integer class-ID images; `255` denotes ignored pixels. Validation image and mask dimensions must match. Preserve image dimensions during preparation; see the SUIM notes below for the historical training-mask size exceptions.

Every split-list line contains an image path and a mask path, separated by a single space. Both paths are relative to `data_root`. For example, an existing FSSG validation entry is:

```text
Images/0_frame_1060.png Masks/0_frame_1060.png
```

During training, the validation list is read automatically from `splits/<dataset>/val.txt`; the YAML `dataset` value must match the directory name, including capitalization. During inference, pass that list through `--split-path`.

The labeled/unlabeled training partitions are not included. Provide your own training lists in the same format and exclude validation and test samples from them. The unlabeled training loader reads only the image and does not use the listed mask.

### SUIM: Download and Prepare

1. Open the [official SUIM page](https://irvlab.cs.umn.edu/resources/suim-dataset) and follow **Download Link: Google Drive** to download `SUIM.zip`. It contains 1,525 pairs under `train_val/` and 110 pairs under `TEST/`. To reproduce the complete 1,635-pair prepared directory, collect images from both `train_val/images/` and `TEST/images/`, and the combined RGB annotations directly under their respective `masks/` directories. Do not use the per-class binary-mask subdirectories. Only the 1,525 `train_val/` pairs participate in the FTC-Seg training/validation setup; keep all 110 `TEST/` pairs out of those lists. `Benchmark_Evaluation/` and pretrained SUIM-Net checkpoints are not needed.
2. Place the images under `Images/`. Preserve each filename stem, including underscores; convert non-PNG images to PNG without resizing so their names match the provided split lists.
3. Convert each RGB annotation into a **single-channel class-ID PNG** under `Masks/`, using the same filename stem as its image. Do not convert the RGB mask to ordinary grayscale intensity.

The RGB-to-ID mapping follows the [official SUIM color encoding](https://irvlab.cs.umn.edu/resources/suim-dataset):

| RGB color | Class ID |
| --- | ---: |
| `(0, 0, 0)` | 0 |
| `(0, 0, 255)` | 1 |
| `(0, 255, 0)` | 2 |
| `(0, 255, 255)` | 3 |
| `(255, 0, 0)` | 4 |
| `(255, 0, 255)` | 5 |
| `(255, 255, 0)` | 6 |
| `(255, 255, 255)` | 7 |

For an 8-bit RGB mask, this is `ID = 4 * (R > 127) + 2 * (G > 127) + (B > 127)`, with Boolean values converted to integers. Save the result as an unsigned 8-bit single-channel PNG. This follows the thresholding used by the [original data utilities](https://github.com/xahidbuffon/SUIM/blob/master/utils/data_utils.py).

**Six validation masks require the same alignment correction used in our experiments.** Their original masks are taller than their corresponding images. Crop the original RGB mask vertically using `mask[offset:offset + image_height, :]`, then convert its colors to class IDs. Offsets are zero-based; keep the entire width and do not resize the mask.

| Filename stem | Original mask width x height | Prepared mask width x height | Vertical offset |
| --- | --- | --- | ---: |
| `f_r_1070_` | 590 x 430 | 590 x 375 | 27 |
| `f_r_1302_` | 590 x 430 | 590 x 375 | 27 |
| `f_r_1866_` | 590 x 430 | 590 x 375 | 0 |
| `f_r_401_` | 910 x 490 | 910 x 435 | 27 |
| `f_r_829_` | 590 x 430 | 590 x 375 | 0 |
| `w_r_1_` | 590 x 430 | 590 x 375 | 27 |

These are project-specific corrections, not an official SUIM annotation update. Apply them only to these six original masks; do not crop an already corrected mask again. Without these corrections, the validation ground truth will differ from that used in the reported experiments.

The remaining 31 original image-mask size mismatches are all in the training pool. To reproduce our existing data preparation, preserve those mask dimensions during color conversion. The existing training transform resizes image and mask to the same sampled output size, using nearest-neighbor interpolation for the mask. Do not extend the six validation corrections to other samples or add a separate resize step during preparation; that would change the training preprocessing.

The resulting structure must match `splits/SUIM/val.txt`, for example:

```text
/path/to/datasets/SUIM/
├── Images/
│   └── d_r_132_.png
└── Masks/
    └── d_r_132_.png
```

Use the provided `splits/SUIM/val.txt` when comparing FTC-Seg validation results. The official `TEST/` split is a different evaluation set.

### ACDC: Download and Prepare

1. Visit the [official ACDC download page](https://acdc.vision.ee.ethz.ch/download), complete any required registration/login, and accept the dataset license yourself.
2. Download these two packages:
   - **`rgb_anon_trainvaltest.zip`**: anonymized RGB images.
   - **`gt_trainval.zip`**: semantic-segmentation annotations for training and validation.
3. Extract both packages. For each of `fog`, `night`, `rain`, and `snow`, collect the adverse-condition `train/` and `val/` images from `rgb_anon/<condition>/<split>/<sequence>/`. Copy the `*_rgb_anon.png` files into `Images/`, keeping their complete filenames.
4. From the corresponding `gt/<condition>/<split>/<sequence>/` directories, copy **`*_gt_labelTrainIds.png`** into `Masks/`, also keeping their complete filenames. These masks already encode class IDs `0–18` and ignore value `255`; no color conversion is required.

Use `labelTrainIds`, not `labelIds`, color visualizations, invalid-region masks, detection annotations, or panoptic annotations. The `*_ref` normal-condition data and the unlabeled official `test/` images are not part of this training/validation setup. Check for duplicate filenames before flattening directories; do not overwrite different files.

The resulting structure must match `splits/ACDC/val.txt`, for example:

```text
/path/to/datasets/ACDC/
├── Images/
│   └── GOPR0476_frame_000761_rgb_anon.png
└── Masks/
    └── GOPR0476_frame_000761_gt_labelTrainIds.png
```

The corresponding split-list entry is:

```text
Images/GOPR0476_frame_000761_rgb_anon.png Masks/GOPR0476_frame_000761_gt_labelTrainIds.png
```

After selecting only these adverse-condition training/validation files, `Images/` and `Masks/` must each contain 2,006 PNG files: 1,600 training pairs and 406 validation pairs. The larger RGB download also contains official test and normal-reference data, so the total archive count is not the prepared-directory count. Set `data_root` in `configs/ACDC.yaml` to this prepared root. Use the provided validation list and keep its images out of both training lists.

## Training

Run single-GPU training with OPR and full ATC:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/FSSG.yaml \
  --labeled-id-path /path/to/labeled.txt \
  --unlabeled-id-path /path/to/unlabeled.txt \
  --save-path ./exp/FTC-Seg/FSSG_5_full \
  --num-opr-layers 1 \
  --atc-mode full \
  --port 29701
```

Replace the split paths with your own files. To train on another dataset, select its YAML configuration and corresponding training lists. Training duration, batch size, learning rate, and crop size are configured in the YAML file.

`--num-opr-layers 1` applies OPR to the last selected backbone feature level. `--atc-mode full` enables both the difficulty and distribution-bias components of ATC.

### Logs and Checkpoints

The training output directory contains:

```text
exp/FTC-Seg/FSSG_5_full/
├── train.log
├── events.out.tfevents.*
├── latest.pth
└── best.pth
```

- `train.log` records training progress and validation results for the student and EMA teacher.
- TensorBoard event files record losses and evaluation metrics.
- `latest.pth` is updated after each completed epoch and validation.
- `best.pth` stores the checkpoint with the highest validation mIoU across the student and EMA teacher.

Checkpoints contain both `model` (student) and `model_ema` (EMA teacher). The `selected_branch` and `selected_miou` fields identify the better branch at the saved epoch. Optimizer and ATC statistics are also saved.

To resume, rerun the same training command with the same configuration and output directory; `latest.pth` is loaded automatically. Use a new `--save-path` for a new experiment.

View TensorBoard logs with:

```bash
tensorboard --logdir ./exp/FTC-Seg
```

## Inference and Evaluation

Use `inference.py` to evaluate a saved checkpoint and optionally export segmentation masks. Evaluation requires image-mask pairs with ground-truth annotations.

First, check which branch produced the best score:

```bash
python -c "import torch; c = torch.load('./exp/FTC-Seg/FSSG_5_full/best.pth', map_location='cpu'); print(c['selected_branch'], c['selected_miou'])"
```

Use the printed branch name for `--checkpoint-key`: `model` for the student or `model_ema` for the EMA teacher. The example below assumes the selected branch is `model_ema`:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --checkpoint ./exp/FTC-Seg/FSSG_5_full/best.pth \
  --checkpoint-key model_ema \
  --backbone dinov2_small \
  --num-opr-layers 1 \
  --nclass 11 \
  --data-root /path/to/dataset_root \
  --split-path splits/FSSG/val.txt \
  --mode original \
  --output-dir ./inference_results/FSSG_5_full \
  --save-predictions
```

The backbone, number of classes, and number of OPR layers must match the checkpoint. Use the same validation images, masks, and evaluation settings as training when comparing mIoU.

Choose evaluation settings as follows:

| Dataset | `--nclass` | `--mode` | Additional option |
| --- | ---: | --- | --- |
| FLSMD | 12 | `original` | None |
| FSSG | 11 | `original` | None |
| SUIM | 8 | `original` | None |
| ACDC | 19 | `sliding_window` | `--crop-size 798` |

For ACDC, also select its dataset root, checkpoint, and `splits/ACDC/val.txt`.

The inference output directory **must not already exist**. Each evaluation creates:

```text
inference_results/FSSG_5_full/
├── console.log
├── progress.jsonl
├── result.json
└── predictions/
```

`result.json` contains mIoU in percent, per-class IoU, and evaluation metadata. `predictions/` is created only when `--save-predictions` is set and contains class-ID PNG masks at the original image size.

## Notes

- Replace all `/path/to/...` placeholders before running the commands.
- The provided training configurations and the examples above use DINOv2-S. The inference script also includes compatibility support for historical ResNet-101 checkpoints with one OPR layer; it does not accept arbitrary architectures or checkpoint formats.
- ACDC has no dedicated background class. Its configuration treats all 19 classes as valid OPR target classes, so prototype gates are all one. Prototype reconstruction, residual connections, and orthogonality regularization remain active.


## License

The project code is released under the [MIT License](LICENSE). Please retain the copyright and license notices. Datasets and pretrained weights remain subject to their respective licenses and usage terms.

## Citation

Citation details for **When Noise Meets Long-Tail: Feature-Threshold Dual Calibration for Robust Pseudo-Labeling** will be added when available.
