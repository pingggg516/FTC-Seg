# FTC-Seg

**When Noise Meets Long-Tail: Feature-Threshold Dual Calibration for Robust Pseudo-Labeling**.

## Installation

Training and inference have been tested on Linux with Python 3.10, PyTorch 1.12.1, and torchvision 0.13.1 using CUDA 11.3 builds.

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

Configurations are provided for four datasets:

| Dataset | Configuration | Number of classes | Validation list |
| --- | --- | ---: | --- |
| FLSMD | `configs/FLSMD.yaml` | 12 | `splits/FLSMD/val.txt` |
| FSSG | `configs/FSSG.yaml` | 11 | `splits/FSSG/val.txt` |
| SUIM | `configs/SUIM.yaml` | 8 | `splits/SUIM/val.txt` |
| ACDC | `configs/ACDC.yaml` | 19 | `splits/ACDC/val.txt` |

Prepare images and class-ID masks using the paths listed in the split files. For example:

```text
/path/to/dataset_root/
├── Images/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── Masks/
    ├── 00001.png
    ├── 00002.png
    └── ...
```

Set `data_root` in the corresponding configuration. For FSSG:

```yaml
dataset: FSSG
data_root: /path/to/dataset_root
```

## FSSG Dataset

FSSG contains **3,761 forward-looking sonar images** with **11 classes, including background**.

Download the dataset from [Baidu Netdisk](https://pan.baidu.com/s/10qCqdCkPZ_o7hk7KlC74nw?pwd=0516) (access code: `0516`).

### Class Definition

The class IDs stored in the masks are:

```text
0: Background
1: Block
2: Circle Cage
3: Steel Frame
4: Concrete Column
5: Steel Plate
6: Pot
7: SquareCage
8: Tire
9: Underwater Robot
10: Diver
```

### Usage Terms

The FSSG dataset is available for **non-commercial use only**.

If you use FSSG, cite its source paper and acknowledge the dataset source in resulting publications or research outputs. Dataset usage terms are separate from the code license.

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
