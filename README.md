# FTC-Seg

**When Noise Meets Long-Tail: Feature-Threshold Dual Calibration for Robust Pseudo-Labeling**.

## Installation

```bash
cd FTC-Seg
conda create -n FTC-Seg python=3.10 -y
conda activate FTC-Seg
python -m pip install -r requirements.txt \
  "numpy<2" "opencv-python<4.12" \
  torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

Run the following commands from the repository root.

## Pretrained Backbone

Download the [DINOv2 ViT-S/14 pretrained backbone](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) and save it as:

```text
pretrained/dinov2_small.pth
```

## Datasets

| Dataset | Download |
| --- | --- |
| FSSG | [Baidu Netdisk](https://pan.baidu.com/s/10qCqdCkPZ_o7hk7KlC74nw?pwd=0516) · Code: `0516` |
| FLSMD | Link and access code coming soon |
| SUIM | [Official website](https://irvlab.cs.umn.edu/resources/suim-dataset) |
| ACDC | [Official website](https://acdc.vision.ee.ethz.ch/download) |

FSSG and FLSMD are provided in the required format. Prepare SUIM and ACDC as described below.

### Data Format

```text
dataset_root/
├── Images/
└── Masks/
```

Set the dataset root in `configs/<DATASET>.yaml`:

```yaml
data_root: /path/to/dataset_root
```

Each split-file line contains an image path and a mask path relative to `data_root`:

```text
Images/example.png Masks/example.png
```

Masks must be single-channel class-ID images. Training automatically uses `splits/<dataset>/val.txt`, where `<dataset>` is the YAML `dataset` value.

### SUIM

Download `SUIM.zip` from the official website. Use the 1,525 image-mask pairs in `train_val/`:

- **Training:** 1,219 pairs.
- **Validation:** 306 pairs, specified by `splits/SUIM/val.txt`.

The official `TEST/` split contains another 110 pairs and is not used in these experiments.

1. Convert images from `train_val/images/` to PNG under `Images/`, preserving filename stems and image dimensions.
2. Convert the RGB masks directly under `train_val/masks/` to single-channel PNGs under `Masks/`, preserving filename stems.
3. Use the following RGB-to-class-ID mapping, saving IDs as unsigned 8-bit integers:

```text
class_id = 4 × int(R ≥ 128) + 2 × int(G ≥ 128) + int(B ≥ 128)
```

To reproduce our validation preprocessing, crop these six original masks vertically before conversion:

| Filename stem | Vertical offset |
| --- | ---: |
| `f_r_1070_` | 27 |
| `f_r_1302_` | 27 |
| `f_r_1866_` | 0 |
| `f_r_401_` | 27 |
| `f_r_829_` | 0 |
| `w_r_1_` | 27 |

Use `mask[offset:offset + image_height, :]`, retaining the full width. These are project-specific alignment corrections; apply them only to the six original masks.

Preserve the original training-mask dimensions during conversion. The training transform handles resizing.

### ACDC

Download these packages from the official website:

- `rgb_anon_trainvaltest.zip`
- `gt_trainval.zip`

Use only the adverse-condition `train/` and `val/` subsets of `fog`, `night`, `rain`, and `snow`:

- **Training:** 1,600 pairs.
- **Validation:** 406 pairs, specified by `splits/ACDC/val.txt`.

Copy:

```text
rgb_anon/<condition>/<split>/<sequence>/*_rgb_anon.png
    → Images/

gt/<condition>/<split>/<sequence>/*_gt_labelTrainIds.png
    → Masks/
```

Preserve complete filenames. Use `labelTrainIds` masks, which already contain class IDs `0–18` and ignore value `255`. Exclude official test images and normal-condition reference images.

The prepared `Images/` and `Masks/` directories should each contain 2,006 PNG files.

Example split entry:

```text
Images/GOPR0476_frame_000761_rgb_anon.png Masks/GOPR0476_frame_000761_gt_labelTrainIds.png
```

### Dataset Sources

Please cite the corresponding dataset papers:

- **FSSG:** [CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels](https://arxiv.org/abs/2603.21071). Non-commercial use only. Please acknowledge the [original dataset release](https://github.com/pingggg516/CTFS).
- **FLSMD:** [The Marine Debris Dataset for Forward-Looking Sonar Semantic Segmentation](https://arxiv.org/abs/2108.06800) and [The Marine Debris Forward-Looking Sonar Datasets](https://arxiv.org/abs/2503.22880). Our copy reformats the [original data](https://zenodo.org/records/15101686) and retains its [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **SUIM:** [Semantic Segmentation of Underwater Imagery: Dataset and Benchmark](https://arxiv.org/abs/2004.01241). Follow the original provider's usage terms.
- **ACDC:** [ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding](https://acdc.vision.ee.ethz.ch/citation). Follow the [official dataset license](https://acdc.vision.ee.ethz.ch/license).

## Training

Run single-GPU training with OPR and ATC:

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

For another dataset, select its configuration and training lists. Set epochs, batch size, and learning rate in the YAML configuration.

### Training Outputs

```text
exp/FTC-Seg/FSSG_5_full/
├── train.log
├── events.out.tfevents.*
├── latest.pth
└── best.pth
```

- `train.log`: training progress and student/EMA validation results.
- `events.out.tfevents.*`: TensorBoard metrics.
- `latest.pth`: checkpoint saved after each completed epoch and validation.
- `best.pth`: checkpoint with the highest validation mIoU across the student and EMA teacher.

Checkpoints include `model`, `model_ema`, optimizer state, and ATC statistics. `selected_branch` and `selected_miou` identify the selected branch and its score.

To resume training, rerun the same command with the same configuration and output directory. `latest.pth` is loaded automatically.

```bash
tensorboard --logdir ./exp/FTC-Seg
```

## Inference

Check the selected checkpoint branch:

```bash
python -c "import torch; c = torch.load('./exp/FTC-Seg/FSSG_5_full/best.pth', map_location='cpu'); print(c['selected_branch'], c['selected_miou'])"
```

Set `--checkpoint-key` to the printed branch: `model` or `model_ema`.

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

The backbone, class count, and OPR layer count must match the checkpoint. Use the same validation data and evaluation settings as training.

| Dataset | `--nclass` | `--mode` | Additional option |
| --- | ---: | --- | --- |
| FLSMD | 12 | `original` | — |
| FSSG | 11 | `original` | — |
| SUIM | 8 | `original` | — |
| ACDC | 19 | `sliding_window` | `--crop-size 798` |

`original` evaluates whole images; `sliding_window` evaluates overlapping crops. For ACDC, use the same crop size as training validation.

Use a new output directory for each evaluation:

```text
inference_results/FSSG_5_full/
├── console.log
├── progress.jsonl
├── result.json
└── predictions/
```

`result.json` contains mIoU, per-class IoU, and evaluation settings. `predictions/` is created when `--save-predictions` is enabled.

## Notes

- The provided configurations use DINOv2-S.
- Historical ResNet-101 checkpoints with one OPR layer are also supported by the inference script.
- ACDC treats all 19 classes as OPR target classes. Prototype gates are therefore all one; reconstruction, residual connections, and orthogonality regularization remain active.

## License

This project is released under the [MIT License](LICENSE).

## Citation

Citation information will be added when available.
