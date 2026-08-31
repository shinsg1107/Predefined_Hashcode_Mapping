# Reference Results

This file lists the reference outputs for the main campaign configurations.
The values are provided for comparison with `results/main/main_eval.csv`.

## Running the Campaign

```bash
python run_main_campaign.py --devices 0 1 2 3
python eval_main_campaign.py --device cuda:0
```

The campaign configurations use `P_mismatch = 10`, `lambda = 0.5`, and
`beta = 0.1`.

Metrics are reported as follows:

- CIFAR-10: mAP@All and HDC@10
- ImageNet-100: mAP@1000 and HDC@100
- NUS-WIDE: mAP@5000 and HDC@100

HDC is reported as a percentage and lower values indicate less concentrated
code utilization. Set 2 is evaluated with the same trained model and without
additional training.

| Configuration | Dataset | Codes | Set 1 mAP | Set 1 HDC | Set 2 mAP | Set 2 HDC |
|---|---|---:|---:|---:|---:|---:|
| `cifar_vit16` | CIFAR-10 | 3,848 | 0.9400 | 32.53 | 0.9139 | 45.73 |
| `cifar_vit32` | CIFAR-10 | 938 | 0.8813 | 57.14 | 0.6962 | 47.86 |
| `cifar_swin16` | CIFAR-10 | 3,848 | 0.9713 | 53.27 | 0.9484 | 61.26 |
| `cifar_swin32` | CIFAR-10 | 938 | 0.8854 | 56.30 | 0.7134 | 49.31 |
| `cifar_eva16` | CIFAR-10 | 3,848 | 0.9865 | 74.33 | 0.9854 | 81.13 |
| `cifar_eva32` | CIFAR-10 | 938 | 0.9808 | 86.82 | 0.9420 | 86.58 |
| `in_vit16` | ImageNet-100 | 9,148 | 0.8740 | 61.04 | 0.8753 | 76.03 |
| `in_vit32` | ImageNet-100 | 7,676 | 0.8557 | 62.44 | 0.8769 | 78.96 |
| `in_swin16` | ImageNet-100 | 9,148 | 0.9277 | 82.68 | 0.9258 | 87.14 |
| `in_swin32` | ImageNet-100 | 7,676 | 0.9318 | 85.39 | 0.9317 | 88.51 |
| `in_eva16` | ImageNet-100 | 9,148 | 0.9317 | 82.55 | 0.9298 | 90.53 |
| `in_eva32` | ImageNet-100 | 7,676 | 0.9285 | 86.56 | 0.9318 | 92.60 |
| `nus_vit16` | NUS-WIDE | 5,657 | 0.8434 | 64.45 | 0.8308 | 62.32 |
| `nus_vit32` | NUS-WIDE | 3,315 | 0.8302 | 72.15 | 0.8025 | 68.68 |
| `nus_swin16` | NUS-WIDE | 5,657 | 0.8557 | 48.03 | 0.8487 | 53.74 |
| `nus_swin32` | NUS-WIDE | 3,315 | 0.8324 | 76.75 | 0.8073 | 73.72 |
| `nus_eva16` | NUS-WIDE | 5,657 | 0.8393 | 58.40 | 0.8328 | 58.73 |
| `nus_eva32` | NUS-WIDE | 3,315 | 0.8234 | 79.67 | 0.8048 | 77.13 |

Re-fine-tuning the image encoders can lead to small numerical differences.
