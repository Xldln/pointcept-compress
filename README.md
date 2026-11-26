# A Study on Token Compression and Quantization for PTv3 Point Cloud Segmentation

<p align="center">
  <a href="https://huggingface.co/yubinux/PTv3-PTQ"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Model%20%C2%B7%20PTv3--PTQ-cb8625?style=flat-square" alt="Hugging Face · PTv3-PTQ" /></a>
</p>

This project studies inference-time token merging and reduced-precision inference for Point Transformer V3 (PTv3) on S3DIS Area 5.

## GitMerge3D Token Merging

The PTv3 integration adapts GitMerge3D's patch matching to merge similar tokens before attention and restore the original token layout afterward. The merge ratio and stride are configurable through `--r` and `--stride`.

## FP16 and INT8 Post-Training Quantization

- **FP16** runs a trained checkpoint in half precision without calibration.
- **INT8 (W8A8)** uses calibration data to set activation scales for linear layers and supported sparse convolutions, with per-output-channel INT8 weight quantization. The evaluator selects this path when given a `ptv3-w8a8-v1` checkpoint.

### Download Checkpoints

Run these commands from the repository root to download the published weights. The INT8 command saves the checkpoint to the evaluator's default `--int8` path. [Hugging Face files](https://huggingface.co/yubinux/PTv3-PTQ/tree/main)

```bash
python -m pip install huggingface_hub

model_dir=exp/s3dis/ptv3-area5/model
hf download yubinux/PTv3-PTQ model_best_fp16.pth --local-dir "$model_dir"
hf download yubinux/PTv3-PTQ ptv3_w8a8.pth --local-dir "$model_dir/quant"

# Optional FP32 checkpoint for the baseline and GitMerge3D evaluation.
hf download yubinux/PTv3-PTQ model_best.pth --local-dir "$model_dir"
```

## Evaluation

From the repository root, use a configured Pointcept environment, preprocessed S3DIS data with an `Area_5` directory, and compatible local checkpoints.

```bash
export POINTCEPT_PYTHON="$(command -v python)" NUM_GPUS=1
eval_script=scripts/test_ptv3_s3dis_area5.sh
sh "$eval_script" --data-root data/s3dis_le_1m --gitmerge3d --r 0.8 --stride 5
sh "$eval_script" --data-root data/s3dis_le_1m --fp16 --weight exp/s3dis/ptv3-area5/model/model_best_fp16.pth
sh "$eval_script" --data-root data/s3dis_le_1m --int8
```

For INT8, use a CUDA environment with spconv, Triton, and the CUDA toolkit available. `--int8` defaults to `exp/s3dis/ptv3-area5/model/quant/ptv3_w8a8.pth` when `--weight` is omitted. Add `--gitmerge3d --r 0.7 --stride 4` to evaluate INT8 with token merging.

Export room predictions as colored PLY files using the same data root as the evaluation run.

```bash
python tools/export_s3dis_predictions.py path/to/run/result --data-root data/s3dis_le_1m
```

### S3DIS Area 5 Results

| Method | Type | `r` | Stride | Patch size | mIoU | mAcc | allAcc | Throughput (Mpts/s/GPU) | Mean AGM (MiB) | Peak AGM (GiB) | Weight size (MiB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PTv3 | Baseline | — | — | 128 | 80.05% | 84.65% | 93.56% | 1.0222 | 423.9 | 6.83 | 176.41 |
| PTv3 + RandomDrop | TC | 0.8 | — | 26 | 76.07% | 81.48% | 91.91% | 0.7225 | 384.4 | 3.64 | 176.41 |
| PTv3 + GitMerge3D | TC | 0.5 | 2 | 64 | 79.79% | 84.35% | 93.50% | 0.7480 | 423.9 | 6.55 | 176.41 |
| PTv3 + GitMerge3D | TC | 0.7 | 4 | 39 | 79.73% | 84.29% | 93.41% | 0.8088 | 423.9 | 6.42 | 176.41 |
| PTv3 + GitMerge3D | TC | 0.8 | 5 | 26 | 78.90% | 83.42% | 93.10% | 0.8283 | 423.9 | 6.37 | 176.41 |
| PTv3 + GitMerge3D | TC | 0.9 | 10 | 13 | 76.87% | 81.52% | 92.22% | 0.8816 | 423.9 | 6.34 | 176.41 |
| PTv3 FP16 | PTQ | — | — | 128 | 80.05% | 84.65% | 93.56% | 1.4035 | 237.9 | 5.51 | 88.27 |
| PTv3 INT8 | PTQ | — | — | 128 | 80.06% | 84.67% | 93.57% | 1.2970 | 106.8 | 5.39 | 44.78 |
| PTv3 INT8 + GitMerge3D | TC + PTQ | 0.7 | 4 | 39 | 79.75% | 84.32% | 93.42% | 0.9864 | 106.8 | 4.82 | 44.78 |
| PTv3 INT8 + GitMerge3D | TC + PTQ | 0.8 | 5 | 26 | 78.92% | 83.46% | 93.12% | 1.0280 | 106.8 | 4.78 | 44.78 |

*Table 1. PTv3 semantic segmentation on S3DIS Area 5 with token compression, FP16 inference, and INT8 post-training quantization. A dash indicates that a parameter does not apply.*

The full experimental results will be available in the [PTv3-PTQ repository on Hugging Face](https://huggingface.co/yubinux/PTv3-PTQ/tree/main).

- **Method types.** TC stands for token compression. The post-training quantization (PTQ) group includes FP16 as an uncalibrated half-precision comparison and INT8 as calibrated W8A8, which quantizes both weights and activations in supported layers.
- **Compression parameters.** `r` is the target fraction of tokens merged within each GitMerge3D attention patch, or the fraction of input points dropped by RandomDrop before inference. GitMerge3D uses the stride as the bin width and selects one destination token per full bin. Patch size reports the token count for a full 128-token patch after GitMerge3D. For RandomDrop, 26 is the equivalent retained count out of 128 at `r = 0.8`. RandomDrop samples points per room before the model forms attention patches.
- **Segmentation metrics.** mIoU is the mean of per-class intersection-over-union scores. mAcc is the mean of per-class accuracies. allAcc is the fraction of all evaluated points classified correctly.
- **Efficiency metrics.** Throughput is measured in millions of model input points per second per GPU, using forward-pass time after five warm-up passes per GPU. AGM stands for allocated GPU memory. Mean AGM averages the allocation at the end of each timed forward pass. Peak AGM is the highest allocation recorded during a timed forward pass. Weight size is the on-disk size of the model weights.

## References and Acknowledgments

We thank the authors and maintainers of these projects for releasing their code.

- **Point Transformer V3 (PTv3)** for the segmentation backbone. [GitHub repository](https://github.com/Pointcept/PointTransformerV3)
- **Pointcept** for the point cloud training and evaluation framework. [GitHub repository](https://github.com/Pointcept/Pointcept)
- **GitMerge3D** for the token merging method and reference implementation. [GitHub repository](https://github.com/anhtuanhsgs/GitMerge3D)
