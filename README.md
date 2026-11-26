# A Study on Token Compression and Quantization for PTv3 Point Cloud Segmentation

This project studies inference-time token merging and reduced-precision inference for Point Transformer V3 (PTv3) on S3DIS Area 5.

## GitMerge3D Token Merging

The PTv3 integration adapts GitMerge3D's patch matching to merge similar tokens before attention and restore the original token layout afterward. The merge ratio and stride are configurable through `--r` and `--stride`.

## FP16 and INT8 Post-Training Quantization

- **FP16** runs a trained checkpoint in half precision without calibration.
- **INT8 (W8A8)** uses calibration data to set activation scales for linear layers and supported sparse convolutions, with per-output-channel INT8 weight quantization. The evaluator selects this path when given a `ptv3-w8a8-v1` checkpoint.

## Evaluation

From the repository root, use a configured Pointcept environment, preprocessed S3DIS data with an `Area_5` directory, and compatible local checkpoints.

```bash
export POINTCEPT_PYTHON="$(command -v python)" NUM_GPUS=1
eval_script=scripts/test_ptv3_s3dis_area5.sh
sh "$eval_script" --data-root data/s3dis_le_1m --weight path/to/model.pth --gitmerge3d --r 0.8 --stride 5
sh "$eval_script" --data-root data/s3dis_le_1m --weight path/to/model.pth --fp16
sh "$eval_script" --data-root data/s3dis_le_1m --weight path/to/model-w8a8.pth
```

## References and Acknowledgments

We thank the authors and maintainers of these projects for releasing their code.

- **Point Transformer V3 (PTv3)** for the segmentation backbone. [GitHub repository](https://github.com/Pointcept/PointTransformerV3)
- **Pointcept** for the point cloud training and evaluation framework. [GitHub repository](https://github.com/Pointcept/Pointcept)
- **GitMerge3D** for the token merging method and reference implementation. [GitHub repository](https://github.com/anhtuanhsgs/GitMerge3D)
