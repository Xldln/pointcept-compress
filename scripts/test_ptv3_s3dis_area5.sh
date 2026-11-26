#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

python_bin="${POINTCEPT_PYTHON:-/root/miniconda3/envs/pointcept-torch2.5.0-cu12.4/bin/python}"
output_root="${POINTCEPT_OUTPUT_ROOT:-$HOME/pointcept-eval}"
grid_size="${GRID_SIZE:-0.02}"
data_root="${POINTCEPT_DATA_ROOT:-data/s3dis_le_1m}"
legacy_sparse_features="${LEGACY_SPARSE_FEATURES:-true}"
test_workers="${TEST_WORKERS_PER_GPU:-2}"
model_dir="$PWD/exp/s3dis/ptv3-area5/model"
weight=

test_w8a8=false
test_fp16=false
seed=25326354
gitmerge3d=false
# ttr=false
randomdrop=false
ratio_set=false
stride_set=false
merge_ratio=0.8
merge_stride=5
while [ "$#" -gt 0 ]; do
    case "$1" in
        --int8) test_w8a8=true; shift ;;
        --fp16) test_fp16=true; shift ;;
        --randomdrop) randomdrop=true; shift ;;
        # --ttr) gitmerge3d=true; ttr=true; shift ;;
        --gitmerge3d) gitmerge3d=true; shift ;;
        --r|--stride)
            if [ -z "${2:-}" ]; then
                printf '%s requires a value\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --r) merge_ratio=$2; ratio_set=true ;;
                --stride) merge_stride=$2; stride_set=true ;;
            esac
            shift 2 ;;
        --weight|--data-root)
            if [ -z "${2:-}" ]; then
                printf '%s requires a path\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --weight) weight="$2" ;;
                --data-root) data_root="$2" ;;
            esac
            shift 2 ;;
        *)
            printf 'Usage: %s [--fp16 | --int8] [--weight PATH] [--data-root PATH] [--gitmerge3d | --randomdrop] [--r FRACTION] [--stride N]\n' "$0" >&2
            exit 2 ;;
    esac
done

if "$test_w8a8"; then
    if "$test_fp16"; then
        echo '--int8 and --fp16 select different checkpoints; choose one' >&2
        exit 2
    fi
    python_bin="${POINTCEPT_PYTHON:-$(command -v python)}"
    weight="${weight:-$model_dir/quant/ptv3_w8a8.pth}"
    # cumm discovers CUDA headers for INT8 kernel compilation through nvcc.
    export PATH="$(dirname "$python_bin"):$PATH"
else
    weight="${weight:-$model_dir/model_best.pth}"
fi

if "$randomdrop"; then
    if "$gitmerge3d" || "$stride_set"; then
        echo '--randomdrop cannot be combined with --gitmerge3d or --stride' >&2
        exit 2
    fi
elif "$ratio_set" || "$stride_set"; then
    gitmerge3d=true
fi

if [ ! -d "$data_root/Area_5" ]; then
    printf 'S3DIS Area_5 directory not found: %s/Area_5\n' "$data_root" >&2
    exit 1
fi
case "$test_workers" in
    *[!0-9]*)
        printf 'TEST_WORKERS_PER_GPU must be a non-negative integer.\n' >&2
        exit 2
        ;;
esac

if [ ! -x "$python_bin" ]; then
    printf 'Pointcept Python interpreter is unavailable: %s\n' "$python_bin" >&2
    exit 1
fi
if [ ! -r "$weight" ]; then
    printf 'Checkpoint is unreadable: %s\n' "$weight" >&2
    exit 1
fi

mkdir -p "$output_root"
# Use a fresh directory because SemSegTester reuses saved predictions.
save_path=$(mktemp -d "$output_root/ptv3-area5.XXXXXX")
printf 'Evaluation output: %s\n' "$save_path"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
set --
if "$randomdrop"; then
    set -- "data.test.test_cfg.randomdrop_ratio=$merge_ratio" \
        "data.test.test_cfg.randomdrop_seed=$seed"
fi
# Disabled option: "model.backbone.gitmerge3d_ttr=$ttr"
exec "$python_bin" -u tools/test.py \
    --config-file configs/s3dis/semseg-pt-v3m1-1-rpe.py \
    --num-gpus "${NUM_GPUS:-2}" \
    --num-machines 1 \
    --dist-url auto \
    --options "save_path=$save_path" "weight=$weight" "seed=$seed" \
    "data.test.data_root=$data_root" \
    "data.test.test_cfg.voxelize.grid_size=$grid_size" \
    "test_w8a8=$test_w8a8" \
    "test_fp16=$test_fp16" measure_forward_performance=true \
    "test_num_workers_per_gpu=$test_workers" \
    "model.backbone.legacy_sparse_features=$legacy_sparse_features" \
    "model.backbone.gitmerge3d=$gitmerge3d" \
    "model.backbone.gitmerge3d_r=$merge_ratio" \
    "model.backbone.gitmerge3d_stride=$merge_stride" "$@"
