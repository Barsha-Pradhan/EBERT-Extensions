"""
transformers==3.3.1's TrainingArguments only exposes `warmup_steps` (an absolute
step count), not `warmup_ratio`. Since the required configuration specifies a
10% warmup ratio, this helper converts that ratio into an absolute step count
for a given GLUE task, batch size, and number of epochs, so it can be passed
as `--warmup_steps $(python scripts/compute_warmup_steps.py --task TASK --batch_size 16 --epochs 5)`.

Train-split sizes below are the standard GLUE numbers (single-GPU, single process).
"""
import argparse
import math

GLUE_TRAIN_SIZES = {
    "cola": 8551,
    "sst-2": 67349,
    "mrpc": 3668,
    "stsb": 5749,
    "sts-b": 5749,
    "qqp": 363846,
    "mnli": 392702,
    "qnli": 104743,
    "rte": 2490,
    "wnli": 635,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="GLUE task name, e.g. SST-2, MRPC, QNLI")
    parser.add_argument("--batch_size", type=int, required=True, help="per_device_train_batch_size (assumes 1 GPU)")
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    args = parser.parse_args()

    key = args.task.lower().replace("_", "-")
    if key not in GLUE_TRAIN_SIZES:
        raise ValueError(f"Unknown task '{args.task}'. Known tasks: {sorted(GLUE_TRAIN_SIZES)}")

    steps_per_epoch = math.ceil(GLUE_TRAIN_SIZES[key] / args.batch_size)
    total_steps = int(steps_per_epoch * args.epochs)
    warmup_steps = int(round(total_steps * args.warmup_ratio))
    print(warmup_steps)


if __name__ == "__main__":
    main()
