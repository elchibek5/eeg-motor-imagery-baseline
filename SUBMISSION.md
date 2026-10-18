# Submission Notes

## What this produces
- `predictions/y_pred_<id>.npy` for each evaluation subject ID
- Each prediction file has shape `(N, 1)` and contains class labels.

## How to generate
    source .venv/bin/activate

    python3 run_pipeline.py \
      --train_dir path/to/Training \
      --eval_dir path/to/Evaluation \
      --out_dir predictions \
      --model_out model.joblib

    zip -r predictions.zip predictions

Submit `predictions.zip`.
