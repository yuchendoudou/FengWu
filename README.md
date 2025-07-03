# FengWu: The operational medium-range deterministic weather forecasting can be extended beyond a 10-day lead time

This repository presents the training code of FengWu, a deep learning-based weather forecasting model that pushes the skillful global weather forecasts beyond 10 days lead. 

## Requirements

```
pip install -r requirements.txt
```

## Train

```
./train_script.sh
```

## Files

```plain
├── config
│   ├── fengwu.yaml
│   ├── fengwu_finetune.yaml
├── datasets
│   ├── __init__.py
│   ├── era5_npy_f32.py
│   ├── era5_finetune_f32.py
│   ├── mean_std.json
│   ├── mean_std_single.json
├── models
│   ├── __init__.py
│   ├── model.py
│   ├── MTS2d_model.py
│   ├── MTS2D_finetune.py
├── networks
│   ├── __init__.py
│   ├── LGUnet_all.py
│   ├── utils
│   │   ├── Attention.py
│   │   ├── Blocks.py
│   │   ├── positional_encodings.py
│   │   ├── utils.py
├── replay
│   ├── replay_buff.py
├── utils
│   ├── __init__.py
│   ├── builder.py
│   ├── distributedsample.py
│   ├── logger.py
│   ├── misc.py
├── requirements.txt
├── train_script.sh
├── train.py
```

## Data Format

```plain
├── data
│   ├── single
│   │   ├── 1979
│   │   │   ├── 1979-01-01
│   │   │   │   ├─ 00:00:00-msl.npy
│   │   │   │   ├─ 00:00:00-t2m.npy
│   │   │   │   ├─ 00:00:00-u10.npy
│   │   │   │   ├─ 00:00:00-v10.npy
│   │   │   │   ├─ ...
│   │   │   │   ├─ 23:00:00-msl.npy
│   │   ├── 1980
│   │   ├── ...
│   │   ├── 2018
│   ├── 1979
│   │   ├── 1979-01-01
│   │   │   ├─ 00:00:00-q-50.0.npy
│   │   │   ├─ ...
│   │   │   ├─ 00:00:00-q-1000.0.npy
│   │   │   ├─ 00:00:00-r-50.0.npy
│   │   │   ├─ ...
│   │   │   ├─ 23:00:00-z-1000.0.npy
│   │   ├── 1979-01-02
│   │   ├── ...
│   │   ├── 1979-12-31
│   ├── 1980
│   ├── ...
│   ├── 2018
```

