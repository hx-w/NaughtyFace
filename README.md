
## Ref

- Age Estimation: https://github.com/yu4u/age-estimation-pytorch
- Face Anti-Spoofing: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing

## Installation

1. Install CMake, and `python -m pip install cmake boost`

2. Custom deps
    ```bash
    git clone https://github.com/hx-w/NaughtyFace

    cd NaughtyFace/

    python -m pip install -r requirements.txt # python <= 3.9
    ```

3. Download model(age estimation) parameters from [**release**](https://github.com/yu4u/age-estimation-pytorch/releases/tag/v1.0), put pth file in `age_estimator/misc`


## Usage

```bash
python video_offline_detect.py -i <video_path> -s <sample_rate>
```

**params**

- `--inputs / -i`: local video path, only mp4 supported (for now).
- `--sample_rate / -s`: video frames sample rate $\in [0.0, 1.0]$

