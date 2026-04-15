# Person Tracking Pipeline

Multi-person detection and tracking on NVIDIA Jetson using DeepStream 7.1.

## Demo

| Input | Output |
|:--:|:--:|
| ![input](demo/preview_input.jpg) | ![tracked](demo/preview_tracked.jpg) |

[Download input video (MOT16-08)](demo/input_MOT16-08.mp4) | [Download output video](demo/output_tracked.mp4)

## Pipeline

```
filesrc → qtdemux → h264parse → nvv4l2decoder (NVDEC)
→ nvstreammux → nvinfer (PeopleNet) → nvtracker (NvDeepSORT)
→ nvdsosd → nvvideoconvert → nvv4l2h264enc (NVENC) → mp4mux → filesink
```

- **PeopleNet v2.3.4**: ResNet-34 detector, INT8 quantized, 960×544 input, 3 classes (person/bag/face)
- **NvDeepSORT + OSNet**: 512-dim re-ID features (MSMT17-trained), occlusion recovery up to ~2s
- Full GPU pipeline: NVDEC → TensorRT → NVENC, no CPU-GPU copies

## Design Choices

**Why PeopleNet over YOLOv8?** PeopleNet is trained specifically on large-scale pedestrian datasets, whereas YOLOv8 is a general-purpose 80-class detector where "person" is just one of many categories. PeopleNet ships with an official INT8 calibration table and integrates natively with DeepStream on Jetson — no custom ONNX export or post-processing parser needed. In dense pedestrian scenes, a specialized detector consistently outperforms a general one.

**Why NvDeepSORT over simple SORT/ByteTrack?** Pure motion-based trackers (Kalman + IoU matching) lose track IDs the moment a person is occluded. NvDeepSORT adds a 512-dimensional appearance descriptor (OSNet, trained on 1,041 identities) so that when a person reappears after occlusion, the tracker can re-identify them by visual similarity rather than just position. This is critical in crowded scenes with frequent occlusions.

**Why DeepStream over OpenCV + TensorRT?** DeepStream provides an end-to-end GPU pipeline where decode, inference, tracking, and encode run in parallel with zero-copy memory (NVMM). Achieving the same with OpenCV would require manual thread management, explicit `cudaMemcpy` calls, and significantly more code for comparable throughput.

## Quick Start

### 1. Build

```bash
sudo docker build --network=host -t person-tracker .
```

When the build succeeds, you should see:

```
Step 11/12 : ENTRYPOINT ["python3", "main.py"]
...
Successfully built xxxxxxxxxx
Successfully tagged person-tracker:latest
```

### 2. Run

For example, if your input video is `video.mp4` located in `~/Downloads`:

```bash
mkdir -p output
sudo docker run --runtime=nvidia --network=host \
  -v /home/j4012/Downloads:/data \
  -v $(pwd)/output:/app/output \
  person-tracker \
  --input /data/video.mp4 \
  --output /app/output/tracked.mp4 2>&1 | tee run.log
```

First run compiles TensorRT engines (~9 min). Subsequent runs start instantly.

When the video is processed successfully, you should see:

```
===== NvVideo: NVENC =====
...
[Pipeline] Input:  /data/video.mp4
[Pipeline] Output: /app/output/tracked.mp4
[Pipeline] Processing...
[Pipeline] EOS
[Pipeline] Done. 625 frames in 42.0s (14.9 fps)
[Pipeline] Output saved: /app/output/tracked.mp4
```

The tracked output video can be found in the `output/` folder.

## Requirements

- NVIDIA Jetson (JetPack 6.x / L4T R36.x)
- Docker with [NVIDIA Container Runtime](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/) (`sudo apt install nvidia-container-runtime`) 
- H.264 MP4 input video

**Do NOT install `nvidia-cuda-toolkit` from the Ubuntu apt repository on Jetson.**
Jetson's CUDA is pre-installed via JetPack/L4T. The Ubuntu generic `nvidia-cuda-toolkit` package pulls in `libnvidia-compute-590-server` (an x86/server driver library) as a recommended dependency, which overwrites the native NVML library and causes a driver/library version mismatch

## Performance

| Metric | Value |
|--------|-------|
| FPS | ~15 (with re-ID) |
| Input | 1920×1080 @ 30fps |
| Detection | INT8 |
| Re-ID | FP16 |
