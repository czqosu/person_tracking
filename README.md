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

```bash
docker build --network=host -t person-tracker .

mkdir -p output
docker run --runtime=nvidia \
  -v /path/to/your/videos:/data \
  -v $(pwd)/output:/app/output \
  person-tracker \
  --input /data/video.mp4 \
  --output /app/output/tracked.mp4
```

First run compiles TensorRT engines (~9 min). Subsequent runs start instantly.

## Requirements

- NVIDIA Jetson (JetPack 6.x / L4T R36.x)
- Docker with [NVIDIA Container Runtime](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/) (`sudo apt install nvidia-container-runtime`) 
- H.264 MP4 input video

## Performance

| Metric | Value |
|--------|-------|
| FPS | ~15 (with re-ID) |
| Input | 1920×1080 @ 30fps |
| Detection | INT8 |
| Re-ID | FP16 |
