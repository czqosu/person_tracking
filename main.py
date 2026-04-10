#!/usr/bin/env python3
"""
Person detection and tracking using NVIDIA DeepStream + PeopleNet.

Usage:
    python3 main.py --input video.mp4 --output tracked.mp4
"""

import argparse
import sys
from pipeline.tracking_pipeline import TrackingPipeline


def main():
    parser = argparse.ArgumentParser(description="Person tracking with DeepStream")
    parser.add_argument("--input",  required=True, help="Input MP4 video path")
    parser.add_argument("--output", default="output/tracked.mp4", help="Output MP4 path")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    pipeline = TrackingPipeline(args.input, args.output)
    pipeline.build()
    pipeline.run()


if __name__ == "__main__":
    main()
