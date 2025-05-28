# 1 default (largest-area rule, no saving)
python3 video_interface.py

# 2 weighted rule, still no saving
python3 video_interface.py --detect_choice weighted

# 3 weighted rule + save JPEG frames into /tmp/run1/
python3 video_interface.py --detect_choice weighted \
                          --save_frames \
                          --save_dir /tmp/run1

# 4. Just saving for SLAM or running on bounding box and saving:
python3 video_interface.py --detect_choice weighted \
                          --save_frames \
                          --save_dir /SLAMrun1 \

python3 video_interface.py --detect_choice weighted --save_frames --save_dir /SLAM_run1

# 5. Running with SIDE instad of bounding box and saving
python3 video_interface.py --detect_choice weighted \
                          --save_frames \
                          --save_dir /SLAMrun1 \
                          --calculate_depth \

python3 video_interface.py --detect_choice weighted --save_frames --save_dir /SLAM_run1 --calculate_depth

| Flag | Default | Description |
|------|---------|-------------|
| `--detect_choice {area,weighted}` | `area` | **`area`** – chooses the single largest bounding box.  <br> **`weighted`** – combines YOLO confidence, box area (√-normalised), and IoU with the previous frame (weights = 0.55 / 0.25 / 0.20). |
| `--save_frames` | *false* | Save every raw camera frame as a JPEG. |
| `--save_dir PATH` | `captured_frames` | Root directory for saved frames. A timestamped sub-folder is created per run. |
| `--calculate_depth` | *false* | Enables MiDaS-based depth estimation. If omitted, distance is inferred from bounding-box size. |
| `--gst_pipeline STRING` | *(ROS 2 parameter)* | Override the default GStreamer pipeline if your video source differs. |