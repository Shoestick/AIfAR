# 1 default (largest-area rule, no saving)
python3 video_interface.py

# 2 weighted rule, still no saving
python3 video_interface.py --detect_choice weighted

# 3 weighted rule + save JPEG frames into /tmp/run1/
python3 video_interface.py --detect_choice weighted \
                          --save_frames \
                          --save_dir /tmp/run1
