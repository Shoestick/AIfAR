#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import gi
import numpy as np
from ultralytics import YOLO
import cv2
import sys                    
import argparse               
from pathlib import Path
import datetime as dt   
       
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# ───────────────────────────────────────────────── helpers ──
@staticmethod
def _iou(bb1, bb2):
    """
    bb = [x1, y1, x2, y2] in absolute pixels
    returns IoU in [0, 1]
    """
    if bb1 is None or bb2 is None:
        return 0.0
    xA = max(bb1[0], bb2[0])
    yA = max(bb1[1], bb2[1])
    xB = min(bb1[2], bb2[2])
    yB = min(bb1[3], bb2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return inter / float(area1 + area2 - inter)
# ────────────────────────────────────────────────────────────

class VideoInterfaceNode(Node):
    def __init__(self,
                 detect_choice: str = 'area',  # 'area' or 'weighted'
                 save_frames: bool = False,         
                 save_dir: str = 'captured_frames'   
                 ):
        super().__init__('video_interface')
        
        # ───── frame‑capture config ────────────────────────────
        self.save_frames = save_frames              
        if self.save_frames:                        
            parent = Path(save_dir)             
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')  
            self.session_dir = parent / timestamp   
            self.session_dir.mkdir(parents=True, exist_ok=True)       
            self.frame_idx = 0                      
            #self.get_logger().info
            print(f'Saving frames into {self.session_dir}')

        if detect_choice in ['area', 'weighted']:
            self.detect_choice = detect_choice
        else:
            print("Defaulting to area detection")
            self.detect_choice = 'area'
            
            
        # Publisher: sends object position to the RELBot
        # Topic `/object_position` is watched by the robot controller for actuation
        self.position_pub = self.create_publisher(Point, '/object_position', 10)
        # ───── YOLOv8 model (load once) ────────────────────────────────────
        weights_path = 'best.pt' # Path to YOLOv8 weights
        self.model = YOLO(weights_path)
        self.get_logger().info(f'Loaded YOLOv8 weights: {weights_path}')
        
        # Declare GStreamer pipeline as a parameter for flexibility
        self.declare_parameter('gst_pipeline', (
            'udpsrc port=5000 caps="application/x-rtp,media=video,'
            'encoding-name=H264,payload=96" ! '
            'rtph264depay ! avdec_h264 ! videoconvert ! '
            'video/x-raw,format=RGB ! appsink name=sink'
        ))
        pipeline_str = self.get_parameter('gst_pipeline').value

        # Initialize GStreamer and build pipeline
        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name('sink')
        # Drop late frames to ensure real-time processing
        self.sink.set_property('drop', True)
        self.sink.set_property('max-buffers', 1)
        self.pipeline.set_state(Gst.State.PLAYING)

        # Timer: fires at ~30Hz to pull frames and publish positions
        # The period (1/30) sets how often on_timer() is called
        self.timer = self.create_timer(1.0 / 30.0, self.on_timer)
        self.get_logger().info('VideoInterfaceNode initialized, streaming at 30Hz')
        
        # Memory and confidence management
        self.confidence_threshold = 0.3  # Confidence threshold for detections
        self.last_box = None  # Last known position of the object

    def on_timer(self):
        # Pull the latest frame from the GStreamer appsink
        sample = self.sink.emit('pull-sample')
        if not sample:
            # No new frame available
            return

        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            # Failed to map buffer data
            return

        # Convert raw buffer to numpy array [height, width, channels]
        frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
        buf.unmap(mapinfo)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # --- optional saving ------------------------------------------------
        if self.save_frames:        
            print("made it to saving")   # debug statement                                        
            filename = self.session_dir / f'{self.frame_idx:06d}.jpg'      
            cv2.imwrite(str(filename), frame_bgr)                          
            self.frame_idx += 1
                        
        # --- Inference -------------------------------------------------------------------------
        # single Results object in list
        results = self.model(frame_bgr, imgsz=640, conf=self.confidence_threshold, device="cpu", stream=False)[0]
        boxes = results.boxes
        n = len(boxes)  # number of detections
        
        # SET placeholders for center_x and area
        centre_x = 0.0
        area = 0.0
        if n > 0:
            ### choose the *largest* detection, if any
            if self.detect_choice == 'area':
                # tensor: (n, 6) [x1, y1, x2, y2, conf, cls]
                boxes_xyxy = results.boxes.xyxy.cpu().numpy()    # (n,4)
                areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
                # find the largest area
                idx = int(areas.argmax())
                x1, y1, x2, y2 = boxes_xyxy[idx]

                # compute centre & area
                centre_x = (x1 + x2) / 2.0
                centre_x = centre_x / width * 320.0 - 20.0 # scale to 0-400 range
                area = areas[idx]
                
            ### Choose based on weighted average between Confidence, Area and IoU    
            elif self.detect_choice == 'weighted':
                # tensors → CPU numpy for convenience
                xyxy   = boxes.xyxy.cpu().numpy()        # (n,4)
                conf   = boxes.conf.cpu().numpy()        # (n,)
                areas  = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

                # --- build composite score -------------------------------------------------
                norm_area = np.sqrt(areas / float(width * height))      # 0-1
                ious = np.array([_iou(bb, self.last_box) for bb in xyxy]) \
                if self.last_box is not None else np.zeros(n, dtype=float)


                # weight hyper-parameters – tune to taste
                w_conf, w_area, w_iou = 0.55, 0.25, 0.20

                score = w_conf * conf + w_area * norm_area + w_iou * ious
                idx   = int(score.argmax())

                # selected box data
                x1, y1, x2, y2 = xyxy[idx]
                centre_x = (x1 + x2) / 2.0        # px
                centre_x = centre_x / width * 320.0 - 20.0
                area     = areas[idx]

                # update memory *before* publishing so next frame has it
                self.last_box = xyxy[idx].copy()
            
            
            ## TODO: Insert single depth estimation logic here to determine the distance
            # --- Single Image Depth Estimation -----------------------------------------------
            
            ## TODO: After single image depth estimation determine the proper z value
            # --- Calculate scaling factor for z ---------------------------------------------
            # update memory *before* publishing so next frame has it
            
            print(" Publshing object position:")
            print(f"centre_x: {centre_x}, area: {area}")
            msg = Point()
            msg.x = float(centre_x)      # horizontal position
            msg.y = 0.0                  # flat‑ground assumption
            msg.z = 500.0         # area acts as “distance” proxy
            self.position_pub.publish(msg)
        # optional: publish “no object” flag (e.g. area = ‑1) if nothing seen
        else: # If no object detected, publish the middle of the frame
            # Compute and publish object position:
            # x = horizontal center coordinate of the object
            # y = unused (flat-ground assumption)
            # z = object area (controller caps at 10000 to stop robot when object is too large)
            print("No object detected, publishing default position:")
            msg = Point()
            msg.x = 200.0  # object center x-coordinate
            msg.y = 0.0  # y-coordinate unused
            msg.z = 10001.0  # object area; >10000 indicates 'too close'
            self.position_pub.publish(msg)
            self.last_box = None  # Reset last box if no object detected
            # To adjust robot behavior, apply a scaling factor to 'z' (e.g., couple with depth estimation)
            # Log at debug level if needed:
            # self.get_logger().debug(f'Published position: ({msg.x}, {msg.y}, {msg.z})')

        # --- Visualisation -------------------------------------------------
        # Visualize the frame with YOLO detections
        annotated = results.plot()
        cv2.imshow('YOLOv8 Live', annotated)
        cv2.waitKey(1)
        
        # keeps the window responsive
        # Display the raw input frame for debugging
        #cv2.imshow('Input Stream', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        #cv2.waitKey(1)

        # TODO: Insert detection/tracking logic here to compute object position

        # TODO: Insert detection/tracking logic here to compute object position
        
        

    def destroy_node(self):
        # Cleanup GStreamer resources on shutdown
        self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


# def main(args=None):
#     rclpy.init(args=args)
#     node = VideoInterfaceNode()
#     try:
#         rclpy.spin(node)  # Keep node alive, invoking on_timer periodically
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

def main(argv: list[str] | None = None) -> None:
    """
    CLI entry-point.  Accepts both user flags and ROS2 launch args.

    recognised user flags
    ---------------------
    --detect_choice {area,weighted}
    --save_frames                 (store_true)
    --save_dir  <folder>
    """
    # ---------------------------------------------------------------------
    # Step 1 ‒ split "user" vs "ROS" arguments
    # ---------------------------------------------------------------------
    if argv is None:
        argv = sys.argv[1:]

    # rclpy provides a helper to *remove* ROS-specific args
    user_args = rclpy.utilities.remove_ros_args(args=argv)
    ros_args = [a for a in argv if a not in user_args]

    # ---------------------------------------------------------------------
    # Step 2 ‒ parse the remaining user arguments
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="RELBot video interface node",
        add_help=True)

    parser.add_argument("--detect_choice",
                        default="area",
                        choices=["area", "weighted"],
                        help="Strategy for selecting the target box")
    parser.add_argument("--save_frames",
                        action="store_true",
                        help="Save incoming frames as JPGs at 30 FPS")
    parser.add_argument("--save_dir",
                        default="captured_frames",
                        help="Root directory for saved frame sessions")

    opts = parser.parse_args(user_args)

    # ---------------------------------------------------------------------
    # Step 3 ‒ run the ROS2 node
    # ---------------------------------------------------------------------
    rclpy.init(args=ros_args)

    node = VideoInterfaceNode(detect_choice=opts.detect_choice,
                              save_frames=opts.save_frames,
                              save_dir=opts.save_dir)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()