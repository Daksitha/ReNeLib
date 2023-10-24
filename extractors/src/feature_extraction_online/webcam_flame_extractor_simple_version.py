import cv2
import time
import mediapipe as mp
from threading import Lock
import torch
from gdl_apps.EMOCA.utils.load import load_model
import argparse
from pathlib import Path
import gdl
from gdl_apps.EMOCA.utils.io import test
from gdl.utils.lightning_logging import _fix_image
from skimage.transform import rescale, estimate_transform, warp
import numpy as np
import sys
import json
import zmq
import logging
import sys, traceback
# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from threading import Thread
from collections import deque
from threading import Event
stop_event = Event()

def capture_frames(webcam_stream, frame_deque):
    while not stop_event.is_set():
        frame = webcam_stream.read()
        frame_deque.append(frame)  # Older frames will be discarded automatically if deque is full
        time.sleep(0.01)  # Sleep to avoid busy-waiting


def process_frames(webcam_stream, frame_deque, emoca_model, publisher, arguments):
    try:
        while not stop_event.is_set():
            start_time = time.time()
            if not frame_deque:
                time.sleep(0.01)  # Sleep if deque is empty
                continue
            frame = frame_deque.popleft()  # Retrieve and remove the leftmost frame
            #frame = frame_queue.get(timeout=1)  # Use timeout to avoid blocking indefinitely
            cropped_face = webcam_stream.get_cropped_face(frame)
            if cropped_face is not None:
                vals, visdict = test(emoca_model, cropped_face)

                exp_nparr = vals["expcode"][0].detach().cpu().numpy()
                pose_nparr = vals["posecode"][0].detach().cpu().numpy()

                flame_frame = np.concatenate((exp_nparr, pose_nparr), axis=None)
                timestamp = time_hns()
                flame_vec = flame_frame.tolist()

                elapsed_time = time.time() - start_time
                zmq_send_frame(publisher, flame_vec, arguments.topic, logger, elapsed_time)

                if arguments.show_overlay:
                    mesh_ov_img = _fix_image(torch_img_to_np(visdict['output_images_coarse'][0]))
                    cv2.imshow('Cropped Face', mesh_ov_img)
            cv2.imshow('Webcam Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        webcam_stream.stop()
class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id
        self.vcap = cv2.VideoCapture(self.stream_id)
        if not self.vcap.isOpened():
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)

        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        self.grabbed, self.frame = self.vcap.read()
        if not self.grabbed:
            print('[Exiting] No more frames to read')
            exit(0)

        self.stopped = True
        self.lock = Lock()
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

        # Initialize MediaPipe Face Detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.2)

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while not stop_event.is_set():
            if self.stopped:
                break
            grabbed, frame = self.vcap.read()
            if not grabbed:
                print('[Exiting] No more frames to read')
                self.stop()
                break
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)  # Reduce CPU usage
        self.vcap.release()

    def read(self):
        with self.lock:
            frame = self.frame.copy()
        return frame

    def stop(self):
        self.stopped = True

    def get_cropped_face(self, frame):
        # Check for empty frame
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            try:
                bboxC = results.detections[0].location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Check for non-positive width or height of the cropped face
                if w <= 0 or h <= 0:
                    return None

                cropped_face = frame[y:y + h, x:x + w]

                # Normalize pixel values to [0, 1]
                cropped_face = cropped_face / 255.0

                # Define the destination points and resolution
                resolution_inp = 224
                DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])

                # Define the source points
                src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

                # Estimate the similarity transform and apply it
                tform = estimate_transform('similarity', src_pts, DST_PTS)
                dst_image = warp(cropped_face, tform.inverse, output_shape=(resolution_inp, resolution_inp))
                dst_image = dst_image.transpose(2, 0, 1)

                # Convert to PyTorch tensor
                dst_image_tensor = torch.tensor(dst_image).float()

                return {
                    'image': dst_image_tensor,
                }
            except Exception:
                logger.error("In the get_cropped_face() function. Maybe this is due to unavailable face")
                return None


        return None


# Usage
def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)

def time_hns():
    """Return time in 100 nanoseconds (more precise with >= python 3.7)"""

    # Python 3.7 or newer use nanoseconds
    if (sys.version_info.major == 3 and sys.version_info.minor >= 7) or sys.version_info.major >= 4:
        # 100 nanoseconds / 0.1 microseconds
        time_now = int(time.time_ns() / 100)
    else:
        # timestamp = int(time.time() * 1000)
        # timestamp = timestamp.to_bytes((timestamp.bit_length() + 7) // 8, byteorder='big')

        # match 100 nanoseconds / 0.1 microseconds
        time_now = int(time.time() * 10000000)  # time.time()
        # time_now = time.time()

    return time_now

def zmq_send_frame(publisher, flame_vec, topic, logger, elapsed_time):
    key_name = topic
    timestamp = time_hns()
    timestamp_enc = str(timestamp).encode('ascii')
    key = key_name.encode('ascii')

    json_data = {
        f"{key_name}": flame_vec,
        "timestamp_utc": timestamp
    }
    encoded_data = json.dumps(json_data).encode('utf8')
    publisher.send_multipart([key, timestamp_enc, encoded_data])

    fps = 1 / elapsed_time
    logger.info(f'Published at:{time.time()}, FPS: {fps:.2f}')

def main():


    # torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    # add the input folder arg
    parser.add_argument('--output_folder', type=str, default="webcam_output",
                        help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str,
                        default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False,
                        help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False, help="If true, output meshes will be saved")
    parser.add_argument('--mode', type=str, default='detail', help="coarse or detail")

    parser.add_argument('--ip', type=str, default='127.0.0.1', help="ip address to stream extracted FLAME features ")
    parser.add_argument('--port', type=str, default='5556', help="port for FLAME features")
    parser.add_argument('--topic', type=str, default='flame', help="")

    parser.add_argument('--batch_size', type=int, default=24, help="number of video frames to process FLAME")
    parser.add_argument('--webcam_fps', type=int, default=24, help="check your webcam fps and set it here")
    parser.add_argument('--show_overlay', type=bool, default=True, help="show cv2 image and morphed 3D face")

    arguments = parser.parse_args()

    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_models = arguments.path_to_models
    model_name = arguments.model_name
    output_folder = arguments.output_folder + "/" + model_name

    mode = arguments.mode

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    # publisher.bind("tcp://127.0.0.1:8867")
    publisher.bind(f"tcp://{arguments.ip}:{arguments.port}")

    # 1) Load the model
    with torch.no_grad():
        emoca_model, conf = load_model(path_to_models, model_name, mode)
        emoca_model.cuda()
        emoca_model.eval()

    webcam_stream = WebcamStream()
    webcam_stream.start()

    # frame_deque = deque(maxlen=30)
    # capture_thread = Thread(target=capture_frames, args=(webcam_stream, frame_deque))
    # process_thread = Thread(target=process_frames, args=(webcam_stream, frame_deque, emoca_model, publisher, arguments))
    #
    # capture_thread.start()
    # process_thread.start()
    #
    # capture_thread.join()
    # process_thread.join()
    #
    try:
        while not stop_event.is_set():
            start_time = time.time()
            frame = webcam_stream.read()
            cropped_face = webcam_stream.get_cropped_face(frame)
            if cropped_face is not None:
                vals, visdict = test(emoca_model, cropped_face)

                exp_nparr = vals["expcode"][0].detach().cpu().numpy()
                pose_nparr = vals["posecode"][0].detach().cpu().numpy()

                flame_frame = np.concatenate((exp_nparr, pose_nparr), axis=None)
                timestamp = time_hns()
                flame_vec = flame_frame.tolist()

                elapsed_time = time.time() - start_time
                zmq_send_frame(publisher, flame_vec, arguments.topic, logger, elapsed_time)

                if arguments.show_overlay:
                    mesh_ov_img = _fix_image(torch_img_to_np(visdict['output_images_coarse'][0]))
                    cv2.imshow('Cropped Face', mesh_ov_img)
                    cv2.imshow('Webcam Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        print(traceback.format_exc())

    finally:
        cv2.destroyAllWindows()
        webcam_stream.stop()

if __name__ == "__main__":
    main()
