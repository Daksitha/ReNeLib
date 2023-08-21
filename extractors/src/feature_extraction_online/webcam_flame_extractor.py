"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
import logging

from scalene import scalene_profiler
from gdl_apps.EMOCA.utils.load import load_model
from ImageTestDatasetLocal import TestWebcamData
import gdl
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
from gdl.utils.lightning_logging import _fix_image
import cv2
import multiprocessing
import threading
import queue
import zmq
import time
import sys
from gdl.utils.lightning_logging import _fix_image
import json
from collections import deque

# Initialize queue for storing batches to be processed
batch_queue = deque(maxlen=100)
image_queue = deque(maxlen=40)

# Set up logging configuration
logging.basicConfig(filename='flame_extractor.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# evaluation measurements
# EVALUATION = False
# capture_latencies = []
# processing_latencies = []
# publishing_latencies = []
# lock = threading.Lock()


stop_all_event = threading.Event()


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
def zmq_send_array(zmqsocket, arry: np.array, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(arry.dtype),
        shape=arry.shape,
    )
    zmqsocket.send_json(md, flags | zmq.SNDMORE)
    return zmqsocket.send(arry, flags, copy=copy, track=track)

def zmq_send_frame(publisher, flame_json, topic):
    zmq_send_frame_start = time.time()
    #logging.info(f'zmq_send_frame start: {zmq_send_frame_start}s')
    key_name = topic
    timestamp = time_hns()
    timestamp_enc = str(timestamp).encode('ascii')
    key = key_name.encode('ascii')
    # fill the queue with padding values zeros
    for ts,flame_vec in flame_json.items():
        json_data = {
            f"{key_name}": flame_vec,
            "timestamp_utc": ts
        }
        #print(json_data)
        encoded_data = json.dumps(json_data).encode('utf8')
        publisher.send_multipart([key, timestamp_enc, encoded_data])

    logging.info(f'Publish:{time.time() - zmq_send_frame_start}')



def extract_features(input_image_list, args, publisher, emoca, output_folder, show_overlay):
    extractor_start = time.time()
    #logging.info(f'extract_features started: {extractor_start}s')

    # Create a dataset
    dataset = None
    dataset = TestWebcamData(input_image_list, face_detector='mp', max_detection=1)

    ## Run the model on the data
    batch_size = args.batch_size
    flame_vector_len = 56
    #flame_array = np.empty((batch_size,flame_vector_len), dtype=np.float32)

    # json data to send
    json_flame = {}
    exp_nparr = None
    pose_nparr= None
    flame_frame = None
    img = None


    with torch.no_grad():
        for j, batch in enumerate(auto.tqdm(dataset, desc="Webcam FLAME extractor", disable=True)):
            current_bs = batch["image"].shape[0]
            img = batch
            vals, visdict = test(emoca, img)
            for i in range(current_bs):
                #name = batch["image_name"][i]


                try:
                    exp_nparr = vals["expcode"][i].detach().cpu().numpy()
                    pose_nparr = vals["posecode"][i].detach().cpu().numpy()

                    flame_frame = np.concatenate((exp_nparr, pose_nparr), axis=None)
                    # use numpy if you want to send via zmq_send_array
                    #flame_array[j]= flame_frame

                    # prepare json object
                    timestamp = time_hns()
                    json_flame[f"{timestamp}"] = flame_frame.tolist()

                    if show_overlay:
                        mesh_ov_img = _fix_image(torch_img_to_np(visdict['output_images_coarse'][i]))
                        image_queue.append(mesh_ov_img)
                    #cv2.imshow('overlay', mesh_ov_img)
                except IndexError as ie:
                    pass

    logging.info(f'Process:{time.time()-extractor_start}')

    #zmq_send_array(publisher, flame_array)

    # Calculate publishing latency
    #publishing_start = time.time()
    zmq_send_frame(publisher, json_flame, topic=args.topic)

    publishing_end = time.time()
    #if EVALUATION:
        #publishing_latencies.append(publishing_end - publishing_start)

    del dataset, vals, visdict




# Define function for processing batches in a thread

def process_batch(emoca_args, publisher , emoca_model, output_folder, show_overlay, stop_all_event):
    #publish_counter = 0

    while not stop_all_event.set():
        # Get batch from queue
        processing_start = time.time()
        if len(batch_queue):
            batch = batch_queue.pop()

            # Extract features from batch
            extract_features(input_image_list=batch, args=emoca_args,publisher=publisher,
                             emoca=emoca_model, output_folder=output_folder, show_overlay=show_overlay)

            processing_end = time.time()



def main(arguments, DEBUG=False):


    ###### webcam implementation
    # Initialize webcam
    source=0
    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        raise RuntimeError('Warning: unable to open video source: ', source)

    start_time = time.time()
    num_frames = 0
    # Initialize empty batch
    batch = []

    # Initialize empty batch and feature list
    batch = []
    features = []

    # initialise the zeromq publisher
    # Set up ZeroMQ publisherVIDEOIO
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    #publisher.bind("tcp://127.0.0.1:8867")
    publisher.bind(f"tcp://{arguments.ip}:{arguments.port}")
    show_overlay = arguments.show_overlay

    # Start threads for processing batches
    #for i in range(num_threads):
    t = threading.Thread(target=process_batch, args=(arguments, publisher, emoca_model, output_folder,
                                                     show_overlay, stop_all_event))
    t.daemon = True
    t.start()

    # Loop through video stream
    capture_start = time.time()
    #logging.info(f"Webcam capture time start:{capture_start}")
    evaluation_counter =0
    while not stop_all_event.set():
        # Capture frame from webcam

        ret, image_rgb = cap.read()

        # format the incoming frame
        #image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        # let's downscale the image using new  width and height
        down_width = 300
        down_height = 200
        down_points = (down_width, down_height)
        frame = cv2.resize(image_rgb, down_points, interpolation=cv2.INTER_LINEAR)
        if not ret:
            break

        num_frames += 1

        elapsed_time = time.time() - start_time


        # Add frame to batch
        batch.append(frame)

        # Check if batch is full
        if len(batch) == arguments.batch_size:
            # Put batch into queue for processing
            batch_queue.appendleft(batch)

            logging.info(f"Capture:{time.time()-capture_start}")
            capture_start = time.time()


            # Clear batch
            batch = []



        if len(image_queue) and show_overlay:
            # Display frame
            ov_im = image_queue.pop()
            cv2.imshow('frame', frame)
            # overlay image
            cv2.imshow('overlay', ov_im)
        elif len((image_queue)):
            ov_im = image_queue.pop()

        # Check for exit key
        if cv2.waitKey(1) == ord('q'):
            #print(capture_latencies, processing_latencies, processing_latencies)
            break

        if elapsed_time > 1:
            fps = int(num_frames / elapsed_time)
            print(f"Webcam FPS:{fps}")
            num_frames = 0
            start_time = time.time()








    # Wait for all batches to be processed
    #batch_queue.join()

    # Release webcam and close window
    cap.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
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

    parser.add_argument('--batch_size', type=int, default=30, help="number of video frames to process FLAME")
    parser.add_argument('--webcam_fps', type=int, default=30, help="check your webcam fps and set it here")
    parser.add_argument('--show_overlay', type=bool, default=True, help="show cv2 image and morphed 3D face")

    arguments = parser.parse_args()

    # path_to_models = '/ps/scratch/rdanecek/emoca/finetune_deca'
    # path_to_models = '/is/cluster/work/rdanecek/emoca/finetune_deca'
    path_to_models = arguments.path_to_models
    model_name = arguments.model_name
    output_folder = arguments.output_folder + "/" + model_name

    mode = arguments.mode
    # mode = 'detail'
    # mode = 'coarse'

    # 1) Load the model
    with torch.no_grad():
        emoca_model, conf = load_model(path_to_models, model_name, mode)
        emoca_model.cuda()
        emoca_model.eval()
        main(arguments)


