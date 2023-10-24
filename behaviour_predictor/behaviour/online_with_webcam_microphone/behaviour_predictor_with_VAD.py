"""
Simple example of using zmq log handlers

This starts a number of subprocesses with PUBHandlers that generate
log messages at a regular interval.  The main process has a SUB socket,
which aggregates and logs all of the messages to the root logger.
"""
import argparse
import threading
import json
import time
from threading import Thread
import zmq
from pathlib import Path
import logging
from collections import deque
import sys


import traceback

stop_event = threading.Event()
stop_main_flag = threading.Event()

# incoming message queue
from LL.modules.fact_model import setup_model, calc_logit_loss
from LL.vqgan.vqmodules.gan_models import setup_vq_transformer
from LL.test_vq_decoder import run_model, generate_prediction, save_pred

from utils.zero_mq_utils import zmq_send_array, zeromq_frame_collector, DataFrame
import numpy as np
import torch
import os
from utils.filters import time_hns
import queue



# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)




def l2l_initiate_models(l2l_args, l_model_path, checkpoint, l_vqconfig):
    l_vq_model, _, _ = setup_vq_transformer(l2l_args, l_vqconfig,
                                            load_path=l_model_path,
                                            test=True)
    l_vq_model.eval()
    vq_configs = {'l_vqconfig': l_vqconfig, 's_vqconfig': None}

    ## setup Predictor model
    load_path = checkpoint
    print('> checkpoint', load_path)
    generator, _, _ = setup_model(l2l_config, l_vqconfig,
                                  mask_index=0, test=True, s_vqconfig=None,
                                  load_path=load_path)
    generator.eval()

    return l_vq_model, generator, l2l_config


def generate_l2l_output(l2largs, l2lconfig, lvqmodel, generator_model, testX, testY, testaudio, body_mean_Y,
                        body_std_Y, logger):
    rng = np.random.RandomState(23456)
    seq_len = 32
    patch_size = 8
    num_out = 1

    ## run model and save/eval
    with torch.no_grad():
        unstd_pred, probs, unstd_ub = run_model(l2largs, l2lconfig, lvqmodel, generator_model,
                                            testX, testY, testaudio, seq_len,
                                            patch_size, rng=rng)

    overall_l2 = np.mean(np.linalg.norm(testY[:, seq_len:, :] - unstd_pred[:, seq_len:, :], axis=-1))
    logger.debug(f'overall l2 with unstandardized:{overall_l2}' )


    return unstd_pred



# def generate_behaviour(mps, logger):
#     """
#     mps: message per second.
#
#     """
#     logger.info(f"Generator is waiting for both incoming data streams")
#     event_mfcc.wait()
#     #event_flame.wait()
#     logger.info(f"Generator is receiving mfcc and flame data ")
#     test_Y = None
#
#     publish_counter = 0
#     # MFCC publish chunk size for 0.3 seconds
#     smallest_mfcc_chunk = 28
#     mfcc_one_second_size = smallest_mfcc_chunk /0.3
#
#     while not stop_main_flag.is_set():
#         try:
#             if mfcc_counter_queue.qsize() >= mfcc_one_second_size:
#                 processing_start = time.time()
#
#                 logger.debug(f"len:-,flame_sp_q:{len(flame_sp_q)},mfcc_q:{len(mfcc_q)}")
#                 # copy data to a local list and release the thread lock
#                 with threading.Lock():
#                     audio_recent = list(mfcc_q)
#                     flame_recent = list(flame_sp_q)
#
#
#                 audio_map = map(DataFrame.get_processed_data, audio_recent)
#                 speaker_map = map(DataFrame.get_processed_data, flame_recent)
#
#
#                 test_X = np.float32(np.array([list(speaker_map)]))
#
#                 test_audio = np.float32(np.array([list(audio_map)]))
#
#                 logger.debug("Test_X type:",type(test_X), "test audio type:", type(test_audio))
#
#                 # initialise with mean vector repeating 64 times
#                 #initialise_vec = body_mean_Y
#                 initialise_vec =  np.zeros((body_mean_Y.shape))
#
#                 if test_Y is None:
#                     test_Y = np.repeat(initialise_vec, repeats=64, axis=1)  #
#                 else:
#                     test_Y = pred_y
#                 logger.debug(f"data:- test_X:{test_X.shape},test_Y:{test_Y.shape},test_audio:{test_audio.shape}")
#
#
#                 unstd_pred = generate_l2l_output(l2largs=l2l_args, l2lconfig=l2l_config,
#                                              lvqmodel=l_vq_model, generator_model=generator,
#                                              testX=test_X, testY=test_Y, testaudio=test_audio,
#                                              body_mean_Y=body_mean_Y, body_std_Y=body_std_Y, logger=logger)
#
#                 # standardize output
#                 pred_y = unstd_pred * body_std_Y + body_mean_Y
#
#
#                 logger.info(f"Process:{time.time() - processing_start}")
#
#
#                 publishing_start = time.time()
#                 zmq_send_array(socket=socket_pub, Arr=pred_y, flags=0, copy=True, track=False)
#                 logger.info(f"Publish:{time.time()-publishing_start}")
#
#                 if mfcc_counter_queue.qsize() >= mfcc_one_second_size:
#                     for _ in range(int(mfcc_one_second_size)-1):
#                         try:
#                             mfcc_counter_queue.get_nowait()
#                         except queue.Empty:
#                             break
#
#                 # wait till predicted behaviour is animated
#                 time.sleep(1 / mps)
#
#             else:
#                 time.sleep(1/mps)
#
#             #count += 1
#
#         except IndexError as ie:
#             traceback.print_exc()
#             logger.error(ie)
#         except KeyboardInterrupt:
#             stop_event.set()

def generate_behaviour(mps, logger):
    logger.info("Generator is waiting for both incoming data streams")
    event_mfcc.wait()
    logger.info("Generator is receiving mfcc and flame data")

    smallest_mfcc_chunk = 28
    mfcc_one_second_size = smallest_mfcc_chunk / 0.3
    # deduct processing time
    processing_time = 0.5
    remove_len = 256


    while not stop_main_flag.is_set():
        pred_y = None
        test_Y = None
        sleep_duration = (1 / mps) - processing_time


        try:
            enough_mfcc = mfcc_counter_queue.qsize() >= mfcc_one_second_size
            if enough_mfcc:
                processing_start = time.time()

                with threading.Lock():
                    audio_recent = list(mfcc_q)
                    flame_recent = list(flame_sp_q)

                # Use NumPy arrays directly
                audio_map = np.array([frame.get_processed_data() for frame in audio_recent], dtype=np.float32)
                speaker_map = np.array([frame.get_processed_data() for frame in flame_recent], dtype=np.float32)

                # Reuse variables
                test_X = np.array([speaker_map])
                test_audio = np.array([audio_map])

                initialise_vec = np.zeros((body_mean_Y.shape))
                test_Y = pred_y if test_Y is not None else np.repeat(initialise_vec, repeats=64, axis=1)

                unstd_pred = generate_l2l_output(
                    l2largs=l2l_args, l2lconfig=l2l_config, lvqmodel=l_vq_model,
                    generator_model=generator, testX=test_X, testY=test_Y,
                    testaudio=test_audio, body_mean_Y=body_mean_Y, body_std_Y=body_std_Y, logger=logger
                )

                pred_y = unstd_pred * body_std_Y + body_mean_Y
                logger.info(f"Predicted: {pred_y.shape}")
                logger.debug(f"Process:{time.time() - processing_start}")

                publishing_start = time.time()
                zmq_send_array(socket=socket_pub, Arr=pred_y, flags=0, copy=True, track=False)
                logger.debug(f"Publish:{time.time() - publishing_start}")

                # make it reactive by removing frames from the generation queue

                if mfcc_counter_queue.qsize() >= remove_len:
                    for _ in range(int(remove_len)):
                        try:
                            mfcc_counter_queue.get_nowait()
                        except queue.Empty:
                            break
                # wait for queue to fill
                processing_time = time.time() - processing_start
                if sleep_duration>0:
                    logger.info(f"Sleep for :{sleep_duration} to match message per second (mps)")
                    time.sleep(sleep_duration)
            else:
                # wait for queue to fill
                time.sleep(0.2)
        except Exception as e:
            logger.error(e)
        except KeyboardInterrupt:
            stop_event.set()



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)


    parser = argparse.ArgumentParser()
    # config_df = Path(BASE_DIR) / "external_lib/learning2listen/src/configs/vq/delta_v6.json"
    # checkpoint_df = Path(BASE_DIR) / "external_lib/learning2listen/src/models/delta_v6_er2er_best.pth"
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = Path(FILE_DIR).parents[1]
    # scenario 1 : conan tv representor
    #config_df = BASE_DIR/ "external_lib/learning2listen/LL/configs/vq/delta_v6.json"
    #checkpoint_df = BASE_DIR/ "external_lib/learning2listen/LL/models/delta_v6_er2er_best.pth"
    # base_dir = BASE_DIR/ "external_lib/learning2listen/LL/"
    # scenario 2: therapist seated left
    # config_df = BASE_DIR / "external_lib/learning2listen/LL/configs/vq/benecke_left_v6.json"
    # checkpoint_df = BASE_DIR / "external_lib/learning2listen/LL/models/beneckeleft_v6_er2er_best.pth"
    # base_dir = BASE_DIR / "external_lib/learning2listen/LL/"
    # scenario 3: therapist seated right
    config_df = BASE_DIR / "external_lib/learning2listen/LL/configs/vq/benecke_right_restart_v6.json"
    checkpoint_df = BASE_DIR / "external_lib/learning2listen/LL/models/benecke_right_restart_v6_er2er_best.pth"
    base_dir = BASE_DIR / "external_lib/learning2listen/LL/"

    parser.add_argument('--config', type=str, default=str(config_df), required=False)
    parser.add_argument('--base_dir', type=str, default=str(base_dir), required=False)
    parser.add_argument('--checkpoint', type=str, default=str(checkpoint_df), required=False)
    parser.add_argument('--speaker', type=str, default='conan', required=False)
    parser.add_argument('--etag', type=str, default='')
    parser.add_argument('--sample_idx', type=int, default=None)
    parser.add_argument('--save', action='store_true')
    l2l_args = parser.parse_args()
    print(l2l_args)

    # ________________________ l2l confgs _____________________________________#
    with open(config_df) as f:
        l2l_config = json.load(f)
    pipeline = l2l_config['pipeline']
    tag = l2l_config['tag']

    ## setup VQ-VAE model
    with open(os.path.join(base_dir, l2l_config['l_vqconfig'])) as f:
        l_vqconfig = json.load(f)
    logger.info(os.path.join(base_dir, l2l_config['l_vqconfig']))
    # l_vqconfig["tag"], l_vqconfig["pipeline"]
    l_model_path = base_dir / 'vqgan' / l_vqconfig['model_path'] / \
                   f'{l_vqconfig["tag"]}{l_vqconfig["pipeline"]}_best.pth'
    preprocess_model_info = np.load(base_dir / 'vqgan' / l_vqconfig['model_path'] / \
                                    f'{l_vqconfig["tag"]}{l_vqconfig["pipeline"]}_preprocess_core.npz')
    # print(preprocess_model_info)
    with torch.no_grad():
        l_vq_model, generator, l2l_config = l2l_initiate_models(l2l_args, l_model_path, checkpoint_df, l_vqconfig)

    body_mean_Y = preprocess_model_info['body_mean_Y']
    body_std_Y = preprocess_model_info['body_std_Y']
    logger.info(f"body_mean_Y:{body_mean_Y.shape} and body_std_Y: {body_std_Y.shape}")

    start_time = time.time()
    count = 0
    speaker_fp, listener_fp, audio_fp = [], [], []

    # ________________________ streaming socket confgs _____________________________________#
    with open('config.json') as f:
        local_config = json.load(f)
    # l_ip = local_config["listener_flame_streamer"]["ip"]
    # l_port = local_config["listener_flame_streamer"]["port"]
    # l_topic = local_config["listener_flame_streamer"]["topic"]

    # speaker
    s_ip = local_config["speaker_flame_streamer"]["ip"]
    s_port = local_config["speaker_flame_streamer"]["port"]
    s_topic = local_config["speaker_flame_streamer"]["topic"]

    # mfcc
    mfcc_ip = local_config["speaker_audio_mfcc_streamer"]["ip"]
    mfcc_port = local_config["speaker_audio_mfcc_streamer"]["port"]
    mfcc_topic = local_config["speaker_audio_mfcc_streamer"]["topic"]
    # out
    pub_out_ip = local_config["pub_output_to_fastapi"]["ip"]
    pub_out_port = local_config["pub_output_to_fastapi"]["port"]
    pub_out_topic = local_config["pub_output_to_fastapi"]["topic"]
    pub_out_bind = local_config["pub_output_to_fastapi"]["bind"]

    autoregressive_flag = local_config["autoregressive"]

    # Creates a socket instance for pubout in main thread
    context_pub = zmq.Context()
    socket_pub = context_pub.socket(zmq.PUB)
    socket_pub.bind(f"tcp://{pub_out_ip}:{pub_out_port}")

    ###### threads #######
    threads = []

    # flame_sp_q = deque(maxlen=100)
    # flame_ls_q = deque(maxlen=100)
    # mfcc_q = deque(maxlen=100)

    # initialise the dequeue with the size of feature frames need for l2l method
    flame_sp_q = deque(maxlen=64)
    mfcc_q = deque(maxlen=256)

    # counter q to work with VAD
    mfcc_counter_queue = queue.Queue()


    # fill with zeors 128 mfcc vectors
    timestamp = time_hns()
    timestamp_enc = str(timestamp).encode('ascii')


    for i in range(256):
        key = "mfcc".encode('ascii')
        json_data = {
            "mfcc": np.zeros((128)).astype(np.float32).tolist(),
            "timestamp_utc": timestamp
        }
        mencoded_data = json.dumps(json_data).encode('utf8')
        mfcc_q.append(DataFrame("mfcc".encode('ascii'), "0".encode('ascii'),mencoded_data ))
    # fill with 56 flame vectors
    for i in range(64):
        key = "flame".encode('ascii')
        fjson_data = {
            "flame": np.zeros((56)).astype(np.float32).tolist(),
            "timestamp_utc": timestamp
        }
        fencoded_data = json.dumps(fjson_data).encode('utf8')
        flame_sp_q.append(DataFrame("flame".encode('ascii'), "0000".encode('ascii'),fencoded_data ))



    # threads workers
    event_mfcc = threading.Event()
    event_flame = threading.Event()
    speaker_th = Thread(target=zeromq_frame_collector, args=(s_ip, s_port, s_topic, flame_sp_q,
                                                             event_flame, stop_event,logger ),
                        kwargs=dict(sequence_length=64,mask_leng=1, bind=False, log=False, name="flame"), daemon=True)

    mfcc_th = Thread(target=zeromq_frame_collector, args=(mfcc_ip, mfcc_port, mfcc_topic, mfcc_q,
                                                          event_mfcc,stop_event,logger,),
                     kwargs=dict(sequence_length=(64 * 4),mask_leng=1, bind=False, log=False, name="mfcc",
                                 mfcc_counter_queue=mfcc_counter_queue),
                     daemon=True)

    message_per_second = 1.5 # number of anim sequ process and send to fastapi
    generator_th = Thread(target=generate_behaviour, args=(message_per_second,logger),
                     kwargs=dict(),
                     daemon=True)
    speaker_th.start()
    mfcc_th.start()
    generator_th.start()
    threads.append(mfcc_th)
    threads.append(speaker_th)
    threads.append(generator_th)

    try:
        # Wait for all threads to complete
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        # Stop all threads
        logger.info('Keyboard interrupt: stopping all threads')
        stop_event.set()

    logger.info("Exiting main thread")
