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
from zmq.log.handlers import PUBHandler
from config.config_manager import BASE_DIR
import traceback

stop_event = threading.Event()
stop_main_flag = threading.Event()
# incoming message queue

from external_lib.learning2listen.LL.modules.fact_model import setup_model, calc_logit_loss
from external_lib.learning2listen.LL.vqgan.vqmodules.gan_models import setup_vq_transformer
from external_lib.learning2listen.LL.utils.load_utils import *
from external_lib.learning2listen.LL.test_vq_decoder_changed import run_model, generate_prediction, save_pred
import numpy as np
import cv2

import torch

####### L2l Machine learning modules ####
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


def bilateral_filter(outputs):
    """ smoothing function

    function that applies bilateral filtering along temporal dim of sequence.
    TODO: Fix this for live streaming
    """
    outputs_smooth = np.zeros(outputs.shape)
    for b in range(outputs.shape[0]):
        for f in range(outputs.shape[2]):
            smoothed = np.reshape(cv2.bilateralFilter(
                outputs[b, :, f], 5, 20, 20), (-1))
            outputs_smooth[b, :, f] = smoothed
    return outputs_smooth.astype(np.float32)


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
                        body_std_Y):
    rng = np.random.RandomState(23456)
    seq_len = 32
    patch_size = 8
    num_out = 1024

    ## run model and save/eval
    unstd_pred, probs, unstd_ub = run_model(l2largs, l2lconfig, lvqmodel, generator_model,
                                            testX, testY, testaudio, seq_len,
                                            patch_size, rng=rng)

    overall_l2 = np.mean(np.linalg.norm(testY[:, seq_len:, :] - unstd_pred[:, seq_len:, :], axis=-1))
    print('overall l2 with unstandardized:', overall_l2)

    # standardize output
    # B, T, _ = unstd_pred.shape

    test_Y = unstd_pred * body_std_Y + body_mean_Y

    return test_Y


######### Communication modules ###############

class DataFrame():
    def __init__(self, key, timestamp, data):
        if isinstance(data, bytes):
            data = data.decode('utf-8')
            self.data = json.loads(data)
        else:
            raise RuntimeError("Invalid data: incoming not in bytes")
        self.recv_timestamp = int(timestamp.decode('ascii'))
        self.key = key.decode('ascii')
        self.snt_timestamp = self.data['timestamp_utc']
        if 'flame' in self.data:
            self.process_data = self.data['flame']
        elif 'mfcc' in self.data:
            self.process_data = self.data['mfcc']
        else:
            raise RuntimeError("Invalid data, neither mfcc nor flame")

    def get_timestamp(self):
        return self.recv_timestamp

    def get_key_value(self):
        return self.key

    def get_processed_data(self):
        return self.process_data


def zeromq_frame_collector(ip, port, topic, dequeue, sequence_length=64, mask_leng=1, bind=False, log=True, name=""):
    """

    Args:
        ip:
        port:
        topic:
        dequeue:
        sequence_length:
        mask_leng: even intiger between 2-64
                    how to duplicate the collected data sequence.
                    This is in action when you want to speed up the throughput by duplicating sequence_length
        bind:
        log:
        name:

    Returns:

    """
    ctx = zmq.Context()
    sub_zmq = ctx.socket(zmq.SUB)
    if bind:
        sub_zmq.bind(f'tcp://{ip}:{port}')
    else:
        sub_zmq.connect(f'tcp://{ip}:{port}')
    # socket.setsockopt(zmq.SUBSCRIBE, topicfilter)
    sub_zmq.setsockopt_string(zmq.SUBSCRIBE, topic)

    # print('test Hello from process %s and thread %s' % (os.getpid(), threading.current_thread()))
    logging.debug(
        f"starting {name} at with {ip}, port {port}, topic {topic}, [thread {threading.current_thread()}, pid {os.getpid()} ")

    data_bin = []
    # avoid zero division
    if not 1 <= mask_leng < sequence_length:
        raise RuntimeError("Invalid masking length for data_bin ")

    while True:
        key, timestamp, data = sub_zmq.recv_multipart()
        if data:
            # print(data)
            q_frame = DataFrame(key, timestamp, data)
            data_bin.append(q_frame)

            stop_ln = sequence_length / mask_leng
            if len(data_bin) == stop_ln:
                send_data = np.tile(data_bin,mask_leng)
                dequeue.append(send_data)
                logging.info(f"append {name} data of shape:{send_data.shape} with masking:{mask_leng} ")
                data_bin = []



    logging.debug("Closing the zeromq_frame_collector socket")
    sub_zmq.close()


def zmq_send_array(socket, Arr: np.array, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(Arr.dtype),
        shape=Arr.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(Arr, flags, copy=copy, track=track)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)

    logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')

    parser = argparse.ArgumentParser()
    # config_df = Path(BASE_DIR) / "external_lib/learning2listen/src/configs/vq/delta_v6.json"
    # checkpoint_df = Path(BASE_DIR) / "external_lib/learning2listen/src/models/delta_v6_er2er_best.pth"
    config_df = Path(BASE_DIR) / "external_lib/learning2listen/src/configs/vq/benecke_right_restart_v6.json"
    checkpoint_df = Path(BASE_DIR) / "external_lib/learning2listen/src/models/benecke_right_restart_v6_er2er_best.pth"
    base_dir = Path(BASE_DIR) / "external_lib/learning2listen/src/"

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
    print(os.path.join(base_dir, l2l_config['l_vqconfig']))
    # l_vqconfig["tag"], l_vqconfig["pipeline"]
    l_model_path = base_dir / 'vqgan' / l_vqconfig['model_path'] / \
                   f'{l_vqconfig["tag"]}{l_vqconfig["pipeline"]}_best.pth'
    preprocess_model_info = np.load(base_dir / 'vqgan' / l_vqconfig['model_path'] / \
                                    f'{l_vqconfig["tag"]}{l_vqconfig["pipeline"]}_preprocess_core.npz')
    # print(preprocess_model_info)

    l_vq_model, generator, l2l_config = l2l_initiate_models(l2l_args, l_model_path, checkpoint_df, l_vqconfig)

    body_mean_Y = preprocess_model_info['body_mean_Y']
    body_std_Y = preprocess_model_info['body_std_Y']
    print(f"body_mean_Y:{body_mean_Y.shape} and body_std_Y: {body_std_Y.shape}")

    start_time = time.time()
    count = 0
    speaker_fp, listener_fp, audio_fp = [], [], []

    # ________________________ streaming socket confgs _____________________________________#
    with open('config.json') as f:
        local_config = json.load(f)
    l_ip = local_config["listener_flame_streamer"]["ip"]
    l_port = local_config["listener_flame_streamer"]["port"]
    l_topic = local_config["listener_flame_streamer"]["topic"]

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

    ##################################### threads ##################
    threads = []
    # TODO: should i limit the length of the queue? depends on the cpu and memory usage
    # flame_sp_q = deque(maxlen=100)
    # flame_ls_q = deque(maxlen=100)
    # mfcc_q = deque(maxlen=100)

    # autoregressive will be false when you want to test l2 loss with listener flame
    flame_sp_q = deque()
    flame_ls_q = deque()
    mfcc_q = deque()

    speaker_th = Thread(target=zeromq_frame_collector, args=(s_ip, s_port, s_topic, flame_sp_q),
                        kwargs=dict(sequence_length=64,mask_leng=1, bind=False, log=False, name="speaker"), daemon=True)

    mfcc_th = Thread(target=zeromq_frame_collector, args=(mfcc_ip, mfcc_port, mfcc_topic, mfcc_q),
                     kwargs=dict(sequence_length=(64 * 4),mask_leng=1, bind=False, log=False, name="audio_mfcc"),
                     daemon=True)
    speaker_th.start()
    mfcc_th.start()
    threads.append(mfcc_th)
    threads.append(speaker_th)

    if not autoregressive_flag:
        listener_th = Thread(target=zeromq_frame_collector, args=(l_ip, l_port, l_topic, flame_ls_q),
                             kwargs=dict(sequence_length=64,mask_leng=1, bind=False, log=False, name="listener"), daemon=True)
        listener_th.start()
        threads.append(listener_th)

    # threads.append(main_th)
    ################################## Machine learning section #####################
    test_Y = None

    while not stop_main_flag.is_set():
        try:

            if len(mfcc_q) and len(flame_sp_q) and (len(flame_ls_q) or autoregressive_flag):
                print(f"len:- flame_ls_q:{len(flame_ls_q)},flame_sp_q:{len(flame_sp_q)},mfcc_q:{len(mfcc_q)}")

                speaker_map = map(DataFrame.get_processed_data, flame_sp_q.popleft())
                audio_map = map(DataFrame.get_processed_data, mfcc_q.popleft())

                test_X = np.array([list(speaker_map)], dtype=np.float32)
                test_audio = np.array([list(audio_map)], dtype=np.float32)
                if not autoregressive_flag:
                    listener_map = map(DataFrame.get_processed_data, flame_ls_q.popleft())
                    test_Y = np.array([list(listener_map)], dtype=np.float32)
                    print(f"data:- test_X:{test_X.shape},test_Y:{test_Y.shape},test_audio:{test_audio.shape}")
                else:
                    # initialise with mean vector repeating 64 times
                    if test_Y is None:
                        test_Y = np.repeat(body_mean_Y, repeats=64, axis=1)
                    print(f"data:- test_X:{test_X.shape},test_Y:{test_Y.shape},test_audio:{test_audio.shape}")

                pred_y = generate_l2l_output(l2largs=l2l_args, l2lconfig=l2l_config,
                                             lvqmodel=l_vq_model, generator_model=generator,
                                             testX=test_X, testY=test_Y, testaudio=test_audio,
                                             body_mean_Y=body_mean_Y, body_std_Y=body_std_Y)

                if autoregressive_flag:
                    test_Y = pred_y

                # sending
                timestamp_send = time_hns()
                b_timestamp_send = str(timestamp_send).encode('ascii')
                zmq_send_array(socket=socket_pub, Arr=pred_y, flags=0, copy=True, track=False)

            count += 1

        except IndexError as ie:
            traceback.print_exc()
            print(ie)
            # print(f"Index error after {(time.time() - start_time)}seconds ")
            start_time = time.time()
            count = 0
            # every min it fills 30 frames: first one of course 0
            # time.sleep(1)
        except KeyboardInterrupt:
            stop_event.set()

        # time.sleep(2)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    print("Exiting main thread")
