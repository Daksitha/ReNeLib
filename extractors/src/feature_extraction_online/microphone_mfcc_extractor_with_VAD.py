import json
import zmq
import threading
import argparse
import sys
import numpy as np
import time
import librosa
from collections import deque
import sounddevice as sd
from PIL import Image
import torch

import logging
stop_event = threading.Event()
# logging.basicConfig(filename='mfcc_extractor.log', level=logging.DEBUG,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda')  # Use 'cuda' if GPU is available
torch.set_num_threads(1)






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

def publish_data_frame(topic, publisher, data_array):
    #publishing_start = time.time()
    timestamp = time_hns()
    timestamp_enc = str(timestamp).encode('ascii')
    key = topic.encode('ascii')
    # fill the queue with padding values zeros
    for mfcc_vec in data_array:
        json_data = {
            "mfcc": mfcc_vec.tolist(),
            "timestamp_utc": timestamp
        }
        encoded_data = json.dumps(json_data).encode('utf8')
        publisher.send_multipart([key, timestamp_enc, encoded_data])

    logger.info(f"Published:{data_array.shape}")

    return 1


def process_audio_data(audio_queue, publisher, arguments, logger, VADIterator, silero_model):
    in_speech = False  # State to keep track of whether we are in a speech segment or not

    vad_iterator = VADIterator(silero_model)
    with torch.no_grad():
        silero_model.eval().to(device)

    while not stop_event.set():
        try:
            if len(audio_queue):
                audio_data = audio_queue.pop()
                if audio_data is None:
                    continue

                # Convert the audio data to PyTorch tensor
                waveform = torch.tensor(audio_data).float().to(device)

                # Check if the current chunk contains voice activity using Silero VAD

                speech_dict = vad_iterator(waveform, return_seconds=True)

                if speech_dict:
                    # Update the in_speech state based on the speech_dict
                    if 'start' in speech_dict:
                        logger.info(f"{speech_dict}")
                        in_speech = True
                    elif 'end' in speech_dict:
                        logger.info(f"{speech_dict}")
                        in_speech = False
                # else:
                #     in_speech = False
                #     #vad_iterator.reset_states()
                #     logger.debug("No Speech detected")

                # If we are in a speech segment, process and send the data
                if in_speech:
                    processing_start = time.time()
                    # Extract MFCC features
                    sample_rate = arguments.sr
                    win_len = int(0.025 * sample_rate)
                    hop_len = int(0.010 * sample_rate)
                    fft_len = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0)))
                    S_dB = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128, hop_length=hop_len)

                    # Do some resizing to match frame rate
                    im = Image.fromarray(S_dB)
                    _, feature_dim = im.size
                    scale_four = arguments.video_duration *arguments.audio_duration * 4
                    im = im.resize((int(scale_four), feature_dim), Image.ANTIALIAS)
                    # Take transpose to save in the format of (4T, 128)
                    mfcc_arry = np.array(im).transpose()

                    logger.info(f"Audio feature size to publish {mfcc_arry.shape}")

                    publish_data_frame(arguments.topic, publisher, mfcc_arry)



                    # logger.debug(f"Process:{time.time() - processing_start}")
        except KeyboardInterrupt:
            logging.info('Audio processing interrupted')
            stop_event.set()
def record_audio_time(audio_queue, duration, fs, logger):
    capture_start = time.time()
    while not stop_event.set():
        try:
            recording = sd.rec(frames=int(duration*fs), samplerate=fs, channels=1, dtype='float64')
            sd.wait()
            audio_queue.appendleft(recording[:, 0])

            logger.debug(f"Capture:{time.time()- capture_start}")
            capture_start = time.time()

        except Exception as e:
            logger.error(f"Error occurred: {e}")
        except KeyboardInterrupt:
            logger.erro('Audio recorder interrupted')
            stop_event.set()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add the input folder arg

    parser.add_argument('--ip', type=str, default='127.0.0.1', help="ip address to stream extracted FLAME features ")
    parser.add_argument('--port', type=str, default='5555', help="mfcc features")
    parser.add_argument('--topic', type=str, default='mfcc', help="")
    parser.add_argument('--audio_duration', type=float, default=0.3, help="time to record the audio in seconds")
    parser.add_argument('--sr', type=float, default=16000, help="audio sampling rate")
    parser.add_argument('--video_duration', type=int, default=24, help="Number of video frames process during the "
                                                                       "audio_duration. Ex: 1 second audio means 30 "
                                                                       "video frames from a 30 fps webcam")
    #parser.add_argument('--l2l', type=int, default=256, help="number of features expected by l2l model for one forward "
                                                             # "pass")
    """
     1. audio duration records a audio input for the given interval and process them
     2. video_duration is used to resample the extracted mfcc feature to align with video frame
     given the flame extractor 32 fps. Then set this interval 32 and set audio duration as 1 second. 
     If you wish to process data at a faster rate, then you can choose 16 frames at 0.5 seconds. 
      
    """

    arguments = parser.parse_args()
    # Set up ZeroMQ publisher
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://{arguments.ip}:{arguments.port}")

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True,
                                )

    (_, _, _,VADIterator,collect_chunks) = utils

    # Check if the model is loaded successfully
    if model is None:
        raise RuntimeError("Failed to load Silero VAD model.")



    # Set up the audio queue
    audio_queue = deque(maxlen=2)

    threads = []

    # Start the worker thread for processing audio data
    worker_thread = threading.Thread(target=process_audio_data, args=(audio_queue, publisher,
                                                                      arguments), kwargs=dict(logger=logger, VADIterator=VADIterator, silero_model=model))


    producer_thread = threading.Thread(target=record_audio_time, args=(audio_queue, arguments.audio_duration, arguments.sr),
                                       kwargs=dict(logger=logger)
                                       )

    worker_thread.start()
    producer_thread.start()
    threads.append(worker_thread)
    threads.append(producer_thread)

    try:
        # Wait for all threads to complete
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        # Stop all threads
        logging.error('Keyboard interrupt: stopping all threads')
        stop_event.set()

