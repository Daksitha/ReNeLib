import json

import pyaudio
import numpy as np
from python_speech_features import mfcc
import zmq
import threading
import time
import queue
import librosa
import PIL.Image as Image
import sounddevice as sd
import argparse

import sys
from collections import deque

import logging
stop_event = threading.Event()
logging.basicConfig(filename='mfcc_extractor.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
    publishing_start = time.time()
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

    logging.info(f"Publish:{time.time()- publishing_start}")

    return 1
def process_audio_data(audio_queue, publisher, arguments):
    publish_counter = 0
    while not stop_event.set():
        try:

            if len(audio_queue):
                audio_data = audio_queue.pop()
                if audio_data is None:
                    break

                processing_start = time.time()
                # Extract MFCC features
                sample_rate = arguments.sr
                win_len = int(0.025 * sample_rate)
                hop_len = int(0.010 * sample_rate)
                fft_len = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0)))
                S_dB = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128, hop_length=hop_len)

                # do some resizing to match frame rate
                im = Image.fromarray(S_dB)
                _, feature_dim = im.size
                scale_four = arguments.video_duration * 4
                im = im.resize((scale_four, feature_dim), Image.ANTIALIAS)
                # take transpose to save in the format of (4T, 128)
                mfcc_arry = np.array(im).transpose()
                print(mfcc_arry.shape)

                publish_data_frame(arguments.topic, publisher, mfcc_arry)

                logging.info(f"Process:{time.time() - processing_start}")
        except KeyboardInterrupt:
            logging.info('Audio processing interrupted')
            stop_event.set()

        #audio_queue.task_done()
def record_audio_time(audio_queue, duration, fs):
    capture_start = time.time()
    while not stop_event.set():
        try:
            recording = sd.rec(frames=int(duration*fs), samplerate=fs, channels=1, dtype='float64')
            sd.wait()
            audio_queue.appendleft(recording[:, 0])

            logging.info(f"Capture:{time.time()- capture_start}")
            capture_start = time.time()
        except KeyboardInterrupt:
            logging.info('Audio recorder interrupted')
            stop_event.set()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add the input folder arg

    parser.add_argument('--ip', type=str, default='127.0.0.1', help="ip address to stream extracted FLAME features ")
    parser.add_argument('--port', type=str, default='5555', help="mfcc features")
    parser.add_argument('--topic', type=str, default='mfcc', help="")
    parser.add_argument('--audio_duration', type=float, default=0.5, help="time to record the audio in seconds")
    parser.add_argument('--sr', type=float, default=16000, help="audio sampling rate")
    parser.add_argument('--video_duration', type=int, default=15, help="Number of video frames process during the "
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

    # Set up the audio queue
    audio_queue = deque(maxlen=2)

    threads = []

    # Start the worker thread for processing audio data
    worker_thread = threading.Thread(target=process_audio_data, args=(audio_queue, publisher,
                                                                      arguments))


    producer_thread = threading.Thread(target=record_audio_time, args=(audio_queue, arguments.audio_duration, arguments.sr))

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
        logging.info('Keyboard interrupt: stopping all threads')
        stop_event.set()

