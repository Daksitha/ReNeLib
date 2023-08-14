import zmq
import numpy as np
import json
import threading
import os
import sys
import time
class DataFrame():
    def __init__(self, key, timestamp, data):
        if isinstance(data, bytes):
            data = data.decode('utf-8')
            self.data = json.loads(data)
        else:
            raise RuntimeError("Invalid data: incoming not in bytes")
        #self.recv_timestamp = int(timestamp.decode('ascii'))
        #self.key = key.decode('ascii')
        self.snt_timestamp = self.data['timestamp_utc']
        if 'flame' in self.data:
            flame_data = self.data['flame']
            nump_flame = np.array(flame_data).astype(np.float32)
            #print(nump_flame.dtype)
            self.process_data = nump_flame

        elif 'mfcc' in self.data:
            mfcc_data = self.data['mfcc']
            np_mfcc = np.array(mfcc_data).astype(np.float32)
            #print(np_mfcc.dtype)
            self.process_data = np_mfcc


        else:
            raise RuntimeError("Invalid data, neither mfcc nor flame")

    def get_timestamp(self):
        return self.recv_timestamp

    def get_key_value(self):
        return self.key

    def get_processed_data(self):
        return self.process_data




def zeromq_frame_collector(ip, port, topic, dequeue, event,stop_event,logging,
                           sequence_length=64, mask_leng=1, bind=False, log=True, name=""):
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

    # fps calculator
    start_time = time.time()
    x = 1  # displays the frame rate every 1 second
    counter = 0

    while not stop_event.is_set():
        start_capture = time.time()
        key, timestamp, data = sub_zmq.recv_multipart()
        if data:
            q_frame = DataFrame(key, timestamp, data)
            dequeue.append(q_frame)
            logging.info(f"C_{name}: {time.time()-start_capture}")
            event.set()

            counter += 1
            if (time.time() - start_time) > x:
                logging.info(f"FPS at {name} thread : { counter / (time.time() - start_time)} ")
                counter = 0
                start_time = time.time()


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



def zeromq_numpy_array_collector(ip, port, topic, dequeue,stop_event,logging,
                                 sequence_length=64, mask_leng=1, bind=False, log=True, name="", ):
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

    while not stop_event.is_set():
        """recv a numpy array"""
        md = sub_zmq.recv_json(flags=0)
        data_arr = sub_zmq.recv(flags=0, copy=True, track=False)
        arr = np.frombuffer(data_arr, dtype=md['dtype']).reshape(md['shape'])
        print(name,":",arr.shape)
        #dequeue.append(arr)


        # key, timestamp, data = sub_zmq.recv_multipart()
        # if data:
        #     # print(data)
        #     q_frame = DataFrame(key, timestamp, data)
        #     data_bin.append(q_frame)
        #
        #     stop_ln = sequence_length / mask_leng
        #     if len(data_bin) == stop_ln:
        #         send_data = np.tile(data_bin,mask_leng)
        #         dequeue.append(send_data)
        #         logging.info(f"append {name} data of shape:{send_data.shape} with masking:{mask_leng} ")
        #         data_bin = []



    logging.debug("Closing the zeromq_frame_collector socket")
    sub_zmq.close()


