from threading import Thread, Event
import zmq

from collections import deque
import json
import threading
import time

import sys
import logging
stop_event = threading.Event()

import numpy as np
# get root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
stop_event = Event()
def angle_axis_to_quaternion(angle_axis: np.ndarray) -> np.ndarray:
    """Convert an angle axis to a quaternion.

    DECA project adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (np.ndarray): numpy array with angle axis.

    Return:
        np.ndarray: numpy array with quaternion.

    Shape:
        - Input: `(N, 3)` where `N` is the number of angle axis vectors.
        - Output: `(N, 4)`

    Example:
    """
    if not isinstance(angle_axis, np.ndarray):
        raise TypeError("Input type is not a numpy.ndarray. Got {}".format(
            type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError("Input must be an array of shape Nx3 or 3. Got {}"
                         .format(angle_axis.shape))
    # unpack input and compute conversion
    a0 = angle_axis[..., 0:1]
    a1 = angle_axis[..., 1:2]
    a2 = angle_axis[..., 2:3]
    theta_squared = a0 * a0 + a1 * a1 + a2 * a2

    theta = np.sqrt(theta_squared)
    half_theta = theta * 0.5

    mask = theta_squared > 0.0
    ones = np.ones_like(half_theta)

    k_neg = 0.5 * ones
    k_pos = np.sin(half_theta) / theta
    k = np.where(mask, k_pos, k_neg)
    w = np.where(mask, np.cos(half_theta), ones)

    quaternion = np.zeros_like(angle_axis)
    quaternion[..., 0:1] += a0 * k
    quaternion[..., 1:2] += a1 * k
    quaternion[..., 2:3] += a2 * k
    return np.concatenate([w, quaternion], axis=-1)

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

# def send_flame_to_blender(message, topic,pub_zmq, fps):
#     # blender is waiting for  topic, timestamp, msg
#     timestamp = time_hns()
#     timestamp_enc = str(timestamp).encode('ascii')
#
#     if not isinstance(message, bytes):
#         if not isinstance(message, str):
#             message = json.dumps(message)
#         # change from string to bytes
#         message = message.encode('utf-8')
#         #print(message)
#
#     pub_zmq.send_multipart([str(topic).encode('ascii'),timestamp_enc, message])
#     time_sleep = 1/fps
#     time.sleep(time_sleep)

def angle_axis_to_quat_bactch(flame_pose):
    # convert axis angles to euler
    # first 3 axis angles are global and the last three are jaw pose
    #print("flame_pose shape:", flame_pose.shape)

    glob_qat = angle_axis_to_quaternion(flame_pose[:, :3])  # Nx4
    jaw_qat = angle_axis_to_quaternion(flame_pose[:, 3:])  # Nx4
    #print(glob_qat.shape, jaw_qat.shape)


    return glob_qat, jaw_qat
def arrange_blender_batch(flame_batch, ignore_fr_from_start=30):
    """
    expecting a 2D array containing flame parameters
    :param flame_vector: 1x56 vector where 1-50 are flame shapekey values and 50-56 are pose information
    :pose angles are in axis angles, it needs to be converted to Euler anles
    :return: json_dic containing Apple ARKit converted shapekey (beaware this is for Charamel character
    so naming might slightly differ)
    """

    json_exp = {}
    json_pose = {}
    json_dic = {}
    flame_arr = np.array(flame_batch, dtype=np.float32)



    # take first flame parameters. we have only defined for charamel vectors for 20
    flame_exp = flame_arr[ignore_fr_from_start:,:50]
    flame_pose = flame_arr[ignore_fr_from_start:,50:56]
    glob_quat, jaw_quat = angle_axis_to_quat_bactch(flame_pose)

    #print("shapes:",flame_exp.shape,glob_quat.shape , jaw_quat.shape)
    glob_jaw = np.concatenate((glob_quat, jaw_quat), axis=1)
    transformed_batch = np.concatenate((flame_exp, glob_jaw), axis=1)

    #print(transformed_batch.shape)

    return transformed_batch


def zeromq_frame_publisher(ip, port, topic, queue: deque, pub_zmq,fps, level=logging.DEBUG, bind=False, log=True, name=""):
    if bind:
        pub_zmq.bind(f'tcp://{ip}:{port}')
    else:
        pub_zmq.connect(f'tcp://{ip}:{port}')

    en_topic = topic.encode('ascii')

    while not stop_event.is_set():
        if len(queue):
            try:
                message = queue.popleft()
                batch = arrange_blender_batch(message[0], ignore_fr_from_start=30)

                for frame in batch:

                    data = {"flame":frame.tolist()}

                    if data:
                        # if data not yet bytes
                        if not isinstance(data, bytes):
                            # if data != JSON:
                            if not isinstance(data, str):
                                # change to json string
                                data = json.dumps(data)
                            # change from string to bytes
                            data = data.encode('utf-8')

                        timestamp = time_hns()
                        timestamp_enc = str(timestamp).encode('ascii')

                        pub_zmq.send_multipart([en_topic, timestamp_enc, data])

                    #time.sleep(1/fps)
                    logger.debug(f"Published message: {len(data)} to {ip}:{port}, topic: {topic}")



                # for k, data in message.items():
                #     if k != "frame":
                #         if not isinstance(data, bytes):
                #             data = json.dumps(data).encode('utf-8')
                #         pub_zmq.send_multipart([en_topic, data])
                #         time.sleep(0.01)
                #         logger.debug(f"Published message {k}: {data} to {ip}:{port}, topic: {topic}")

            except:
                logger.exception("Error in zeromq_frame_publisher thread")


def zeromq_frame_collector(ip, port, topic, queue, sub_zmq, level=logging.DEBUG, bind=False, log=True, name=""):
    if bind:
        sub_zmq.bind(f'tcp://{ip}:{port}')
    else:
        sub_zmq.connect(f'tcp://{ip}:{port}')

    sub_zmq.setsockopt_string(zmq.SUBSCRIBE, topic)

    while not stop_event.is_set():
        md = sub_zmq.recv_json(flags=0)
        data_arr = sub_zmq.recv(flags=0, copy=True, track=False)
        arr = np.frombuffer(data_arr, dtype=md['dtype']).reshape(md['shape'])
        queue.append(arr)
        logger.debug(f"Appending {arr.shape} anim seq to queue")

    sub_zmq.close()


if __name__ == "__main__":
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    pub = ctx.socket(zmq.PUB)
    dq_flames = deque()

    ml_module = Thread(target=zeromq_frame_collector, args=("127.0.0.1", "6680", "", dq_flames, sub),
                       kwargs=dict(bind=False, log=True, name="ml_in"),
                       daemon=True)
    ml_module.start()

    sender = Thread(target=zeromq_frame_publisher, args=("127.0.0.1", "5580", "blender", dq_flames, pub, 30),
                    kwargs=dict(bind=True, log=True, name="ml_output"),
                    daemon=True)
    sender.start()

    threads = [ml_module, sender]

    for t in threads:
        t.join()