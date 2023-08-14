from pathlib import Path
import time
import zmq
import threading
import os
import logging
from collections import deque
from threading import Thread
# fast api
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import uvicorn
from typing import List
import numpy as np
from fastApi_backend.flame_to_charamel import flame_to_arkit_vector, flame_to_arkit_vector_local

################# Global Variables #####
threads = []
stop_threads = False
stop_event = threading.Event()
test_with_local_flame = False
dummy_data_flag  = False
# setup loggers

logging.basicConfig(filename='fastapi.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# get root logger

# incoming message queue
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
dq = deque(maxlen=100)
# to test I use the 5567 port from listener port
ip_, port_, topic_ = "127.0.0.1", "6680", ""


app = FastAPI(title='virtual-agent')
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_flame_message(self, message: str, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

import asyncio

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)


    async def send_animations():
        publish_counter = 0
        while True:
            if len(dq):

                processing_end = time.time()
                data_container = dq.popleft()
                charamel_json = flame_to_arkit_vector(data_container[0], ignore_fr_from_start=30, logger=logging)

                processing_end = time.time()

                print(f"ARKit message: {len(charamel_json)}")

                # time captured during async might not be accurate
                publishing_start = time.time()
                await manager.send_flame_message(charamel_json, websocket)
                logging.info(f"Publish:{time.time()-processing_end}")


            elif dummy_data_flag:
                with open("data/incoming_data/incoming_data.json") as jf:
                    charamel_json = json.load(jf)
                await manager.send_flame_message(charamel_json, websocket)
            elif test_with_local_flame:
                file_name = "flame_NICHT.npy"
                test_data = np.load(f"data/incoming_data/{file_name}")
                print(f"loading {test_data.shape}")
                charamel_json = flame_to_arkit_vector_local(test_data)
                print(f"charamel_json len {len(charamel_json)}")
                await manager.send_flame_message(charamel_json, websocket)
            await asyncio.sleep(1)  # Send a new animation every second

    send_animations_task = asyncio.create_task(send_animations())

    try:
        while True:
            data = await websocket.receive_text()
            print(data)
    except WebSocketDisconnect:
        send_animations_task.cancel()
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left")



def start_server(host_="127.0.0.1",port_=8000, relod_fl=True):
    uvicorn.run("fast_api_server:app", host=host_, port=port_, reload=relod_fl)
############### Fast API ####################

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
def zeromq_frame_collector(ip, port, topic, queue,sub_zmq,
                           level=logging.DEBUG, bind=False, log=True, name=""):
    latencies = []
    if bind:
        sub_zmq.bind(f'tcp://{ip}:{port}')
    else:
        sub_zmq.connect(f'tcp://{ip}:{port}')
    # socket.setsockopt(zmq.SUBSCRIBE, topicfilter)
    sub_zmq.setsockopt_string(zmq.SUBSCRIBE, topic)



    logging.debug(f"starting {name} at with {ip}, port {port}, topic {topic}, [thread {threading.current_thread()}, pid {os.getpid()} ")

    while not stop_event.is_set():

        """recv a numpy array"""
        md = sub_zmq.recv_json(flags=0)
        processing_start = time.time()
        data_arr = sub_zmq.recv(flags=0, copy=True, track=False)
        arr = np.frombuffer(data_arr, dtype=md['dtype']).reshape(md['shape'])
        queue.append(arr)
        logging.info(f"Appending {arr.shape} anim seq")

        logging.info(f"Capture:{time.time()-processing_start}")


    logging.info("Closing the zeromq_frame_collector socket")
    sub_zmq.close()



@app.on_event("startup")
async def startup_event():

    ml_module = Thread(target=zeromq_frame_collector, args=(ip_, port_, topic_, dq, sub),
                       kwargs=dict(bind=False, log=True, name="ml_output"),
                       daemon=True)
    ml_module.start()
    threads.append(ml_module)
@app.on_event("shutdown")
async def shutdown_event():
    # clean up the connection started
    stop_event.set()
    print("shutting down")


if __name__ == "__main__":
    start_server()
    # clean the connection
    sub.close()
    ctx.term()
    for t in threads:
        t.join()

    print("Exiting main thread")

