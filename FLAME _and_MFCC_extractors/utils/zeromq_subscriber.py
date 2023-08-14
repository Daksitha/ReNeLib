import zmq
import numpy as np

def subscriber():
    # Set up ZeroMQ subscriber
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://127.0.0.1:6689")

    # Subscribe to the "mfcc" topic
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "mfcc")

    while True:
        # Receive the topic and MFCC features
        key, timestamp_enc, encoded_data = subscriber.recv_multipart()
        print(encoded_data)

        # Convert the received bytes to a NumPy array
        #mfcc_features = np.frombuffer(mfcc_features_bytes, dtype=np.float32)

        # Print the received MFCC features
        #print("Received MFCC features:", mfcc_features.shape)

if __name__ == "__main__":
    subscriber()
