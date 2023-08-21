"""Prepossesses openface .npy files and publishes its data to FLAMEvatar

Relies on Python 3.6+ due to async generator yield statement"""

# Copyright (c) Stef van der Struijk
# License: GNU Lesser General Public License


import os
# from os.path import join, isfile
from pathlib import Path
import sys
# from functools import partial
import argparse
import datetime, time
# import glob
import json
import asyncio
import pandas as pd
import logging
import numpy as np
import cv2


# FLAMEvatar imports; if statement for documentation
if __name__ == '__main__':
    # use a local config value to set default values
    with open("config.json") as f:
        local_config = json.load(f)
    # import the library
    sys.path.append(local_config["base_dir"])
    from request_handler import RequestHandler, time_hns





# goes through 'openface' folder to find latest .npy
class CrawlerNPY:
    """Crawls through a directory to look for .npy generate by FLAME and the already cleaned versions

    Filenames with _P* are messaged together
    """

    def __init__(self):
        pass
        #self.filter_npy = FilterNPY()

    # returns list of .npy files used for generating messages
    def gather_npy_list(self, npy_folder_raw, npy_arg):
        npy_folder_raw = Path(npy_folder_raw)

        # get all npy files in folder raw
        npy_raw = self.search_npy(npy_folder_raw)
        print(npy_raw)

        # # rename folder to folder_clean
        # npy_folder_clean = npy_folder_raw.parent / (npy_folder_raw.parts[-1] + '_clean')
        # # get all npy files in folder clean
        # npy_clean = self.search_npy(npy_folder_clean)

        # raw and clean folder not found, return empty list
        # if not npy_raw and not npy_clean:
        #     return []

        # perform cleaning on files in raw that have not been cleaned yet
        # for raw in npy_raw:
        #     # no cleaned file exist
        #     if raw not in npy_clean:
        #         # call clean on npy and save in clean folder
        #         self.filter_npy.clean_controller(npy_folder_raw / raw, npy_folder_clean)

        #   use argument to determine which npy files will be returned for message generation
        npy_message_list = []
        # find specific file if file name and not a number is given as argument
        if not npy_arg.isdigit() and (npy_arg != '-1') and (npy_arg != '-2'):
            print(f"\nFile is given as argument: {npy_arg}")
            npy_message_list = [sorted(self.search_npy(npy_folder_raw, npy_arg, True))]

        # number is given as argument
        else:
            npy_arg = int(npy_arg)
            print(f"Number is given as argument: {npy_arg}")
            # get all cleaned npy, including new ones
            npy_all_clean = sorted(self.search_npy(npy_folder_raw, "*", True))

            # files were found
            if npy_all_clean:
                print(f"All files found in {npy_folder_raw}")
                no_files = len(npy_all_clean)
                for i, npy in enumerate(npy_all_clean):
                    print(f"[{i}] {npy.name}")

                # return all files
                if npy_arg == -2:
                    # every npy in separate list to create npy groups existing out of 1 npy file
                    print("\nre-listing")
                    print(npy_all_clean)
                    npy_message_list = [[x] for x in npy_all_clean]
                    print()
                    print(npy_message_list)
                    print("\n")

                # return specific file
                elif npy_arg >= -1:
                    user_input = None

                    if npy_arg >= 0:
                        user_input = npy_arg

                    # let user choose file if no file selected
                    while not isinstance(user_input, int) or not 0 <= user_input < no_files:
                        user_input = input("Please choose file you want to send as messages: ")
                        try:
                            user_input = int(user_input)

                            # if user number is outside numbered files
                            if not 0 <= user_input < no_files:
                                print("Input number does not match any listed file")

                        except ValueError:
                            print("Given input is not a number")

                    # single npy file in npy group
                    npy_message_list = [[npy_all_clean[user_input]]]

            else:
                print(f"No npy files found in folder {npy_folder_raw}")

        # return final list of npy files
        print(f"List of npy files for messaging: {npy_message_list}")
        return npy_message_list

    # return latest .npy
    def search_npy(self, npy_path, npy_arg="*", full_path=False):
        if npy_path.exists():
            # add .npy if filename is given without
            if npy_arg[-4:] != ".npy":
                npy_arg += ".npy"

            # find all files matching argument
            npy_path_list = npy_path.glob(npy_arg)

            npy_file_list = []
            for npy_file in npy_path_list:
                if full_path:
                    npy_file_list.append(npy_file)
                else:
                    npy_file_list.append(npy_file.name)

            return npy_file_list

        else:
            print(f"Folder '{npy_path}' not found.")
            return []


class MFCCMessage:
    """FLAME npy based Dataframe to ZeroMQ message"""

    def __init__(self, smooth=True):
        self.msg = dict()
        self.smooth = smooth

    def set_df(self, npy):
        print(f"setting npy {npy.shape}")
        self.npy = npy

    def set_msg(self, frame_tracker):
        # This is an iterator function
        # get single frame
        row = self.npy[frame_tracker][:]
        #print(f"get row {row}")
        # init a message dict
        self.msg = dict()
        self.msg['confidence'] = 1.0
        # setup dummy time stamp as it does not have recorded one
        self.msg['timestamp'] = time.mktime(datetime.datetime.now().timetuple()) * 1000
        self.msg['frame'] = frame_tracker
        if not self.smooth:
            self.msg['smooth'] = False
        # tolist convert an nd array to be able to serialize
        self.msg['mfcc'] = row.tolist()
        self.msg['timestamp_utc'] = time_hns()  # self.time_now()

    def set_reset_msg(self):
        # init a message dict
        self.msg = dict()
        # metadata in message
        self.msg['frame'] = -1
        self.msg['smooth'] = self.smooth  # don't smooth these data
        self.msg['Exp'] = np.zeros(128)



# generator for rows in FLAME / head pose dataframe from FilterNPY
class FLAMEMsgFromNPY:
    """
    Publishes FLAME (Action Units) and head pos data from a cleaned FLAME .npy

    """

    def __init__(self, npy_arg, npy_folder='flame', every_x_frames=1, reset_frames=0, smooth=True, preprocess_model=None):  # client
        """
        generates messages from FLAME .npy files

        :param npy_arg: npy_file_name, -2, -1, >=0
        :param npy_folder: where to look for npy files
        :param every_x_frames: send message when frame % every_x_frames == 0
        """

        self.crawler = CrawlerNPY()
        self.npy_list = self.crawler.gather_npy_list(npy_folder, npy_arg)
        print(f"using npy files: {self.npy_list}")
        self.reset_msg = MFCCMessage(False)
        self.reset_msg.set_reset_msg()
        self.every_x_frames = every_x_frames
        self.reset_frames = reset_frames
        self.smooth = smooth
        self.preprocess = preprocess_model

    # loop over all npy groups ([1 npy file] if single person, P1, P2, etc [multi npy files]
    async def msg_gen(self):
        print(f"npy group: {self.npy_list}")
        for npy_group in self.npy_list:
            print("\n\n")
            time_start = time.time()
            print(f"npy group: {npy_group}")

            async for i, msg in self.msg_from_npy(npy_group):
                #print(f"JSON message just before publish {msg}")
                timestamp = time.time()

                # return filename, timestamp and msg as JSON string
                yield f"p{i}." + npy_group[i].stem, timestamp - time_start, json.dumps(msg)

            # send empty frames
            if self.reset_frames > 0:
                # send few empty messages when npy group is done
                await asyncio.sleep(.5)
                for i in range(self.reset_frames):
                    # self.reset_msg.msg['frame'] += i
                    await asyncio.sleep(.1)
                    yield "reset", timestamp - time_start, json.dumps(self.reset_msg.msg)
                await asyncio.sleep(.2)

        # return that messages are finished (Python >= 3.6)
        yield None

    # generator for FLAME and head pose messages
    async def msg_from_npy(self, npy_group):
        """
        Generates messages from a npy file

        :param npy_group: list of path + file name(s) of npy file(s)
        :return: data of a message to be send in JSON
        """

        # if no npy in npy_group; range(0) == skip
        df_au_row_count = 0
        # dataframe per npy file
        ofmsg_list = []
        test_audio = None
        # load FLAME npy as dataframe
        for audio_fp in npy_group:
            # read npy
            p1_audio = np.load(audio_fp)

            tmp_audio = p1_audio.astype(np.float32)[:,:,:]
            print(f"tmp_audio shape: {tmp_audio.shape}")

            if self.smooth:
                tmp_audio = self.bilateral_filter(tmp_audio)
            self.standardize_speaker_data(tmp_audio)

            count_b = 0
            for batch in range(tmp_audio.shape[0]):
                if test_audio is None:
                    #print("set initial batch")
                    test_audio = tmp_audio[batch,:,:]
                else:
                    #print(f"count {count_b}")
                    test_audio = np.concatenate((test_audio, tmp_audio[batch,:,:]), axis=0)
                count_b += 1

            #MFCC has no smooth function yet
            ofmsg = MFCCMessage(smooth=False)
            ofmsg.set_df(test_audio)
            ofmsg_list.append(ofmsg)
            print(f"ofmsg_list {ofmsg_list}")

        # get current time to match timestamp when publishing
        timer = time.time()

        # send all rows of data 1 by 1
        # message preparation before sleep, then return data
        for frame_tracker in range(test_audio.shape[0]):
            print("FRAME TRACKER: {}".format(frame_tracker))

            # set message for every data frame
            for ofmsg in ofmsg_list:
                ofmsg.set_msg(frame_tracker)

            # get recorded timestamp from first user
            time_npy = ofmsg_list[0].msg['timestamp']

            # # Sleep before sending the messages, so processing time has less influence
            # wait until timer time matches timestamp
            #time_sleep = time_npy - (time.time() - timer)
            time_sleep =0.031 #30 fps
            print("waiting {} seconds before sending next message".format(time_sleep))

            # don't sleep negative time
            if time_sleep > 0:
                # currently can send about 3000 fps
                await asyncio.sleep(time_sleep)  # time_sleep (~0.031)
                # print("Test mode, no sleep: {}".format(time_sleep))

            # reduce frame rate
            if frame_tracker % self.every_x_frames == 0:
                for i, ofmsg in enumerate(ofmsg_list):
                    print(f"DEBUG MSG {ofmsg}")
                    # return msg dict
                    if 'mfcc' in ofmsg.msg:
                        yield i, ofmsg.msg
                    # not enough confidence; return empty msg
                    else:
                        yield i, ''

    def bilateral_filter(self, outputs):
        #print(outputs)
        """ smoothing function

        function that applies bilateral filtering along temporal dim of sequence.
        """
        pass

    def standardize_speaker_data(self, data_X):
        if self.preprocess is not None:
            body_mean_audio = self.preprocess['body_mean_audio']
            body_std_audio = self.preprocess['body_std_audio']

            data_X = (data_X - body_mean_audio) / body_std_audio
            print(f"normalized data: {data_X.shape}")
            return data_X
        else:
            import warnings
            warnings.warn('Could not find the trained model path to access mean and std')
            return data_X

class FLAMEvatarMessages(RequestHandler):
    """Publishes FLAME and Head movement data from .npy files generated by FLAME"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # init class to process .npy files
        logging.debug(f"{self.misc['smooth']}")
        preprocess_model = np.load(os.path.join(self.misc['model_path'],
                         '{}{}_preprocess_core.npz'.format(self.misc['tag'],self.misc['pipeline'] )))
        self.flame_msg = FLAMEMsgFromNPY(self.misc['npy_arg'], self.misc['npy_folder'],
                                               int(self.misc['every_x_frames']), int(self.misc['reset_frames']),
                                               self.misc['smooth'],preprocess_model)

    # publishes flame values per frame to subscription key 'flame'
    async def flame_pub(self):
        """Calls openface_msg.msg_gen() and publishes returned data"""

        msg_count = 0
        start_time = time.time()
        x = 1  # displays the frame rate every 1 second

        # get FLAME message
        async for msg in self.flame_msg.msg_gen():
            #print(msg)
            # send message if we have data

            if msg:
                #start_time = time.time()  # start time of the loop
                #print(f"data for pub: {msg[2]}")
                await self.pub_socket.pub(data=msg[2] )

                msg_count += 1
                #print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop
                if (time.time() - start_time) > x:
                    print("FPS: ", msg_count / (time.time() - start_time))
                    msg_count = 0
                    start_time = time.time()
            # done
            else:
                print("No more messages to publish; FLAME done")

                # tell network messages finished (timestamp == data == None)
                await self.pub_socket.pub('')


# cast argument to bool
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True


if __name__ == '__main__':
    # use a local config value to set default values
    with open("config.json") as f:
        local_config = json.load(f)

    sys.path.append(local_config["base_dir"])
    from request_handler import RequestHandler, time_hns


    parser = argparse.ArgumentParser()

    # logging commandline arguments
    parser.add_argument("--module_id", default="speaker_audio_mfcc_streamer",
                        help="Module id for different instances of same module")
    parser.add_argument("--loglevel", default='INFO',
                        help="Specify how detailed the terminal/logfile output should be;"
                             "DEBUG, INFO, WARNING, ERROR or CRITICAL; Default: INFO")

    # publisher setup commandline arguments
    parser.add_argument("--pub_ip", default=local_config["speaker_audio_mfcc_streamer"]["ip"],
                        help="IP (e.g. 192.168.x.x) of where to pub to; Default: 127.0.0.1 (local)")
    parser.add_argument("--pub_port", default=local_config["speaker_audio_mfcc_streamer"]["port"],
                        help="Port of where to pub to; Default: 5569")
    parser.add_argument("--pub_key", default=local_config["speaker_audio_mfcc_streamer"]["topic"],
                        help="Key for filtering message; Default: fspeaker")
    parser.add_argument("--pub_bind", default=True,
                        help="True: socket.bind() / False: socket.connect();"
                             "Default: False")

    # module specific commandline arguments

    parser.add_argument("--npy_arg", default=local_config["speaker_audio_mfcc_streamer"]["file_name"],
                        help="specific npy (allows for wildcard *), "
                             "-2: message all npy in specified folder, "
                             "-1: show npy list from specified folder, "
                             ">=0 choose specific npy file from list")
    parser.add_argument("--npy_folder", default=local_config["base_dir"]+local_config["speaker_audio_mfcc_streamer"]["stream_data"],
                        help="Name of folder with npy files; Default: test data from original")
    parser.add_argument("--every_x_frames", default="1",
                        help="Send every x frames a msg; Default 1 (all)")
    parser.add_argument("--reset_frames", default="0",
                        help="Not used in mfcc streamer")
    parser.add_argument("--smooth", default=False, type=str2bool,
                        help="Mfccs has not smoothing function yet")

    # machine learning specific

    parser.add_argument("--model_path", default=local_config["base_dir"]+local_config["model_path"],
                        help="machine learning model for speaker classification")
    parser.add_argument("--tag", default=local_config["tag"],
                        help="speaker name or tag labeling the model")
    parser.add_argument("--pipeline", default=local_config["pipeline"],
                        help="prefix to indicate which exp, pose feature to have")


    args, leftovers = parser.parse_known_args()
    print("The following arguments are used: {}".format(args))
    print("The following arguments are ignored: {}\n".format(leftovers))

    # init FLAMEvatar message class
    fanpyatar_messages = FLAMEvatarMessages(**vars(args))

    # start processing messages; give list of functions to call async
    fanpyatar_messages.start([fanpyatar_messages.flame_pub])
