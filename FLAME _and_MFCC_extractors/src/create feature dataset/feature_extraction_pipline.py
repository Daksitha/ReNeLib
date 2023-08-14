"""
p0_list_faces_clean_deca.npy - face features (N x 64 x 184) for when p0 is listener
N sequences of length 64. Features of size 184, which includes the deca parameter set of expression (50D), pose (6D), and details (128D).
p0_speak_audio_clean_deca.npy - audio features (N x 256 x 128) for when p0 is speaking
N sequences of length 256. Features of size 128 mel features
p0_speak_faces_clean_deca.npy - face features (N x 64 x 184) for when p0 is speaking
p0_speak_files_clean_deca.npy - file names of the format (N x 64 x 3) for when p0 is speaking

"""
import os
import gc
import librosa
import PIL.Image as Image
import numpy as np
from config.config_manager import SESSION_DIR, DATASET_DIR
from pathlib import Path
from utils.feature_utils import load_mfcc, angle_axis_to_quaternion, str2bool
#from guppy import hpy
from memory_profiler import profile
from tqdm import tqdm

from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
import gdl
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test

import ffmpeg
import torch




##### extractors #####
def extract_mfcc_from_audio(audio_list, args):
    for audio_f in tqdm(audio_list, desc="Extracting mfcc"):
        #print(audio_f)
        try:
            video_f = audio_f.parent / f"{audio_f.stem}.mp4"
            if args.config == "ap":
                mfcc_file = Path(args.mfcc_pth) / f"mfcc_{audio_f.stem}.npy"
            else:
                mfcc_file = audio_f.parent / "mfcc" / f"mfcc_{audio_f.stem}.npy"

            if mfcc_file.exists():
                continue
            #print(video_f)
            probe = ffmpeg.probe(video_f)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            num_frames = int(video_info['nb_frames'])
            mfcc_np = load_mfcc(audio_f, num_frames)

            mfcc_file.parent.mkdir(parents=True, exist_ok=True)
            # take transpose to save in the format of (4T, 128)
            np.save(mfcc_file,mfcc_np.transpose() )
            #print(f"mfcc_file saved at {mfcc_file}")


        except OSError as err:
            print("OS error: {0}".format(err))
        except ValueError:
            print("Could not convert data to an integer.")
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


def flame_collector(args,emoca,flame_file,input_video,output_folder,processed_subfolder,
   batch_size=1,num_workers=1):
    '''
    emoca: model

    '''
    #
    print("Torch empty cash:")
    torch.cuda.memory_summary(device=None, abbreviated=False)
    torch.cuda.empty_cache()
    gc.collect()
    # save flame for the entire video clip
    faces_clean_deca = []
    ## 1) Process the video - extract the frames from video and detected faces
    try:
        dm = TestFaceVideoDM(input_video, output_folder, processed_subfolder=processed_subfolder,
                         batch_size=batch_size, num_workers=num_workers)
        dm.prepare_data()
        dm.setup()
    except FileExistsError as fe:
        import shutil
        # this happens when the process stop without finishing the prepare_data()
        subfld_path = output_folder / processed_subfolder
        print(f"deleting partially processed folder {subfld_path}")
        shutil.rmtree(subfld_path)

        del dm
        dm = TestFaceVideoDM(input_video, output_folder, processed_subfolder=None,
                             batch_size=batch_size, num_workers=num_workers)
        dm.prepare_data()
        dm.setup()



    ## 3) Get the data loadeer with the detected faces
    dl = dm.test_dataloader()
    print(f" type(dl):{len(dl)}")

    ## 4) Run the model on the data
    # torch no grad help to resolve the memory allocation issue
    '''Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward().
     It will reduce memory consumption for computations that would otherwise have requires_grad=True.'''
    model_name = "EMOCA_v2_lr_mse_20"
    if Path(output_folder).is_absolute():
        outfolder = output_folder
    else:
        outfolder = str(Path(output_folder) / processed_subfolder / Path(input_video).stem / "results" / model_name)

    with torch.no_grad():
        for j, batch in enumerate(auto.tqdm(dl)):
            current_bs = batch["image"].shape[0]
            img = batch
            vals, visdict = test(emoca, img)
            for i in range(current_bs):
                name = batch["image_name"][i]


                # name = f"{(j*batch_size + i):05d}"
                name = batch["image_name"][i]
                exp_nparr = vals["expcode"][i].detach().cpu().numpy()
                #pose_nparr = vals["posecode"][i].detach().cpu().numpy()

                # calculate quaternions  from axis angles
                # 6D array in flame: first 3 global roations (axis angles) and last 3 jaw
                posecode_tensor = vals["posecode"][i]
                glob_qat = angle_axis_to_quaternion(posecode_tensor[:3])
                jaw_qat = angle_axis_to_quaternion(posecode_tensor[3:])

                pose_nparr = torch.cat((glob_qat,jaw_qat)).detach().cpu().numpy()
                # 50 exp, global and jaw rotations in quaternions (8)
                flame_frame = np.concatenate((exp_nparr, pose_nparr), axis=None)
                faces_clean_deca.append(flame_frame)

            if args.save_mesh:
                sample_output_folder = Path(outfolder) / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)
                save_obj(emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, i)
            if args.save_images:
                sample_output_folder = Path(outfolder) / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)
                save_images(outfolder, name, visdict, i)
            if args.save_codes:
                sample_output_folder = Path(outfolder) / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)
                save_codes(Path(outfolder), name, vals, i)

    np.save(flame_file, np.asarray(faces_clean_deca))
    #print(faces_clean_deca)

    # use memory profiler to identify accumulating memory
    # delete accumulating memory and call the garbage collector
    del dm.testdata, dm
    del dl, vals, visdict
    gc.collect()
    print("Done flame_collector")

#@profile
def extract_flame_from_video(videolist, args):
    """
        extract the facial expression and pose details of the two faces for each frame in the video.
         For each person combine the extracted features across the video into a (1 x T x (50+6))
         matrix and save to p0_list_faces_clean_deca.npy or p0_speak_faces_clean_deca.npy files respectively.
         Note, in concatenating the features, expression comes first
        """
    # paths

    path_to_models = str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models")
    model_name = 'EMOCA'
    mode = 'detail'
    # mode = 'coarse'
    ## 2) Load the model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()


    #count_v = 0
    for video_pth in tqdm(videolist, desc="Extracting flame parameters"):
        input_video = video_pth
        output_folder = Path(video_pth).parent / "flame"
        vname = Path(input_video).stem
        if args.config == "ap":
            flame_file = Path(args.flame_pth) / f"flame_{str(vname)}.npy"
        else:
            flame_file = Path(output_folder) / f"flame_{str(vname)}.npy"

        # check


        if flame_file.exists():
            continue

        sub_fld = list(Path(output_folder).glob(f"*/{str(vname)}"))

        if len(sub_fld)==1:
            print(f"Picking the sub folder {sub_fld}")
            processed_subfolder = sub_fld[0].parent.stem
        elif len(sub_fld)>1:
            raise RuntimeError(f"Duplicate folders are being created {sub_fld}")
        else:
            processed_subfolder = None

        print(f"{processed_subfolder} is the processed_subfolder")
        flame_collector(args,emoca, flame_file, input_video, output_folder, processed_subfolder,
                        batch_size=30, num_workers=0)


        # if count_v > 2:
        #     h = hpy()
        #     print(h.heap())
        #     break
        #count_v += 1




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-i', '--config', type=str, required=True,choices=['ap','lc'], help="Which configuration to use,"
                                                                                            "for single file feature extraction "
                                                                              "use 'arg parser'=ap and for a large dataset set 'local config'=lc ")
    parser.add_argument('-v','--video_pth',  type=str,default="", help="input video file to extract FLAME" )
    parser.add_argument('-a', '--audio_pth',type=str,default="", help='input audio file to extract MFCC (.wav, .mp3)')
    parser.add_argument('-f', '--flame_pth',type=str,default="../data/flame_extracted", help='dir path to save the extracted {filename}_flame.npy')
    parser.add_argument('-m', '--mfcc_path', type=str, default="../data/mfcc_extracted", help='dir path to save the extracted {filename}_mfcc.npy')
    parser.add_argument('--save_images', type=str2bool, default=False, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=str2bool, default=True,
                        help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=str2bool, default=False, help="If true, output meshes will be saved")


    args= parser.parse_args()
    if args.config == "ap":
        videolist = [args.video_pth]
        print(f"extracting flame from {videolist}")
        if len(args.video_pth):
            if os.path.isfile(args.video_pth):
                extract_flame_from_video(videolist, args)
            else:
                raise "invalid video file to extract FLAME"
        else:
            print("no video file given to extract FLAME features")

        # MFCC extraction

        print(f"extracting mfcc from {args.audio_pth}")
        if len(args.audio_pth):
            if os.path.isfile(args.audio_pth):
                extract_mfcc_from_audio([args.audio_pth], args)
            else:
                raise "invalid audio file to extract MFCC"
        else:
            print("no audio file given to extract MFCC features")


    elif args.config == "lc":
        # place all videos and audios in wav format inside the dataset directory
        # extract flame
        videolist = sorted(Path(DATASET_DIR).glob(f"*.mp4"))
        print(f"extracting flame from {videolist}")
        if len(videolist):
            extract_flame_from_video(videolist)


        # MFCC extraction
        audio_list = sorted(Path(DATASET_DIR).glob(f"*.wav"))
        print(f"extracting mfcc from {audio_list}")
        if len(audio_list):
            extract_mfcc_from_audio(audio_list)
    else:
        raise "Invalid configuration type, please use local config or arg parser config"
