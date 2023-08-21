import parser
import warnings

from speaker_diarization import speaker_diarisation, define_speech_regions, get_nonoverlapping_speaker_annotations
from config.config_manager import SESSION_DIR, SPEECH_COLLAR, ODP_SESSIONS_JSON, DATASET_DIR, VERBOS, ORIGINAL_VIDEO
from utils.videoutils import extract_video_segment, ffmpeg_crop_video, safe_move, video_convert_to_fps

from pathlib import Path
from tqdm import tqdm
from pyannote.database.util import load_rttm
from utils.audioutils import ffmpeg_extract_audio

import ffmpeg
import tempfile
from moviepy.editor import *
import argparse
from random import shuffle

import sys




def convert_audio_and_video(session_range1,session_range2, fps=25):

    files = list(Path(ORIGINAL_VIDEO).glob(f"{session_range1}*.mp4")) + \
            list(Path(ORIGINAL_VIDEO).glob(f"{session_range2}*.mp4"))
    print(files)

    for file in tqdm(files, desc=f"videos to {fps}fps and audio split",):
        session_id = file.stem
        print(session_id)
        new_video_file = Path(SESSION_DIR) / session_id / f"video_{fps}fps.mp4"
        new_audio_file = Path(SESSION_DIR) / session_id / f"audio_c1_c2.wav"


        # this is to prevent from two containers accessing the same
        #if new_video_file.parent.exists():
            #continue
        new_video_file.parent.mkdir(parents=True, exist_ok=True)
        if new_video_file.exists():
            print(f"video file already exist at {new_video_file}")
            continue
        #clip = VideoFileClip(str(file))
        #clip.write_videofile(str(new_video_file), fps=fps, codec='libx264',audio=True)
        try:
            video_convert_to_fps(fps=fps, input_video=file,output_video=new_video_file)
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e
        print("done converting the video. Moving to audio")
        if new_audio_file.exists():
            continue

        ffmpeg_extract_audio(inputfile=str(new_video_file), outputfile=str(new_audio_file))
        #
        # with tempfile.TemporaryDirectory() as tmpd:
        #     tmp_vf = Path(tmpd) / "video.mp4"
        #
        #     try:
        #         ffmpeg.input(str(file)).output(str(tmp_vf), r=str(fps), vsync="1").run(
        #             quiet=True
        #         )
        #
        #         new_video_file.parent.mkdir(parents=True, exist_ok=True)
        #         # shutil.move(tmpf, new_file)
        #         safe_move(tmp_vf, new_video_file)
        #         ffmpeg_extract_audio(inputfile=str(tmp_vf), outputfile=str(new_audio_file))
        #     except ffmpeg.Error as e:
        #         print('stdout:', e.stdout.decode('utf8'))
        #         print('stderr:', e.stderr.decode('utf8'))
        #         raise e



def generate_roi(session_range1,session_range2, replace=True):
    audio_wav_list = list(Path(SESSION_DIR).glob(f"{session_range1}*/audio_c1_c2.wav")) + \
                     list(Path(SESSION_DIR).glob(f"{session_range2}*/audio_c1_c2.wav"))
    # shuffle them so that two containers won't access the same
    #shuffle(audio_wav_list)
    for audio_file in tqdm(audio_wav_list, desc="generate region of interests"):

        annotation_full, rttm_file = speaker_diarisation(audio_file)
        speaker_dia_dir = rttm_file.parent
        session_id = audio_file.parent.name
        region_of_interest = speaker_dia_dir / f"speech_roi_{session_id}.rttm"

        if region_of_interest.exists() and (not replace):
            continue
        elif replace or not region_of_interest.exists():
            # .unlink()
            for deprecated in list(speaker_dia_dir.glob("speech_roi*.rttm")):
                print(f"{deprecated} is deleted deprecated") if VERBOS else 0
                deprecated.unlink()

        # NOTE: check your uri for channel name. Mine is audio_c1_c2

        # output
        output_folder = rttm_file.parent
        if annotation_full.label_duration("SPEAKER_00") > annotation_full.label_duration("SPEAKER_01"):
            mapping = {"SPEAKER_00": "Patient", "SPEAKER_01": "Therapist"}
        else:
            mapping = {"SPEAKER_00": "Therapist", "SPEAKER_01": "Patient"}

        # rename
        rnmd_annotation = annotation_full.rename_labels(mapping=mapping)

        for person in ["Therapist", "Patient"]:
            anno_person = rnmd_annotation.subset(labels=[person])
            bc_annotation, ss_annotation, ls_annotation = define_speech_regions(full_annotations=anno_person,
                                                                                output_folder=output_folder,
                                                                                label=person, collar_=SPEECH_COLLAR,
                                                                                session_id=session_id)
            # Only take the long speech segments
            if person == "Therapist":
                therapist_lspeaking = ls_annotation
            else:
                patient_lspeaking = ls_annotation

        feature_annotations = get_nonoverlapping_speaker_annotations(therapist_ls_anno=therapist_lspeaking,
                                                                     patient_ls_anno=patient_lspeaking)

        with open(region_of_interest, 'w') as roi_rttm:
            feature_annotations.write_rttm(roi_rttm)


def create_dataset(session_range1,session_range2,replace=False):
    roi_rttm = list(Path(SESSION_DIR).glob(f"{session_range1}*/*/speech_roi_OPD*.rttm")) + \
               list(Path(SESSION_DIR).glob(f"{session_range2}*/*/speech_roi_OPD*.rttm"))
    # shuffle them so that two containers won't access the same
    #shuffle(roi_rttm)

    for roi_pth in tqdm(roi_rttm, desc="create dataset"):
        # there were some currupted roi files
        try:
            roi_anno = load_rttm(file_rttm=str(roi_pth))
        except OSError as oe:
            print(oe)
            roi_pth.unlink()
            generate_roi(session_range1,session_range2)
            roi_anno = load_rttm(file_rttm=str(roi_pth))

        roi_annotations = roi_anno["therapist.pation.longspeech"]
        session_id = roi_pth.parents[1].name

        # session
        if session_id in ODP_SESSIONS_JSON["sessions"]:
            # print(f"{session_id} is in ODP session json")
            therapis_id = ODP_SESSIONS_JSON["sessions"][f"{session_id}"]["therapist_id"]
            dataset_dir = Path(DATASET_DIR) / f"Therapist_{therapis_id}" / f"{session_id}"
        else:
            import warnings
            warnings.warn(f"{session_id} won't be included in the dataset")
            continue

        # chunck videos into ROI
        video_path = roi_pth.parents[1] / "video_25fps.mp4"
        for segment, track, label in roi_annotations.itertracks(yield_label=True):
            output_v_seg = dataset_dir / f"{label}" / f"{label}-{round(segment.start, 2)}-{round(segment.end, 2)}.mp4"
            output_v_seg.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not output_v_seg.exists() or replace:
                    extract_video_segment(segment, str(video_path), str(output_v_seg))
            except OSError as oe:
                print(oe)
                video_path.unlink()



def crop_videos(session_range1,session_range2, label: str, replace=False):
    # print(video_segments)
    if label == "Patient":
        video_aud_segments = list(Path(DATASET_DIR).glob(f"*/{session_range1}*/*/Patient_Long_Speech*.mp4"))+ \
                             list(Path(DATASET_DIR).glob(f"*/{session_range2}*/*/Patient_Long_Speech*.mp4"))
    elif label == "Therapist":
        video_aud_segments = list(Path(DATASET_DIR).glob(f"*/{session_range1}*/*/Therapist_Long_Speech*.mp4")) + \
                             list(Path(DATASET_DIR).glob(f"*/{session_range2}*/*/Therapist_Long_Speech*.mp4"))
    # shuffle them so that two containers won't access the same
    #shuffle(video_aud_segments)

    for va_seg in tqdm(video_aud_segments, desc=f"cropping videosfiles"):
        session_id = va_seg.parents[1].name
        #print(session_id)
        if session_id in ODP_SESSIONS_JSON["sessions"]:
            if label == "Patient":
                spk_p1, spk_p2 = ODP_SESSIONS_JSON["sessions"][f"{session_id}"]["patient_crop"]
                list_p1, list_p2 = ODP_SESSIONS_JSON["sessions"][f"{session_id}"]["therapist_crop"]
            elif label == "Therapist":
                spk_p1, spk_p2 = ODP_SESSIONS_JSON["sessions"][f"{session_id}"]["therapist_crop"]
                list_p1, list_p2 = ODP_SESSIONS_JSON["sessions"][f"{session_id}"]["patient_crop"]
        else:
            warnings.warn(f"{session_id} is not in the json")

        crop_video_speaking_old= va_seg.parent / f"crop_{va_seg.stem}.mp4"
        crop_video_speaking_new = va_seg.parent / f"speaking_{va_seg.stem}.mp4"
        crop_video_listening = va_seg.parent / f"listening_{va_seg.stem}.mp4"

        try:
            # speaker video crop
            if not crop_video_speaking_old.exists() or not crop_video_speaking_new or replace:
                ffmpeg_crop_video(filename=str(va_seg),p1=spk_p1, p2=spk_p2, output=str(crop_video_speaking_new))
            # listener video crop
            if not crop_video_listening.exists() or replace:
                ffmpeg_crop_video(filename=str(va_seg),p1=list_p1, p2=list_p2, output=str(crop_video_listening))
        except OSError as oe:
            print(oe)
            # delete the corrupted files
            va_seg.unlink()

        # rename prefix that named before
        if crop_video_speaking_old.exists():
            old_name = crop_video_speaking_old.stem
            old_extension = crop_video_speaking_old.suffix
            directory = crop_video_speaking_old.parent
            new_name = str(old_name).replace("crop","speaking")+ old_extension
            crop_video_speaking_old.rename(Path(directory, new_name))


def extract_audio_from_video(session_range1,session_range2, label: str, replace=False):
    # print(video_segments)
    if label == "Patient":
        video_aud_segments = list(Path(DATASET_DIR).glob(f"*/{session_range1}*/*/Patient_Long_Speech*.mp4"))+ \
                             list(Path(DATASET_DIR).glob(f"*/{session_range2}*/*/Patient_Long_Speech*.mp4"))

    elif label == "Therapist":
        video_aud_segments = list(Path(DATASET_DIR).glob(f"*/{session_range1}*/*/Therapist_Long_Speech*.mp4")) + \
                             list(Path(DATASET_DIR).glob(f"*/{session_range1}*/*/Therapist_Long_Speech*.mp4"))

    # shuffle them so that two containers won't access the same
    #shuffle(video_aud_segments)

    for va_seg in tqdm(video_aud_segments, desc=f"extracting audio files"):
        session_id = va_seg.parents[1].name
        #print(session_id)
        audio_path = va_seg.parent / f"{va_seg.stem}.wav"
        if not audio_path.exists() or replace:
            try:
                ffmpeg_extract_audio(inputfile=str(va_seg), outputfile=audio_path)
            except OSError as oe:
                print(oe)
                va_seg.unlink()
                create_dataset(session_range1,session_range2)
                ffmpeg_extract_audio(inputfile=str(va_seg), outputfile=audio_path)






if __name__ == "__main__":
    # podman feature_OPD1_OPD2 container run OPD_1** and OPD_2** sessions ranges
    # podman feature_OPD3_OPD4 container run OPD_3** and OPD_4** ranges
    # podman feature_OPD5_OPD6 container run OPD_5** and OPD_6** ranges
    # podman feature_OPD7 container run OPD_7** ranges
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_range1',type=str,
                        choices=["OPD_1","OPD_3","OPD_5","OPD_7",""],
                        help="prefix of the video sessions to process")
    parser.add_argument('--session_range2', type=str,
                        choices=["OPD_2", "OPD_4", "OPD_6", "OPD_8",""],
                        help="prefix of the video sessions to process")

    args = parser.parse_args()
    session_range1 = args.session_range1
    session_range2 = args.session_range2
    #there can be currupted files
    convert_audio_and_video(session_range1,session_range2)
    generate_roi(session_range1,session_range2)
    create_dataset(session_range1,session_range2)

    extract_audio_from_video(session_range1,session_range2,label="Patient")
    extract_audio_from_video(session_range1,session_range2,label="Therapist")

    crop_videos(session_range1,session_range2,label="Patient")
    crop_videos(session_range1,session_range2,label="Therapist")
