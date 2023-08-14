from src.config import SESSION_DIR
from tqdm import tqdm
from pathlib import Path
import subprocess
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import os
from pyannote.core import Segment
from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

import errno
import shutil
import uuid

import ffmpeg
# import cv2


#oroginal moviepy method result blackscreen for 2-3 seconds
#Solu: https://stackoverflow.com/questions/52257731/extract-part-of-a-video-using-ffmpeg-extract-subclip-black-frames
#-vcodec changing from copy to libx264 improve the output video quality
def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
    the times ``t1`` and ``t2``. """
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

    cmd = [get_setting("FFMPEG_BINARY"),"-y",
           "-ss", "%0.2f"%t1,
           "-i", filename,
           "-t", "%0.2f"%(t2-t1),
           "-vcodec", "libx264", "-acodec", "copy", targetname]

    subprocess_call(cmd=cmd ,logger=None)

# def extract_audio_from_video(video_pth,audio_path):
#     #video_list = list(Path(SESSION_DIR).glob("*.mp4"))
#    # for file in tqdm(video_list, desc="extracting audio files"):
#     file = Path(video_pth)
#     #audio_new_path = Path(SESSION_DIR) / file.stem / "audio_c1_c2.wav"
#
#     if not audio_path.exists():
#         audio_path.parent.mkdir(parents=True, exist_ok=True)
#         command = f"ffmpeg -i {file} -ab 160k -ar 44100 -vn {audio_path} "
#         subprocess.call(command, shell=True)
def extract_video_segment(segment: Segment, video_pth: str, out_video_path: str):
    ffmpeg_extract_subclip(filename=video_pth, t1=round(segment.start,2), t2=round(segment.end,2), targetname=out_video_path)



def ffmpeg_crop_video(filename, p1,p2, output):
    """ Makes a new video file with rectangular crop of the origina video
     crop interval
     p1, p2 : [y1,y2] [x1,x2]
     x1,y1 let most corner, x2,y2 are left bottom corners"""

    #crop format: [y1,y2] [x1,x2]
    #p1,p2 = metadata["sessions"][f"{opd_id}"]["patient_crop"]
    x1,y1,h,w = p2[0], p1[0],(p1[1]-p1[0]), (p2[1]-p2[0])
    cmd = [get_setting("FFMPEG_BINARY"),"-y",
           "-i", filename,
           "-filter:v", f'crop={w}:{h}:{x1}:{y1}',
           "-vcodec", "libx264", "-acodec", "copy", output]

    subprocess_call(cmd=cmd ,logger=None)


# def crop_video_from_session_data(p1, p2, video_pth, out_video_path, audio_=True):
#     #format of the ODP_Session_information json p1:[y1,y2], p2:[x1,x2]
#     #px1,py1,ph,pw = p2[0], p1[0],(p1[1]-p1[0]), (p2[1]-p2[0])
#     clip = VideoFileClip(str(video_pth))
#     cropped_video = (clip.fx(vfx.crop,  x1=p2[0], y1=p1[0], x2=p2[1], y2=p1[1]))
#     cropped_video.write_videofile(filename=str(out_video_path), fps=clip.fps, audio=audio_, logger=None)


def safe_move(src, dst):
    """Rename a file from ``src`` to ``dst``.

    *   Moves must be atomic.  ``shutil.move()`` is not atomic.
        Note that multiple threads may try to write to the cache at once,
        so atomicity is required to ensure the serving on one thread doesn't
        pick up a partially saved image from another thread.

    *   Moves must work across filesystems.  Often temp directories and the
        cache directories live on different filesystems.  ``os.rename()`` can
        throw errors if run across filesystems.

    So we try ``os.rename()``, but if we detect a cross-filesystem copy, we
    switch to ``shutil.move()`` with some wrappers to make it atomic.
    """
    try:
        os.rename(src, dst)
    except OSError as err:

        if err.errno == errno.EXDEV:
            # Generate a unique ID, and copy `<src>` to the target directory
            # with a temporary name `<dst>.<ID>.tmp`.  Because we're copying
            # across a filesystem boundary, this initial copy may not be
            # atomic.  We intersperse a random UUID so if different processes
            # are copying into `<dst>`, they don't overlap in their tmp copies.
            copy_id = uuid.uuid4()
            tmp_dst = "%s.%s.tmp" % (dst, copy_id)

            shutil.copyfile(src, tmp_dst)

            # Then do an atomic rename onto the new name, and clean up the
            # source image.
            os.rename(tmp_dst, dst)
            os.unlink(src)
        else:
            raise


# def get_frame_rate(file_path:str):
#     cap = cv2.VideoCapture(file_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cv2.destroyAllWindows()
#     print(f"fps of the  {file_path} is {fps}")
#     return str(fps)


def video_convert_to_fps(fps:int, input_video:str,output_video:str):
    if fps == 25: # NOTE: original video frame rate
        shutil.copy(input_video, output_video)
    else:
        cmd = [get_setting("FFMPEG_BINARY"), "-y",
               "-i", input_video,
               "-filter:v", f"fps=fps={fps}", output_video]
        subprocess_call(cmd=cmd)




