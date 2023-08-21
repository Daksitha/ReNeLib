

from pyannote.audio import Audio

from moviepy.config import get_setting
from moviepy.tools import subprocess_call

def ffmpeg_extract_audio(inputfile,output,bitrate=16,fps=44100):
    """ extract the sound from a video file and save it in ``output``
    input: input video file
     -ar- set the audio sampling frequency
    -ac- Set the number of audio channels
    -ab- Set the audio bitrate"""
    cmd = [get_setting("FFMPEG_BINARY"), "-y", "-i", inputfile, "-ab", "%dk"%bitrate,
         "-ar", "%d"%fps, output]
    subprocess_call(cmd)

def extract_audio_segment(wav_file, segment):
    wavfrm, sr = Audio().crop(wav_file, segment)
    return wavfrm.flatten(), sr


def ffmpeg_extract_audio(inputfile:str, outputfile:str, fps=44100, logger=None):
    """Extract the sound from a video file and save it in ``outputfile``.

    Parameters
    ----------

    inputfile : str
      The path to the file from which the audio will be extracted.

    outputfile : str
      The path to the file to which the audio will be stored.

    bitrate : int, optional
      Bitrate for the new audio file.

    fps : int, optional
      Frame rate for the new audio file.
    """
    cmd = [
        get_setting("FFMPEG_BINARY"),
        "-y",
        "-i",inputfile,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "%d" % fps,
        outputfile,
    ]
    subprocess_call(cmd, logger=logger)
