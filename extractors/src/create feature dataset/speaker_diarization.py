

from config.config_manager import BACKCHANNEL_INTVL, SHORT_SPEECH_INTVL, LONGSPEECH_INTVL, VERBOS
import torch
torch.cuda.empty_cache()

from pathlib import Path
from tqdm import tqdm
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
Segment.set_precision(2)
from utils.plotting import save_session_histograms

from pyannote.database.util import load_rttm


def speaker_diarisation(audiofile):
    # audio_wav_list = list(Path(SESSION_DIR).glob("OPD*/audio_c1_c2.wav"))
    # for audio_file in tqdm(audio_wav_list, desc="speaker diarisation "):
    audio_file = Path(audiofile)
    out_path_parent = audio_file.parent / "speaker_diarization"

    # out_path_spk01 = out_path_parent / 'spk_01.wav'
    # out_path_spk02 = out_path_parent / 'spk_02.wav'
    session_rtmm = out_path_parent / 'anno_speaker-diarization.rttm'
    session_rtmm.parent.mkdir(parents=True, exist_ok=True)

    if session_rtmm.exists():
        print(f'RTMM File exist, loading the session') if VERBOS else 0
        annotations_dia = load_rttm(session_rtmm)["audio_c1_c2"]
        return annotations_dia, session_rtmm


    # start_time = time.time()
    print('Start processing {}'.format(audio_file)) if VERBOS else 0

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    annotations_dia = pipeline(audio_file, num_speakers=2)

    # Save annotation first
    if not session_rtmm.exists():
        if VERBOS:
            print(f"writing rttm for {out_path_parent}") if VERBOS else 0
        with open(session_rtmm, 'w') as rtmmfile:
            annotations_dia.write_rttm(rtmmfile)

    return annotations_dia, session_rtmm

    # # read original wav
    # audio_data, sr = sf.read(audio_file)
    #
    # # getting segementlist from anno
    # segments = [(s, l) for s, _, l in dia.itertracks(yield_label=True)]
    #
    # print("Audio of length: %.2f sec took %.2f sec to diarize" % (len(audio_data) / sr, time.time() - start_time))
    #
    # spk_01 = np.zeros(audio_data.shape)
    # spk_02 = np.zeros(audio_data.shape)
    #
    # for s, l in segments:
    #     start_idx = int(s.start * sr)
    #     end_idx = int(s.end * sr)
    #
    #     if l == 'SPEAKER_01':
    #         spk_01[start_idx:end_idx] = audio_data[start_idx:end_idx]
    #     else:
    #         spk_02[start_idx:end_idx] = audio_data[start_idx:end_idx]
    #
    # print('Saving diarized files to {}'.format(out_path_parent))
    #
    # sf.write(out_path_spk01, spk_01, sr)
    # sf.write(out_path_spk02, spk_02, sr)
    # print('Done processing {}'.format(audio_file))


def define_speech_regions(full_annotations, output_folder, label, collar_=None,
                          save_summary=True, save_region_rttm=False, session_id=0):
    """
    annotations: full session that needs to be devided. Type is pyannote.Annotation

    session_name: unique to define the session

    label: Timeline name tag
    NOTE:
    time_intervals: can be changed with config parameters
    """
    if collar_ is not None:
        print(f"Merging regions that are {collar_} seconds apart") if VERBOS else 0
        annotations = full_annotations.support(collar=collar_)
    else:
        annotations = full_annotations

    pth = Path(output_folder)
    bc_annotation = Annotation(uri=f"{pth.name}.{label}.Backchannel")
    ss_annotation = Annotation(uri=f"{pth.name}.{label}.Shortspeech")
    ls_annotation = Annotation(uri=f"{pth.name}.{label}.Longspeech")

    bc_durations = []
    ss_durations = []
    ls_durations = []

    for segment, track, label in annotations.itertracks(yield_label=True):

        if BACKCHANNEL_INTVL[0] < segment.duration < BACKCHANNEL_INTVL[1]:
            bc_annotation[segment] = f"{label}_Backchannels"
            bc_durations.append(segment.duration)

        if SHORT_SPEECH_INTVL[0] < segment.duration < SHORT_SPEECH_INTVL[1]:
            ss_annotation[segment] = f"{label}_Short_Speech"
            ss_durations.append(segment.duration)

        if LONGSPEECH_INTVL[0] < segment.duration < LONGSPEECH_INTVL[1]:
            ls_annotation[segment] = f"{label}_Long_Speech"
            ls_durations.append(segment.duration)

    if save_summary:
        save_session_histograms(output_folder, label=label, bc_durations=bc_durations, ss_durations=ss_durations,
                                ls_durations=ls_durations, session_id=session_id)
    if save_region_rttm:
        # bc_rtmm = pth / f"{pth.name}.{label}.Backchannel.rttm"
        # ss_rtmm = pth / f"{pth.name}.{label}.ShortSpeech.rttm"
        ls_rtmm = pth / f"{pth.name}.{label}.LongSpeech.rttm"
        # save only long speech sessions
        if not ls_rtmm.exists():
            with open(ls_rtmm, 'w') as lsrttm:
                lsrttm.write_rttm(ls_rtmm)

    return bc_annotation, ss_annotation, ls_annotation


def get_nonoverlapping_speaker_annotations(therapist_ls_anno: Annotation, patient_ls_anno: Annotation):
    thp_pat_ls_anno = Annotation(uri="therapist.pation.longspeech", modality='speaker')
    for person_anno in [therapist_ls_anno, patient_ls_anno]:
        for segment, track, label in person_anno.itertracks(yield_label=True):
            thp_pat_ls_anno[segment] = label

    overlapping = thp_pat_ls_anno.get_overlap()

    print("mode='intersection'") if VERBOS else 0
    # create annotation timeline removing overlapping segments
    speaking_annotations = thp_pat_ls_anno.extrude(removed=overlapping, mode='intersection')

    return speaking_annotations








