import warnings

from config.config_manager import SESSION_DIR, DATASET_DIR, ODP_SESSIONS_JSON
from pathlib import Path
import numpy as np
from tqdm import tqdm

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
def prepare_training_data(audio_data,speak_data,liste_data,file_name, bin_size=64):
    speak_bin = []
    list_bin = []
    audio_bin = []
    count = 1

    for spk in chunks(speak_data,bin_size):
        if spk.shape[0]==bin_size:
            speak_bin.append(spk)
        else:
            pass
            #print(f"Debug: short of bin_size {spk.shape}")
    #print(f"{np.asarray(speak_bin).shape}")

    for lst in chunks(liste_data,bin_size):
        if lst.shape[0] == bin_size:
            list_bin.append(lst)
        else:
            pass
            #print(f"Debug: short of bin_size {lst.shape}")
    #print(f"{np.asarray(list_bin).shape}")

    for audio in chunks(audio_data, bin_size*4):
        if audio.shape[0] == bin_size*4:
            audio_bin.append(audio)
        else:
            pass
            #print(f"Debug: short of bin_size {audio.shape}")
    #print(f"{np.asarray(audio_bin).shape}")

    return np.asarray(speak_bin), np.asarray(list_bin),  np.asarray(audio_bin)


def containerize_data(sessions):
    for sess in tqdm(sessions, desc="grouping flame and mfcc"):
        # Patient_Long_Speech listening and speaking is what matter for L2L
        # saved data
        mfcc_f = sorted(Path(sess).glob(f"Patient_Long_Speech/mfcc/mfcc*.npy"))
        flame_speak_f = sorted(Path(sess).glob(f"Patient_Long_Speech/flame/flame_speaking*.npy"))
        flame_list_f = sorted(Path(sess).glob(f"Patient_Long_Speech/flame/flame_listening*.npy"))

        # machine learning ready data
        if sess.stem in ODP_SESSIONS_JSON["sessions"]:
            therapis_id = ODP_SESSIONS_JSON["sessions"][f"{sess.stem}"]["therapist_id"]
            patient_position = ODP_SESSIONS_JSON["sessions"][f"{sess.stem}"]["patient_position"]
            if patient_position in ['Left', 'left', 'l', 'L']:
                therapist_position = "Right"
            else:
                therapist_position = "Left"
        else:
            warnings.warn(f"{sess} metadata is not available")
            contiue
        # p0_list_faces_clean_deca.npy p0_speak_audio_clean_deca.npy p0_speak_faces_clean_deca.npy p0_speak_files_clean_deca.npy
        therapy_deca_list = sess / f"{therapist_position}_therapy_list_faces_clean_deca.npy"
        therpy_deca_list_arr = None
        patient_deca_speak = sess / f"{patient_position}_patient_speak_faces_clean_deca.npy"
        patient_deca_speak_arr = None
        patient_audio_speak = sess / f"{patient_position}_patient_speak_audio_clean_deca.npy"
        patient_audio_speak_arr = None

        if therapy_deca_list.exists() and patient_deca_speak.exists() and patient_audio_speak.exists():
            # contiue
            pass

        flame_dir = flame_speak_f[0].parent
        if len(flame_list_f) == len(mfcc_f) == len(flame_speak_f):
            # print(f"############ session {sess.stem} ###########")
            # print(f"audio len: {len(mfcc_f)}, speaking_flame len: {len(flame_speak_f)}, listening flame len {len(flame_list_f)}")

            for short_session in tqdm(mfcc_f, desc="go through all chunks of videos, audios"):
                # print(short_session.stem)
                fspeaking_name = short_session.stem.replace('mfcc_', 'flame_speaking_') + '.npy'
                flistening_name = short_session.stem.replace('mfcc_', 'flame_listening_') + '.npy'

                fspeaking_file = flame_dir / fspeaking_name
                flistening_file = flame_dir / flistening_name
                if (fspeaking_file in flame_speak_f) and (flistening_file in flame_list_f):
                    audio_data = np.load(short_session)
                    speak_data = np.load(fspeaking_file)
                    liste_data = np.load(flistening_file)
                    # print(f"audio_data: {audio_data.shape}, speak_data: {speak_data.shape}, audio_data {liste_data.shape}")

                    speak_bin, list_bin, audio_bin = prepare_training_data(audio_data, speak_data, liste_data,
                                                                           short_session)

                    if patient_audio_speak_arr is None and list_bin.ndim == 3 and speak_bin.ndim == 3 and audio_bin.ndim == 3:
                        therpy_deca_list_arr = list_bin
                        patient_deca_speak_arr = speak_bin
                        patient_audio_speak_arr = audio_bin
                    elif list_bin.ndim == 3 and speak_bin.ndim == 3 and audio_bin.ndim == 3:

                        therpy_deca_list_arr = np.concatenate((therpy_deca_list_arr, list_bin[:, :, :]))
                        patient_deca_speak_arr = np.concatenate((patient_deca_speak_arr, speak_bin[:, :, :]))
                        patient_audio_speak_arr = np.concatenate((patient_audio_speak_arr, audio_bin[:, :, :]))

                    else:
                        pass
                        # print(f"speak_bin: {speak_bin.shape}, list_bin: {list_bin.shape}, audio_bin {audio_bin.shape}")

                    # print(f"therpy_deca_list_arr: {therpy_deca_list_arr.shape}, patient_deca_speak_arr: {patient_deca_speak_arr.shape},"
                    #      f" patient_audio_speak_arr {patient_audio_speak_arr.shape}")


                else:
                    raise RuntimeError("ups incomplete session files")

            # save data
            np.save(therapy_deca_list, therpy_deca_list_arr)
            np.save(patient_deca_speak, patient_deca_speak_arr)
            np.save(patient_audio_speak, patient_audio_speak_arr)
            # print(f"files are saved to {sess}")
            print(
                f"session: {sess},therpy_deca_list_arr: {therpy_deca_list_arr.shape}, patient_deca_speak_arr: {patient_deca_speak_arr.shape},"
                f" patient_audio_speak_arr {patient_audio_speak_arr.shape} saved")
            # break

def prepare_dataset_l2l(sessions):
    # p0 as the person on the left side of the video, and p1 as the right side.
    # p0_list_faces_clean_deca.npy, p0_speak_audio_clean_deca.npy, p0_speak_faces_clean_deca.npy, p0_speak_files_clean_deca.npy
    sessions_dir = sessions[0].parent
    # left
    p0_list_faces_f = sessions_dir / "benecke_left"/ "p0_list_faces_clean_deca.npy"
    p0_speak_audio_f = sessions_dir / "benecke_right"/ "p0_speak_audio_clean_deca.npy"
    p0_speak_faces_f = sessions_dir / "benecke_right"/ "p0_speak_faces_clean_deca.npy"
    p0_list_faces, p0_speak_audio,p0_speak_faces = None, None, None
    # right
    p1_list_faces_f = sessions_dir /"benecke_right"/ "p1_list_faces_clean_deca.npy"
    p1_speak_audio_f = sessions_dir /"benecke_left"/ "p1_speak_audio_clean_deca.npy"
    p1_speak_faces_f = sessions_dir /"benecke_left"/ "p1_speak_faces_clean_deca.npy"
    p1_list_faces, p1_speak_audio, p1_speak_faces = None, None, None
    #.parent.mkdir(parents=True, exist_ok=True)
    p1_list_faces_f.parent.mkdir(parents=True, exist_ok=True)
    p0_list_faces_f.parent.mkdir(parents=True, exist_ok=True)

    for sess in tqdm(sessions, desc="data preperation for l2l"):
        #
        npy_data = sorted(Path(sess).glob(f"*.npy"))
        #print(sess.stem)
        for npy in npy_data:

            if 'Left_therapy' in npy.stem:
                if p0_list_faces is None:
                    p0_list_faces = np.load(npy)
                    # right patient
                    p1_speak_audio = np.load(f"{npy.parent}/Right_patient_speak_audio_clean_deca.npy")
                    p1_speak_faces = np.load(f"{npy.parent}/Right_patient_speak_faces_clean_deca.npy")
                    print(p0_list_faces.shape, p1_speak_audio.shape,p1_speak_faces.shape )

                p0_list_faces = np.concatenate((p0_list_faces,np.load(npy)))
                # right patient
                p1_speak_audio = np.concatenate((p1_speak_audio,np.load(f"{npy.parent}/Right_patient_speak_audio_clean_deca.npy")))
                p1_speak_faces = np.concatenate((p1_speak_faces,np.load(f"{npy.parent}/Right_patient_speak_faces_clean_deca.npy")))
            if 'Right_therapy' in npy.stem:
                if p1_list_faces is None:
                    p1_list_faces = np.load(npy)
                    # Left_patient
                    p0_speak_audio = np.load(f"{npy.parent}/Left_patient_speak_audio_clean_deca.npy")
                    p0_speak_faces = np.load(f"{npy.parent}/Left_patient_speak_faces_clean_deca.npy")
                    print(p1_list_faces.shape, p0_speak_audio.shape, p0_speak_faces.shape)
                p1_list_faces = np.concatenate((p1_list_faces,np.load(npy)))
                # Left_patient
                p0_speak_audio = np.concatenate((p0_speak_audio,np.load(f"{npy.parent}/Left_patient_speak_audio_clean_deca.npy")))
                p0_speak_faces = np.concatenate((p0_speak_faces,np.load(f"{npy.parent}/Left_patient_speak_faces_clean_deca.npy")))

    print(f"p1_list_faces:{p1_list_faces.shape}, p0_speak_audio:{p0_speak_audio.shape},p0_speak_faces: {p0_speak_faces.shape}")
    print(f"p0_list_faces:{p0_list_faces.shape}, p1_speak_audio:{p1_speak_audio.shape}, p1_speak_faces:{p1_speak_faces.shape}")

    #save files
    np.save(p0_list_faces_f,p0_list_faces)
    np.save(p0_speak_audio_f, p0_speak_audio)
    np.save(p0_speak_faces_f,p0_speak_faces)

    np.save(p1_list_faces_f, p1_list_faces)
    np.save(p1_speak_audio_f, p1_speak_audio)
    np.save(p1_speak_faces_f, p1_speak_faces)


if __name__ == "__main__":
    sessions = sorted(Path(DATASET_DIR).glob(f"OPD*"))
    prepare_dataset_l2l(sessions)




