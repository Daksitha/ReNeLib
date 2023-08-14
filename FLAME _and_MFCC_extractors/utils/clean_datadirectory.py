from src.config import SESSION_DIR, SPEECH_COLLAR, ODP_SESSIONS_JSON, DATASET_DIR, VERBOS, ORIGINAL_VIDEO
from pathlib import Path
import shutil
from tqdm import auto
from tqdm import tqdm


def delete_files(file_list):
    print(file_list)
    answer = input("Above files will be deleted: Do you wish to continue? [yes (y),no (n)]")
    if answer.lower() in ["y", "yes"] and len(file_list) > 0:
        for file in  tqdm(file_list, desc="Deleting dir"):
            print(f"deleting {file}")
            file.unlink(missing_ok=False)
    elif answer.lower() in ["n", "no"]:
        print("")
    else:
        raise NameError("Invalid input or empty file list")


def delete_dir(directory_list):
    print(f"len(directory_list): {len(directory_list)}")
    answer = input("Above directories will be deleted: Do you wish to continue? [yes (y),no (n)]")
    if answer.lower() in ["y", "yes"] and len(directory_list) > 0:
        #just to make sure it won't delete ongoing files

        for dir in tqdm(directory_list, desc="Deleting dir"):
            shutil.rmtree(dir)
    elif answer.lower() in ["n", "no"]:
        print("")
    else:
        raise NameError("Invalid input or empty file list")


if __name__ == "__main__":
    #file_list = sorted(Path(DATASET_DIR).glob(f"Therapist*/*/*/Patient_Long_Speech*.mp4")) + \
    #            sorted(Path(DATASET_DIR).glob(f"Therapist*/*/*/Therapist_Long_Speech*.mp4"))
    #file_list = sorted((Path(SESSION_DIR).glob(f"OPD*/video_25fps.mp4")))

    # delete only unpacked images. In the future, should need we can use detected folders to extract
    #emoca

    directory_list = sorted(Path(DATASET_DIR).glob(f"Therapist*/*/*/flame/processed*/*/videos"))
    #npys = sorted(Path(DATASET_DIR).glob(f"Therapist*/*/*/flame/*.npy"))
    #print(directory_list)

    delete_dir(directory_list)

    #delete_files(file_list)


    #print((file_list))
