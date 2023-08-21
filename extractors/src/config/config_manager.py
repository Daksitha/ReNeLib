from pathlib import Path
import toml
import numpy as np
from collections import namedtuple, abc
import json


def update(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


base_path = Path(__file__).resolve().parents[0]
config_path = base_path / "config.toml"

CONFIG = toml.load(config_path.open())

config_local_path = base_path / "config.local.toml"
if config_local_path.exists():
    CONFIG = update(CONFIG, toml.load(config_local_path.open()))

SESSION_DIR = CONFIG["project"]["sessions_dir"]
DATASET_DIR = CONFIG["project"]["dataset"]
ORIGINAL_VIDEO = CONFIG["project"]["video_dir"]
VERBOS = CONFIG["project"]["verbos"]


#speaker diarisation labels
BACKCHANNEL_INTVL = CONFIG["speaker_diarisation"]["back_channel"]
SHORT_SPEECH_INTVL = CONFIG["speaker_diarisation"]["short_speech"]
LONGSPEECH_INTVL = CONFIG["speaker_diarisation"]["long_speech"]
SPEECH_COLLAR = CONFIG["speaker_diarisation"]["speech_collar"]

# load session metadata, for example, where the therapist is seated
json_file = Path(__file__).parent / "ODP_session_information.json"
with open(json_file, 'r') as f:
    ODP_SESSIONS_JSON = json.loads(f.read())


