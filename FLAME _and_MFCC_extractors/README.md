<!-- # LLB -->
<h1 align="center">Learning Listening Behaviour (LLB): A
Framework for Interactive Social Agents
Employing Multimodal
Patient-Therapist Discourse Information</h1>


## Installation 

### Dependencies

1) Install [conda](https://docs.conda.io/en/latest/miniconda.html)

<!-- 2) Install [mamba](https://github.com/mamba-org/mamba) -->

<!-- 0) Clone the repo with submodules:  -->
<!-- ``` -->
<!-- git clone --recurse-submodules ... -->
<!-- ``` -->
2) Clone this repo

### Short version 

1) Run the installation script: 

```bash
bash install.sh
```
If this ran without any errors, you now have a functioning conda environment with all the necessary packages to [run the demos](#usage). If you had issues with the installation script, go through the [long version](#long-version) of the installation and see what went wrong. Certain packages (especially for CUDA, PyTorch and PyTorch3D) may cause issues for some users. If you prefer containerized version please follow [docker/podman](#podmandocker-version)
### Long version
working progress....
### Podman/Docker version
working progress...
## Extract FLAME and MFCC from an audio and a video file: 
- Place the video file inside data/videos/ directory
- Place the audio file (if there are any; works without audio file as well) in data/audio directory
```bash
python feature_extraction_pipline.py -i ap -v ../data/videos/<video name>.mp4
```
It will save the extracted features, for example FLAME, in data/flame_extracted/```<video name>_flame.npy```.
Similarly, you can follow MFCC extraction from audio files
```bash
python feature_extraction_pipline.py -i ap -a ../data/audio/<audio name>.wav
```

## Create your own dataset (FLAME and MFCC)

### Define regions of interests
1) Configure src/config/config.toml with your project directories. Define speech activity durations in config/config.toml
   (include images)
2) ``src/create_dataset_from_roi.py`` takes prefix from range of videos that you have in your video corpus. The arg --session1 and --session2 are implemented such a way that iterate all videos from OPD1 and OPD2 in one podman container and OPD3 and OPD4 in another containers. This script enables you to process your videos corpus in following order:
- convert_audio_and_video (frame rate, format)
- generate_roi (define regions of interest using speaker diarization)
- create_dataset (split videos and audios using ROI )
- extract_audio_from_video (split audio and video channels)
- crop_videos (crop the videos to separate patient and therapist)


3)  Extractors to features, [FLAME](https://flame.is.tue.mpg.de/index.html) 3D mesh from videos and MFCC from audio files.

   - set src/config/config.toml data directory path.  This dir should include both video and audio files for each session.
   -  Run the script indicating to use local config file (lc)
   ```bash
      python feature_extraction_pipline.py -i lc
   ```
4) format the extracted dataset to train machine learning models such as [L2L](https://evonneng.github.io/learning2listen/)


