<!-- # LLB -->
<h1 align="center">Setting up EMOCA and MFCC Extractors</h1>


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

### Podman/Docker version

don't use mamba instead use conda to create work38 environment 

```
docker run --rm -it pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

apt-get install sox ffmpeg libcairo2 libcairo2-dev git zip wget
conda create -n work38 python=3.8 
conda activate work38
# comment out all pytorch related libraries in conda-environment_py38_cu11_ubuntu.yml file
conda env update -n work38 --file conda-environment_py38_cu11_ubuntu.yml
pip install -r requirements38.txt
pip install Cython==0.29

# there are two ways to install pytorch3d
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
#or 
conda install -c pytorch pytorch=1.7.0 torchvision cudatoolkit=11.0
conda install -c conda-forge -c iopath fvcore iopath
conda install -c pytorch3d pytorch3d
#test installtion
python
from pytorch3d.structures import Meshes

#install GDL
pip install -e . 
```

Some of the fixes I had to do before running the demo test on images 
```
pip install pandas
# cv2 package gave some errors
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip uninstall opencv-contrib-python-headless
# re-install
pip3 install opencv-contrib-python==4.5.5.62
pip install opencv-python-headless==4.1.2.30
pip install scikit-video
```



Once the container is setup. Install apt-get git, wget,zip

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


