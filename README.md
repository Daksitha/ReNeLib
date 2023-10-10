<h1>ReNeLiB: Real-time Neural Listening Behavior Generation for Socially Interactive Agents</h1>

This paper will be presented at ICMI'23.


![Teaser](docs/teaser.png)

<div class="row">
<div class="col-sm-3"><a href="https://doi.org/10.1145/3577190.3614133" class="btn">The Paper</a></div>
<div class="col-sm-3"><a href="https://github.com/Daksitha/ReNeLib" target="_blank" class="btn">Code</a></div>
<div class="col-sm-3"><a href="#data">Data</a></div>
<div class="col-sm-3"><a href="#video-samples">Samples</a></div>
</div>


## Abstract
Flexible and natural nonverbal reactions to human behavior remain a challenge for socially interactive agents (SIAs) that are predominantly animated using hand-crafted rules. 
While recently proposed machine learning-based approaches to conversational behavior generation are a promising way to address this challenge, they have not yet been employed in SIAs. 
The primary reason for this is the lack of a software toolkit integrating such approaches with SIA frameworks that conforms to the challenging real-time requirements of human-agent interaction scenarios. 
In our work, we for the first time present such a toolkit consisting of three main components: (1) real-time feature extraction capturing multi-modal social cues from the user; (2) behavior generation based on a recent state-of-the-art neural network approach; (3) visualization of the generated behavior supporting both FLAME-based and Apple ARKit-based interactive agents.
We comprehensively evaluate the real-time performance of the whole framework and its components.
In addition, we release new pre-trained behavioral generation models based on real-life psychotherapy interactions that can be used to generate domain-specific listening behaviors.
Our software toolkit will be made fully publicly available and provide researchers with a valuable resource for deploying and evaluating novel listening behavior generation methods for SIAs in interactive real-time scenarios.

Watch the demo video:

<a href="https://youtu.be/I54lP-J0mtU" target="_blank">
 <img src="https://img.youtube.com/vi/I54lP-J0mtU/default.jpg" alt="Watch the video" width="240" height="180" border="10" />
</a>

## Installation
Docker container
docker run --rm -it pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
- $conda install -c fvcore -c iopath -c conda-forge fvcore iopath
- $conda install pytorch3d=0.7.0 -c pytorch3d
- $ipython
- $from pytorch3d.structures import Meshes

## Runtime setup
To Do

## Create your own dataset
To Do
