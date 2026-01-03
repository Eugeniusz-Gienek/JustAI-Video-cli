# JustAI-Video-cli
Video optimizations using AI (e.g. frame interpolation)

## What is this thing?
This is an attempt to optimize videos using local GPUs - CLI script.
Current features included:
* Frames interpolation using [Google FILM frames interpolation method](https://github.com/google-research/frame-interpolation).
  Basically it performed the interpolation frame-by-frame, not using the video-optimized models but rather image-optimized ones. There are multiple reasons for that, however in short this is due to quality - this way (at least according to my subjective judging) the output quality is way better than other methods.
  What also stands out in this frame interpolation interpretation - You can interpolate not only "integer-times" the frames rate but You can also convert, saying, 24 frames to 60, or even 23.987....whatever (float number I mean) to a whole frame rate number.

# Known issues and current limitations
* This is a single-machine one-GPU cli script, multiple GPUs on a single machine are not supported (multi-machines distribution will be implemented in another project, this CLI version is aimed to run on a single machine)
* It takes a LONG time to interpolate frames, especially with some frame multiplication coefficients - current version is using a bit suboptimal from performance perspective (but good in terms of quality) method of generating frames if the multiplication is not "integer" - e.g. if that's not 30-to-60 FPS but, saying, 24 to 60 - it will actually generate 120FPS video and will remove every second frame. Frames removal is kinda suboptimal, it would be better to not generate them in the first place and there are already thoughts on how to do that - will be implemented in future releases.
* Installation instructions are not there yet - in TODO list
* Project was tested and run at the moment of writing this manual ONLY on Linux and ONLY using NVidia GPU (RTX 3090 in particular). AMD, Intel and others (?) are not yet tested.

## What is needed to run this (in principle) - software
* python3.13 (best option) or 3.12 (other versions untested)
* cython
* ffmpeg
* opencv (the "packaged" opencv is limited and should be re-compiled with proper FFMPEG enabled - take a look at file [requirements_opencv.txt](https://github.com/Eugeniusz-Gienek/JustAI-Video-cli/blob/main/requirements_opencv.txt))
* Patience. Your patience.

## Installation

The assumption is that You are using some Linux distibution (tested on Gentoo) and the NVidia GPU - these installers are build with these facts in mind.
Currently there are two installers - manual and Docker. Manual installs the tool directly in the system (python virtualenv isolates all the python packages though and resolves dependencies), Docker installs the script inside, well, Docker.

### Manual installation
`./install_manual.sh`

### Docker installation
In order to run NVidia Docker image You'd need to install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first.
Installation also requires Docker itself, Docker buildx and nvidia drivers installed.
`./install_docker.sh`

## Remark
Due to [GitHub size limitations](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github#distributing-large-binaries) the model file [film_net_fp32.pt](https://github.com/Eugeniusz-Gienek/JustAI-Video-cli/releases/download/0.1-pre-alpha/film_net_fp32.pt) has to be downloaded separately and put into `film_models` folder.
