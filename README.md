# TalkSee tool
This repository contains the tool for generating real-time subtitles for videoconferencing tools or videos with one person on it. It will take the video stream from your screen and the audio stream from speakers, so make sure to open the needed window when generating subtitles. Otherwise the tool will always print that no faces are detected.

## Prerequisites
Install all needed dependencies with
```Shell
pip install -r requirements.txt
```

## Running the tool
To run the tool execute the following command
```Shell
python talksee.py
```

## Configuring 

Depending on the characteristics of your computer you might need to change the following parameters in the executed file:

- `REAL_FRAME_RATE` - how many fps to grab from your screen. Mss can reach the speed up to 30 but we don't recomend to use more than 15.

- `MULTIPLY_FRAMES` - creates as many copies of recently grabbed frame to simulate other fps for the model. Recommended settings 30/`REAL_FRAME_RATE`.

- `FRAMES_PER_CHUNK` - How many frames should be considered as one chunk of video. Smaller value will use less system resources, but compromise the model prediction quality.

We don't recommend changing any other parameters, as it might cause size issues during the model inference, but you are welcome to play with it.

## DEMO

This demo is outdated, so the executed file has a different name, but apart from that this is the expected behaviour of the tool.

## Acknowledgement

We used the [pytorch device avsr tutorial](https://github.com/pytorch/audio) for building this tool, changing the way to obtain audio and visual chunks and synchronization, as this can't be done with pytoch tooling.