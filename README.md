# Preprocessing Demo
## Introduction
This branch contains a code allowing to get the frontal view on the speakers lips, which is a crucial step in the model training pipeline.  

## Prerequisites
After cloning repository you need to navigate to its folder and install the requirements:
```Shell
pip install -r requirements.txt
```

## Running preprocessing
`preprocess.py` allows to run the preprocessing pipeline on users video and save the result to a new file.
```Shell
python preprocess.py input_path=[input_path] \
               output_path=[output_path] 
```

<details open>
  <summary><strong>Arguments</strong></summary>

  - `input_path`: Path to video, which needs to be preprocessed (REQUIRED)
  - `output_path`: Path to which output video. Default will be the same as `input_path` with the `_out` at the end of a filename.
</details>

## Example output
As we use this preprocessing dtep further while training the model on LRS3 dataset, we provide demonstration of the transformation on one of the videos from this dataset. Left side is a video before preprocessing and the right shows the input the model will receive during the training.


https://github.com/sasha-tsepilova/lipreading_enhancment/assets/75687119/63ec47a7-6723-4889-b982-cb2201e1b66d




# Acknowledgement
The code is based on [mediapipeDemos](https://github.com/Rassibassi/mediapipeDemos) and is using predeveloped [face_geometry.py](./mediapipeDemos/custom/face_geometry.py) to get canonical face mesh and project the points from it to the image.
