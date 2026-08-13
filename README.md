# CubeMove AI

CubeMove AI is a machine learning project that recognizes Rubik's Cube moves from video. I built it to explore whether hand movement could be used to automatically convert a person's physical cube turns into standard Rubik's Cube notation.

I have always been interested in both programming and Rubik's Cubes, so I wanted to combine the two into a project that involved more than simply building a standard cube solver. The main challenge is that a solver's hands can move quickly and often cover parts of the cube, making it difficult to determine the exact move from a video.

The current version uses hand movement as the primary source of information. MediaPipe detects the hands in each frame, and a neural network analyzes the movement over time to classify the move.

## What It Recognizes

The model currently recognizes 12 clockwise and counterclockwise 90-degree face turns:

```text
R, R', L, L', U, U', D, D', F, F', B, B'
```

In the dataset and code, prime moves are represented using `p`. For example, `Rp` represents `R'`.

The current version does not recognize 180-degree turns such as `R2` or cube rotations such as `x`, `y`, and `z`.

## How It Works

The system treats a move as a sequence of frames instead of making a prediction from one frame at a time.

First, OpenCV reads the video. MediaPipe then detects the hands in each frame. Each hand is represented using 21 landmarks corresponding to points such as the wrist and finger joints.

The landmarks are normalized so that the model is less dependent on the position and size of the hands in the camera frame. The program also calculates changes between consecutive frames to give the model information about hand movement.

The movement is then processed into a fixed-length sequence of 32 frames. This allows every video to be given to the neural network in the same format.

The processed sequence is passed to a bidirectional GRU (Gated Recurrent Unit) model built with PyTorch. The model analyzes the movement over time and predicts which of the 12 moves was performed.

The current pipeline is:

```text
Video
  ↓
OpenCV
  ↓
MediaPipe Hand Landmarks
  ↓
Feature Normalization
  ↓
Motion Features
  ↓
Sequence Processing
  ↓
Bidirectional GRU
  ↓
Move Prediction
```

## Dataset

I created my own dataset specifically for this project instead of relying entirely on an existing dataset.

The dataset contains 360 video clips:

* 12 move classes
* 30 clips per move
* One move per video clip

The dataset is organized into folders:

```text
dataset/
├── R/
├── Rp/
├── L/
├── Lp/
├── U/
├── Up/
├── D/
├── Dp/
├── F/
├── Fp/
├── B/
└── Bp/
```

I created a separate recording program to make the dataset. It allows me to select a move, record multiple clips, and automatically save each clip into the correct folder with the appropriate filename.

The dataset itself is not required to run the pretrained model, so the video files do not need to be included in the repository.

## Installation

### Requirements

You will need:

* Python 3.10 or newer
* A working webcam
* A computer capable of running PyTorch

The project uses:

* OpenCV
* MediaPipe
* PyTorch
* NumPy

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Project

### Using the Pretrained Model

The repository includes the trained model:

```text
cube_move_ai_model.pth
```

Make sure this file is in the same directory as `cube_move_ai.py`.

Connect a webcam and run:

```bash
python cube_move_ai.py live
```

The program will open the webcam, detect hand landmarks, record a movement sequence, and use the trained model to predict the Rubik's Cube move.

### Controls

* `Space` - Start or stop recording a move
* `C` - Clear the move history
* `R` - Reset the move history
* `Z` - Undo the most recent move
* `Q` - Quit

The program also displays the most recent prediction, its confidence, recent moves, and the model's top predictions.

## Creating Your Own Dataset

To create a new dataset, run:

```bash
python record_dataset.py
```

The recording program allows you to select a move and automatically save individual video clips into the corresponding dataset folder.

Each video should contain one Rubik's Cube move.

The recording program was designed specifically for the dataset used in this project, so some settings may need to be changed when creating a different dataset.

## Training

The same `cube_move_ai.py` program contains the training pipeline.

To train a new model using the dataset, run:

```bash
python cube_move_ai.py train
```

The best-performing model checkpoint is saved as:

```text
cube_move_ai_model.pth
```

## Checking the Dataset and Model

The program also includes a quick diagnostic command:

```bash
python cube_move_ai.py check
```

This can be used to inspect the available dataset clips, check whether MediaPipe is available, and verify the model checkpoint.

## Results

The project produced:

* A custom dataset containing 360 labeled video clips
* 12 move classes
* A trained bidirectional GRU neural network
* A webcam-based move prediction system

The current model is a working prototype and is still being improved. Its performance can vary depending on the camera position, lighting, and the way a person performs a move.

## Limitations

The current model primarily relies on hand movement. Because of this, some moves are more difficult to distinguish than others.

Moves such as `B`, `D`, and `L` can be harder to recognize from a single camera because they are less visible from the camera's perspective. Performance can also change with different lighting, camera positions, and individual hand movement patterns.

The current model does not directly use cube sticker colors.

## Future Improvements

There are several directions I would like to explore:

* Use cube sticker color changes as an additional source of information.
* Automatically locate the Rubik's Cube in the camera frame.
* Apply perspective rectification to standardize the cube's appearance.
* Improve recognition of less-visible moves.
* Train with more people, camera angles, and lighting conditions.
* Add support for 180-degree turns.
* Add support for cube rotations.
* Improve real-time prediction stability.
* Recognize complete solve sequences.
* Analyze complete speedcubing solves.

## Related Work

I researched existing Rubik's Cube move detection projects while developing CubeMove AI.

* [Rubik's Cube Move Detection](https://github.com/felikemath/Rubik-s-Cube-Move-Detection)
* [MagicCube](https://github.com/trincaog/magiccube)

These projects helped me understand existing approaches to Rubik's Cube move detection and cube manipulation.

## Project Structure

```text
Cube_Move_AI/
├── cube_move_ai.py
├── record_dataset.py
├── cube_move_ai_model.pth
├── requirements.txt
├── .gitignore
├── README.md
└── demo.png
```

The dataset and cached MediaPipe features are not included because they are generated locally and can take up a significant amount of storage.

## Author

Amit Kamal Mudududla
