# CubeMove AI

CubeMove AI is a computer vision and machine learning project that recognizes Rubik's Cube moves from video. The goal is to analyze the movements of a person solving a Rubik's Cube and convert them into standard move notation.

The current version uses hand movement as the main source of information. MediaPipe is used to track the hands in each frame, and a neural network analyzes the movement over time to predict which move was performed.

## What It Does

CubeMove AI recognizes the following 12 moves:

* R
* R'
* L
* L'
* U
* U'
* D
* D'
* F
* F'
* B
* B'

The current version focuses on 90-degree face turns. It does not recognize 180-degree turns such as `R2` or cube rotations such as `x`, `y`, and `z`.

## How It Works

The system processes each video as a sequence rather than treating every frame as a separate prediction.

First, OpenCV reads the video and extracts individual frames. MediaPipe then detects the hands in each frame. Each hand is represented using 21 landmarks, including points for the wrist and finger joints.

The landmark positions are normalized so that the model is less dependent on the hands location in the camera frame. The system also calculates changes between consecutive frames, which gives the model information about how the hands are moving.

The resulting sequence is trimmed to focus on the movement and then resampled to a fixed length of 32 frames. This allows videos with different lengths to be passed into the same neural network.

The processed sequence is then passed into a bidirectional GRU (Gated Recurrent Unit) model. The model analyzes the movement over time and produces a prediction for one of the 12 move classes.

The overall process is:

```text
Video -> OpenCV -> MediaPipe Hand Landmarks -> Feature Normalization -> Motion Features -> Sequence Processing -> Bidirectional GRU -> Move Prediction
```

## Dataset

I created a custom dataset specifically for this project.

Each video contains one Rubik's Cube move and is labeled according to the move performed. The dataset contains 360 clips in total, with 30 clips for each of the 12 move classes.

The dataset is organized into folders by move:

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

Prime moves are represented with `p` in the dataset. For example, `Rp` represents `R'` (the same move in the counter-clockwise direction).

## Model

The move classifier is a bidirectional GRU neural network implemented with PyTorch.

The model processes the sequence in both directions and uses temporal pooling before the final classification layers. Dropout and normalization are also used during training.

## Technologies

* Python
* OpenCV
* MediaPipe
* PyTorch
* NumPy

OpenCV handles video and webcam processing. MediaPipe provides the hand landmark data used by the model. PyTorch is used to train and run the neural network, and NumPy is used for numerical processing.

## Limitations

The current system primarily relies on hand movement. This means some moves can be more difficult to distinguish when the hands move in unusual ways or when the cube is partially obscured.

Moves such as B, D, and L can also be more difficult to recognize from a single camera because they are less visible from the camera's perspective.

The current model does not directly use cube sticker colors to make its predictions.

## Future Improvements

There are several directions I would like to explore in future versions:

*** Use cube sticker color changes as an additional source of information.
*** Automatically detect the Rubik's Cube in the video.
*** Train using more people, camera angles, and lighting conditions.
* Apply perspective rectification to make the cube view more consistent.
* Improve recognition of less-visible moves.
* Add support for 180-degree turns.
* Add support for cube rotations.
* Improve predictions during live use.
* Recognize complete solve sequences.
* Analyze and compare complete Rubik's Cube solves.

## Related Work

This project was developed after researching existing work involving Rubik's Cube move detection and computer vision.

* [Rubik's Cube Move Detection](https://github.com/felikemath/Rubik-s-Cube-Move-Detection)
* [MagicCube](https://github.com/trincaog/magiccube)

## Author
Amit Kamal Mudududla
