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

The landmarks are normalized so that the model is less dependent on the position and size of the hands in the camera frame. The program also calculates the changes between consecutive frames to give the model information about hand movement.

The movement is then processed into a fixed-length sequence of 32 frames. This allows every video to be given to the neural network in the same format.

The processed sequence is passed to a bidirectional GRU (Gated Recurrent Unit) model built with PyTorch. The model analyzes the movement over time and predicts which of the 12 moves was performed.

The current pipeline is:

```text
Video -> OpenCV -> MediaPipe Hand Landmarks -> Feature Normalization -> Motion Features -> Sequence Processing -> Bidirectional GRU -> Move Prediction
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

## How to Run It

### Requirements

You will need:

* Python 3.10 or newer
* A working webcam
* A computer capable of running PyTorch
* The Python packages used by the project

Install the required packages with:

```bash
pip install opencv-python mediapipe torch numpy
```

### Recording Your Own Dataset

If you want to create your own dataset, run:

```bash
python record_dataset.py
```

The program opens the webcam and allows you to select which Rubik's Cube move you are recording. It records individual clips and saves them into the corresponding dataset folder.

Each clip should contain one move.

The dataset recorder is currently configured around the dataset used for this project, so you may need to change the paths or recording settings if you want to build a different dataset.

### Training

After creating the dataset, the training program can be used to extract the hand landmark features and train the classification model.

The training process includes:

* Training and validation splitting
* Hand landmark normalization
* Motion feature extraction
* Sequence resampling
* Data augmentation
* Class weighting
* Early stopping
* Gradient clipping
* GPU support when available

The trained model can then be used by the main CubeMove AI program.

### Running CubeMove AI

Connect a webcam and run:

```bash
python cube_move_ai.py
```

The program processes the webcam input, tracks the hands, and uses the trained model to predict the Rubik's Cube move.

## Results

The project successfully produced a custom dataset of 360 labeled video clips and a trained neural network capable of classifying the 12 supported Rubik's Cube moves.

The project is still being improved, so the current model should be considered a working prototype rather than a finished speedcubing analysis system.

## Limitations

The current model primarily relies on hand movement. Because of this, some moves are more difficult to distinguish than others.

Moves such as B, D, and L can be harder to recognize from a single camera because they are less visible from the camera's perspective. Performance can also change with different lighting, camera positions, or people with different hand movements.

The current model does not directly use cube sticker colors. If the cube sticker colors were taken into consideration, an issue would arise for people with different skin tones or fingernail colors. 

## Future Improvements

There are several improvements I would like to make:

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
├── dataset/
└── README.md
```

The dataset may not be included in the repository because of the size of the video files.

## Author

Amit Kamal Mudududla

CubeMove AI is a personal project combining computer vision, hand tracking, and deep learning to recognize Rubik's Cube moves.
