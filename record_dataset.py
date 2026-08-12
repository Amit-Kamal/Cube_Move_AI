from pathlib import Path
import time
import cv2

project = "Cube_Motion"
dataset_folder = "dataset"
move_names = ["R", "L", "U", "D", "F", "B", "Rp", "Lp", "Up", "Dp", "Fp", "Bp"]

camera_width = 1280
camera_height = 720
target_fps = 30

video_extension = ".mp4"
video_codec = "mp4v"  
count_down = 3
clip_length = 2
break_length = 3
clips_per_move = 30
window_name = "Cube Motion Recorder"

number_keys_to_moves = {
    ord("1"): "R",
    ord("2"): "L",
    ord("3"): "U",
    ord("4"): "D",
    ord("5"): "F",
    ord("6"): "B",
    ord("7"): "Rp",
    ord("8"): "Lp",
    ord("9"): "Up",
    ord("0"): "Dp",
    ord("-"): "Fp",
    ord("="): "Bp"
}

letter_keys_to_moves = {
    ord("r"): "R",
    ord("l"): "L", 
    ord("u"): "U",
    ord("d"): "D",
    ord("f"): "F",
    ord("b"): "B",
    ord("R"): "Rp",
    ord("L"): "Lp",
    ord("U"): "Up",
    ord("D"): "Dp",
    ord("F"): "Fp",
    ord("B"): "Bp"
}
 
control_text = ("Select move: 1-0,-,= or letters r/R l/L u/U d/D f/F b/B")

def create_dataset_folder(dataset_folder, move_names):
    dataset_folder.mkdir(parents=True, exist_ok=True)
    for move in move_names:
        move_folder = dataset_folder / move
        move_folder.mkdir(parents=True, exist_ok=True)

def get_video_number(file_path, move_name, extension):
    if file_path.suffix != extension:
        return None
    part = file_path.stem.split("_")
    if len(part) != 2:
        return None
    name_in_file, number_text = part
    if name_in_file != move_name:
        return None
    if not number_text.isdigit():
        return None
    return int(number_text)

def list_saved_video_number(move_folder, move_name, extension):
    video_numbers = []
    for file_path in move_folder.iterdir():
        if not file_path.is_file():
            continue
        video_number = get_video_number(file_path, move_name, extension)
        if video_number is not None:
            video_numbers.append(video_number)
    return video_numbers

def count_saved_videos(move_folder, move_name, extension):
    return len(list_saved_video_number(move_folder, move_name, extension))

def get_next_video_path(move_folder, move_name, extension):
    saved_video_numbers = list_saved_video_number(move_folder, move_name, extension)
    if saved_video_numbers:
        next_video_number = max(saved_video_numbers) + 1
    else:
        next_video_number = 1
    file_name = f"{move_name}_{next_video_number}{extension}"
    return move_folder / file_name

def draw_status(frame, selected_move, is_recording, saved_count, session_message = ""):
    if is_recording:
        status_text = "Recording..."
        status_color = (0, 0, 255)  # Red
    else:
        status_text = "Idle"
        status_color = (0, 255, 0)  # Green
    lines = [
        f"{project} | Move: {selected_move} | Status : {status_text}",
        f"Clips for {selected_move}: {saved_count}",
        session_message,
        "Controls:",
        f"- {control_text}",
        f"- Space: start {clips_per_move}-clips session for selected move",
        "-q : quit"
    ]
    y_position = 30
    for line in lines:
        if not line:
            continue
        if y_position == 30:
            text_color = status_color
        else:
            text_color = (255, 255, 255)  # White
        cv2.putText(
            frame,
            line,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
            cv2.LINE_AA
        )
        y_position += 28

def show_countdown(camera, seconds, selected_move, saved_count, message): 
    for seconds_left in range(seconds, 0, -1):
        countdown_start = time.time()
        while time.time() - countdown_start < 1:
            frame_ok, frame = camera.read()
            if not frame_ok:
                continue
            draw_status(frame, selected_move, False, saved_count, message)
            # Used AI help to figure out how to center the countdown text on the frame
            cv2.putText(
                frame,
                str(seconds_left),
                (frame.shape[1] // 2 - 20, frame.shape[0] // 2 + 20), 
                cv2.FONT_HERSHEY_DUPLEX,
                3.0,
                (0, 255, 255), # Yellow
                5,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                message,
                (frame.shape[1] // 2 - 200, frame.shape[0] // 2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255), # Yellow
                2,
                cv2.LINE_AA
            )
            cv2.imshow(window_name, frame)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord("q"):
                return False
    return True

def create_video_writer(video_path, width, height, fps):
    codec = cv2.VideoWriter_fourcc(*video_codec)
    return cv2.VideoWriter(str(video_path), codec, fps, (width, height))

def stop_recording(video_writer, saved_video_path, saved_count, recorded_move):
    if video_writer is not None:
        video_writer.release()
    if saved_video_path is not None and recorded_move is not None:
        saved_count[recorded_move] += 1
        print(f"Saved: {saved_video_path}")

def get_selected_move_from_key(key_pressed, current_selected_move):
    if key_pressed in number_keys_to_moves:
        return number_keys_to_moves[key_pressed]
    if key_pressed in letter_keys_to_moves:
        return letter_keys_to_moves[key_pressed]
    return current_selected_move

def start_recording_clip(dataset_folder, move_name, frame_width, frame_height, fps):
    move_folder = dataset_folder / move_name
    next_video_path = get_next_video_path(move_folder, move_name, video_extension)
    video_writer = create_video_writer(next_video_path, frame_width, frame_height, fps)
    if not video_writer.isOpened():
        print(f"Error: Could not create video writer for {next_video_path}")
        return None, None
    print(f"Recording clip for move {move_name} to {next_video_path}")
    return video_writer, next_video_path

def main():
    dataset_path = Path(dataset_folder)
    create_dataset_folder(dataset_path, move_names)
    saved_count = {}
    for move_name in move_names:
        move_folder = dataset_path / move_name
        saved_count[move_name] = count_saved_videos(move_folder, move_name, video_extension)
    selected_move = "R"
    is_recording = False
    video_writer = None
    current_video_path = None
    current_recording_move = None
    session_is_active = False
    session_move = None
    session_clip_count = 0
    break_end_time = 0
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    camera.set(cv2.CAP_PROP_FPS, target_fps)
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 1:
        actual_fps = float(target_fps)
    print(f"Webcam Opened: {frame_width}x{frame_height} at {actual_fps:.2f} FPS")
    print("Press q in window to quit.")
    print(control_text)
    try:
        while True:
            frame_ok, frame = camera.read()
            if not frame_ok:
                print("Error: Could not read frame from camera.")
                continue
            current_time = time.time()
            if is_recording and video_writer is not None:
                video_writer.write(frame)
                recording_time = current_time - recording_start_time
                if recording_time >= clip_length:
                    is_recording = False
                    stop_recording(video_writer, current_video_path, saved_count, current_recording_move)
                    video_writer = None
                    current_video_path = None
                    current_recording_move = None
                    session_clip_count += 1
                    if session_clip_count >= clips_per_move:
                        print(f"Session for move {session_move} completed.")
                        session_is_active = False
                        session_move = None
                        break_end_time = None
                    else:
                        break_end_time = current_time + break_length
            if session_is_active and not is_recording and break_end_time is not None:
                if current_time >= break_end_time:
                    video_writer, current_video_path = start_recording_clip(
                        dataset_path,
                        session_move,
                        frame_width,
                        frame_height,
                        actual_fps
                    )
                    if video_writer is not None:
                        session_is_active = True
                        break_end_time = None
                    else:
                        print(f"Error: Could not start the next clip for move {session_move}.")
                        session_is_active = False
                        session_move = None
                        break_end_time = None
            session_message = ""
            if session_is_active:
                if break_end_time:
                    session_message = f"Session for {session_move}: {session_clip_count}/{clips_per_move} clips recorded. Break time: {int(break_end_time - current_time)}s"
                else:
                    session_message = f"Session for {session_move}: {session_clip_count}/{clips_per_move} clips recorded. Recording next clip..."
            draw_status(frame, selected_move, is_recording, saved_count[selected_move], session_message)
            cv2.imshow(window_name, frame)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord("q"):
                break
            if session_is_active:
                continue
            new_move = get_selected_move_from_key(key_pressed, selected_move)
            if new_move != selected_move:
                selected_move = new_move
                continue
            if key_pressed == ord(" "):
                should_continue = show_countdown(
                    camera,
                    count_down,
                    selected_move,
                    saved_count[selected_move],
                    session_message
                )
                if not should_continue:
                    break
            video_writer, current_video_path = start_recording_clip(
                dataset_path,
                selected_move,
                frame_width,
                frame_height,
                actual_fps
            )
            if video_writer is None:
                session_is_active = False
                continue
            session_is_active = True
            session_move = selected_move
            session_clip_count = 0
            break_end_time = None
            current_recording_move = selected_move
            recording_start_time = time.time()
            is_recording = True
            print(f"Session started for move {session_move}. Target: {clips_per_move} clips.")
    finally:
        if video_writer is not None:
            stop_recording(
                video_writer,
                current_video_path,
                saved_count,
                current_recording_move
            )
        camera.release()
        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    main()