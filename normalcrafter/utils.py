from typing import Union, List
import tempfile
import numpy as np
import PIL.Image
import matplotlib.cm as cm
import mediapy
import torch
import cv2


def read_video_frames(video_path, process_length, target_fps, max_res):
    print("==> processing video: ", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("==> original video shape: ", (n_frames, original_height, original_width, 3))

    if max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)
    else:
        height = original_height
        width = original_width

    fps = src_fps if target_fps == -1 else target_fps
    stride = max(round(src_fps / fps), 1)
    frames_idx = list(range(0, n_frames, stride))
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(f"==> final processing shape: ({len(frames_idx)}, {height}, {width}, 3)")

    wanted = set(frames_idx)
    out = {}
    i = 0
    last_wanted = max(frames_idx) if frames_idx else -1
    while i <= last_wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()

    frames = [PIL.Image.fromarray(out[idx]) for idx in frames_idx if idx in out]
    return frames, fps

def save_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10,
    crf: int = 18,
) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    mediapy.write_video(output_video_path, video_frames, fps=fps, crf=crf)
    return output_video_path

def vis_sequence_normal(normals: np.ndarray):
    normals = normals.clip(-1., 1.)
    normals = normals * 0.5 + 0.5
    return normals
