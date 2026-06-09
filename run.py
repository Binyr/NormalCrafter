import gc
import os
import numpy as np
import torch

from diffusers.training_utils import set_seed
from diffusers import AutoencoderKLTemporalDecoder
from fire import Fire

from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
from normalcrafter.utils import vis_sequence_normal, save_video, read_video_frames


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _empty_cache():
    if torch.cuda.is_available():
        _empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = None,
    ):
        device = _select_device()
        # fp16 unstable on MPS for SVD; use fp32 off-CUDA
        weight_dtype = torch.float16 if device == "cuda" else torch.float32
        load_kwargs = {"variant": "fp16"} if device == "cuda" else {}

        unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
            unet_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            unet_path, subfolder="vae"
        )
        vae.to(dtype=weight_dtype)
        unet.to(dtype=weight_dtype)
        # load weights of other components from the provided checkpoint
        self.pipe = NormalCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            vae=vae,
            torch_dtype=weight_dtype,
            **load_kwargs,
        )

        # cpu_offload uses accelerate hooks that require CUDA — skip off-CUDA
        if cpu_offload is not None and device == "cuda":
            if cpu_offload == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to(device)
        # xformers CUDA-only; attention slicing helps on MPS
        if device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(e)
                print("Xformers is not enabled")
        else:
            self.pipe.enable_attention_slicing()

    def infer(
        self,
        video: str,
        save_folder: str = "./demo_output",
        window_size: int = 14,
        time_step_size: int = 10,
        process_length: int = 195,
        decode_chunk_size: int = 7,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = 15,
        seed: int = 42,
        save_npz: bool = False,
    ):
        set_seed(seed)

        frames, target_fps = read_video_frames(
            video,
            process_length,
            target_fps,
            max_res,
        )
        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                decode_chunk_size=decode_chunk_size,
                time_step_size=time_step_size,
                window_size=window_size,
            ).frames[0]
        # visualize the depth map and save the results
        vis = vis_sequence_normal(res)
        # save the depth map and visualization with the target FPS
        save_path = os.path.join(
            save_folder, os.path.splitext(os.path.basename(video))[0]
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_video(vis, save_path + "_vis.mp4", fps=target_fps)
        save_video(frames, save_path + "_input.mp4", fps=target_fps)
        if save_npz:
            np.savez_compressed(save_path + ".npz", depth=res)

        return [
            save_path + "_input.mp4",
            save_path + "_vis.mp4",
        ]

    def run(
        self,
        input_video,
        num_denoising_steps,
        guidance_scale,
        max_res=1024,
        process_length=195,
    ):
        res_path = self.infer(
            input_video,
            num_denoising_steps,
            guidance_scale,
            max_res=max_res,
            process_length=process_length,
        )
        # clear the cache for the next video
        gc.collect()
        _empty_cache()
        return res_path[:2]


def main(
    video_path: str,
    save_folder: str = "./demo_output",
    unet_path: str = "Yanrui95/NormalCrafter",
    pre_train_path: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    process_length: int = -1,
    cpu_offload: str = None,
    target_fps: int = -1,
    seed: int = 42,
    window_size: int = 14,
    time_step_size: int = 10,
    max_res: int = 1024,
    dataset: str = "open",
    save_npz: bool = False
):
    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_train_path=pre_train_path,
        cpu_offload=cpu_offload,
    )
    # process the videos, the video paths are separated by comma
    video_paths = video_path.split(",")
    for video in video_paths:
        depthcrafter_demo.infer(
            video,
            save_folder=save_folder,
            window_size=window_size,
            process_length=process_length,
            time_step_size=time_step_size,
            max_res=max_res,
            dataset=dataset,
            target_fps=target_fps,
            seed=seed,
            save_npz=save_npz,
        )
        # clear the cache for the next video
        gc.collect()
        _empty_cache()


if __name__ == "__main__":
    # running configs
    # the most important arguments for memory saving are `cpu_offload`, `enable_xformers`, `max_res`, and `window_size`
    # the most important arguments for trade-off between quality and speed are
    # `num_inference_steps`, `guidance_scale`, and `max_res`
    Fire(main)
