import gc
import os

import numpy as np
import gradio as gr
import torch
from diffusers.training_utils import set_seed
from diffusers import AutoencoderKLTemporalDecoder

from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter

import uuid
import random
from huggingface_hub import hf_hub_download

from normalcrafter.utils import read_video_frames, vis_sequence_normal, save_video


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# stub @spaces.GPU decorator (HF-only) so app runs locally
def _gpu_noop(*dargs, **dkwargs):
    def deco(fn):
        return fn
    return deco

examples = [
    ["examples/example_01.mp4", 1024, -1, -1],
    ["examples/example_02.mp4", 1024, -1, -1],
    ["examples/example_03.mp4", 1024, -1, -1],
    ["examples/example_04.mp4", 1024, -1, -1],
    # ["examples/example_05.mp4", 1024, -1, -1],
    # ["examples/example_06.mp4", 1024, -1, -1],
]

pretrained_model_name_or_path = "Yanrui95/NormalCrafter"
DEVICE = _select_device()
weight_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
load_kwargs = {"variant": "fp16"} if DEVICE == "cuda" else {}
unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="unet",
    low_cpu_mem_usage=True,
)
vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae")

vae.to(dtype=weight_dtype)
unet.to(dtype=weight_dtype)

pipe = NormalCrafterPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    unet=unet,
    vae=vae,
    torch_dtype=weight_dtype,
    **load_kwargs,
)
pipe.to(DEVICE)
if DEVICE != "cuda":
    pipe.enable_attention_slicing()


@_gpu_noop(duration=120)
def infer_depth(
    video: str,
    max_res: int = 1024,
    process_length: int = -1,
    target_fps: int = -1,
    #
    save_folder: str = "./demo_output",
    window_size: int = 14,
    time_step_size: int = 10,
    decode_chunk_size: int = 7,
    seed: int = 42,
    save_npz: bool = False,
):
    set_seed(seed)
    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)

    frames, target_fps = read_video_frames(video, process_length, target_fps, max_res)

    # inference the depth map using the DepthCrafter pipeline
    with torch.inference_mode():
        res = pipe(
            frames,
            decode_chunk_size=decode_chunk_size,
            time_step_size=time_step_size,
            window_size=window_size,
        ).frames[0]
    
    # visualize the depth map and save the results
    vis = vis_sequence_normal(res)
    # save the depth map and visualization with the target FPS
    save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(video))[0])
    print(f"==> saving results to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_npz:
        np.savez_compressed(save_path + ".npz", normal=res)
    save_video(vis, save_path + "_vis.mp4", fps=target_fps)
    save_video(frames, save_path + "_input.mp4", fps=target_fps)

    # clear the cache for the next video
    gc.collect()
    _empty_cache()

    return [
        save_path + "_input.mp4",
        save_path + "_vis.mp4",

    ]


def construct_demo():
    with gr.Blocks(analytics_enabled=False) as depthcrafter_iface:
        gr.Markdown(
            """
            <div align='center'> <h1> NormalCrafter: Learning Temporally Consistent Video Normal from Video Diffusion Priors </span> </h1> \
                    <a style='font-size:18px;color: #000000'>If you find NormalCrafter useful, please help ⭐ the </a>\
                    <a style='font-size:18px;color: #FF5DB0' href='https://github.com/Binyr/NormalCrafter'>[Github Repo]</a>\
                    <a style='font-size:18px;color: #000000'>, which is important to Open-Source projects. Thanks!</a>\
                        <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/xxx'> [ArXiv] </a>\
                        <a style='font-size:18px;color: #000000' href='https://normalcrafter.github.io/'> [Project Page] </a> </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_video = gr.Video(label="Input Video")

            # with gr.Tab(label="Output"):
            with gr.Column(scale=2):
                with gr.Row(equal_height=True):
                    output_video_1 = gr.Video(
                        label="Preprocessed Video",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        show_share_button=True,
                        scale=5,
                    )
                    output_video_2 = gr.Video(
                        label="Generated Normal Video",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        show_share_button=True,
                        scale=5,
                    )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Row(equal_height=False):
                    with gr.Accordion("Advanced Settings", open=False):
                        max_res = gr.Slider(
                            label="Max Resolution",
                            minimum=512,
                            maximum=1024,
                            value=1024,
                            step=64,
                        )
                        process_length = gr.Slider(
                            label="Process Length",
                            minimum=-1,
                            maximum=280,
                            value=60,
                            step=1,
                        )
                        process_target_fps = gr.Slider(
                            label="Target FPS",
                            minimum=-1,
                            maximum=30,
                            value=15,
                            step=1,
                        )
                    generate_btn = gr.Button("Generate")
            with gr.Column(scale=2):
                pass

        gr.Examples(
            examples=examples,
            inputs=[
                input_video,
                max_res,
                process_length,
                process_target_fps,
            ],
            outputs=[output_video_1, output_video_2],
            fn=infer_depth,
            cache_examples="lazy",
        )
        # gr.Markdown(
        #     """
        #     <span style='font-size:18px;color: #E7CCCC'>Note: 
        #     For time quota consideration, we set the default parameters to be more efficient here,
        #     with a trade-off of shorter video length and slightly lower quality.
        #     You may adjust the parameters according to our 
        #     <a style='font-size:18px;color: #FF5DB0' href='https://github.com/Tencent/DepthCrafter'>[Github Repo]</a>
        #      for better results if you have enough time quota.
        #     </span>
        #     """
        # )

        generate_btn.click(
            fn=infer_depth,
            inputs=[
                input_video,
                max_res,
                process_length,
                process_target_fps,
            ],
            outputs=[output_video_1, output_video_2],
        )

    return depthcrafter_iface


if __name__ == "__main__":
    demo = construct_demo()
    demo.queue()
    # demo.launch(server_name="0.0.0.0", server_port=12345, debug=True, share=False)
    demo.launch(share=True)
