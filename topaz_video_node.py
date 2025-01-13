import os
import numpy as np
import torch
import subprocess
import uuid
from PIL import Image
import tempfile
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('TopazVideoAI')

class TopazUpscaleParamsNode:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = ["auto", "aaa-9", "ahq-12", "alq-13", "alqs-2", "amq-13", "amqs-2", "ghq-5", "iris-2", "iris-3", "nyx-3", "prob-4", "thm-2", "rhea-1", "rxl-1"]
        return {
            "required": {
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "upscale_model": (upscale_models, {"default": "auto"}),
                "compression": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "previous_upscale": ("UPSCALE_PARAMS",),
            }
        }

    RETURN_TYPES = ("UPSCALE_PARAMS",)
    FUNCTION = "get_params"
    CATEGORY = "video"

    def get_params(self, upscale_factor=2.0, upscale_model="auto", compression=1.0, blend=0.0, previous_upscale=None):
        current_params = {
            "upscale_factor": upscale_factor,
            "upscale_model": upscale_model,
            "compression": compression,
            "blend": blend
        }
        
        if previous_upscale is None:
            return ([current_params],)
        else:
            return (previous_upscale + [current_params],)

class TopazVideoAINode:
    def __init__(self):
        self.base_temp_dir = tempfile.gettempdir()
        self.output_dir = os.path.join(self.base_temp_dir, "comfyui_topaz_temp")
        os.makedirs(self.output_dir, exist_ok=True)
        self.temp_files = []
        logger.debug(f"Initialized temp directory at: {self.output_dir}")
        if not CUPY_AVAILABLE:
            logger.warning("CuPy not available. Some GPU operations will be disabled.")

    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = ["auto", "aaa-9", "ahq-12", "alq-13", "alqs-2", "amq-13", "amqs-2", "ghq-5", "iris-2", "iris-3", "nyx-3", "prob-4", "thm-2", "rhea-1", "rxl-1"]
        return {
            "required": {
                "images": ("IMAGE",),
                "enable_upscale": ("BOOLEAN", {"default": False}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "upscale_model": (upscale_models, {"default": "auto"}),
                "compression": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "enable_interpolation": ("BOOLEAN", {"default": False}),
                "target_fps": ("INT", {"default": 60, "min": 1, "max": 240}),
                "interpolation_model": (["auto", "apo-8", "apf-1", "chr-2", "chf-3", "chr-2"], {"default": "auto"}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previous_upscale": ("UPSCALE_PARAMS",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_video"
    CATEGORY = "video"

    def _save_batch(self, frames_batch, frame_dir, start_idx):
        """Helper function to save a batch of frames"""
        frame_paths = []
        for i, frame in enumerate(frames_batch):
            frame_path = os.path.join(frame_dir, f"frame_{start_idx + i:05d}.png")
            img = Image.fromarray(frame)
            img.save(frame_path)
            frame_paths.append(frame_path)
        return frame_paths

    def _batch_to_video(self, image_batch, output_path, use_gpu):
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        if use_gpu and torch.cuda.is_available():
            frames = image_batch.to(device)
            frames = (frames * 255).byte()
            frames = frames.cpu().numpy()
        else:
            frames = image_batch.cpu().numpy()
            frames = (frames * 255).astype(np.uint8)
        
        frame_dir = os.path.join(self.output_dir, f"input_frames_{uuid.uuid4()}")
        os.makedirs(frame_dir, exist_ok=True)
        logger.debug(f"Created frame directory: {frame_dir}")
        
        try:
            batch_size = 32
            frame_paths = []
            
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, len(frames), batch_size):
                    batch = frames[i:i + batch_size]
                    futures.append(
                        executor.submit(self._save_batch, batch, frame_dir, i)
                    )
                
                for future in futures:
                    frame_paths.extend(future.result())
            
            logger.debug(f"Saved {len(frame_paths)} frames")
            
            if not frame_paths:
                raise ValueError("No frames were saved")
            
            cmd = [
                "ffmpeg", "-y",
                "-i", os.path.join(frame_dir, "frame_%05d.png"),
                "-c:v", "hevc_nvenc" if use_gpu else "mpeg4",
                "-q:v", "2",
                "-r", "30",
                output_path
            ]
            
            logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output video not created: {output_path}")
                
            logger.debug(f"Video created successfully at: {output_path}")
            
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def _build_filter_chain(self, enable_upscale, upscale_params, enable_interpolation, 
                          target_fps, interpolation_model):
        """构建FFmpeg滤镜链"""
        filters = []
        
        if enable_upscale:
            if upscale_params:
                for params in upscale_params:
                    if params["upscale_model"] == "auto":
                        filters.append(f"tvai_up=scale={params['upscale_factor']}")
                    else:
                        filters.append(
                            f"tvai_up=model={params['upscale_model']}"
                            f":scale={params['upscale_factor']}"
                            f":estimate=8"
                            f":compression={params['compression']}"
                            f":blend={params['blend']}"
                        )

        if enable_interpolation:
            if interpolation_model == "auto":
                filters.append(f"tvai_fi=fps={target_fps}")
            else:
                filters.append(f"tvai_fi=model={interpolation_model}:fps={target_fps}")
                
        return ','.join(filters) if filters else None

    def _video_to_batch(self, video_path, use_gpu):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        frame_dir = os.path.join(self.output_dir, f"output_frames_{uuid.uuid4()}")
        os.makedirs(frame_dir, exist_ok=True)
        logger.debug(f"Created output frame directory: {frame_dir}")
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vsync", "0",
                os.path.join(frame_dir, "frame_%05d.png")
            ]
            
            logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
            logger.debug(f"Found {len(frame_files)} output frames")
            
            if not frame_files:
                raise ValueError(f"No frames extracted from video: {video_path}")
            
            frames = []
            
            if use_gpu and CUPY_AVAILABLE:
                logger.debug("Using CuPy for frame processing")
                with cp.cuda.Device(0):
                    for frame_file in frame_files:
                        frame_path = os.path.join(frame_dir, frame_file)
                        img_np = np.array(Image.open(frame_path))
                        frame_gpu = cp.asarray(img_np)
                        frames.append(cp.asnumpy(frame_gpu))
            else:
                logger.debug("Using CPU for frame processing")
                for frame_file in frame_files:
                    frame_path = os.path.join(frame_dir, frame_file)
                    img = Image.open(frame_path)
                    frame = np.array(img)
                    frames.append(frame)
            
            frames_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
            logger.debug(f"Created tensor with shape: {frames_tensor.shape}")
            
            return frames_tensor
            
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def process_video(self, images, enable_upscale, upscale_factor, upscale_model, compression, blend,
                     enable_interpolation, target_fps, interpolation_model, use_gpu, previous_upscale=None):
        operation_id = str(uuid.uuid4())
        input_video = os.path.join(self.output_dir, f"{operation_id}_input.mp4")
        intermediate_video = os.path.join(self.output_dir, f"{operation_id}_intermediate.mp4")
        output_video = os.path.join(self.output_dir, f"{operation_id}_output.mp4")
        self.temp_files.extend([input_video, intermediate_video, output_video])
        
        try:
            logger.info("Converting image batch to video...")
            self._batch_to_video(images, input_video, use_gpu)
            
            current_input = input_video
            current_output = intermediate_video

            # 首先处理upscale
            if enable_upscale and (previous_upscale or upscale_model != "auto"):
                all_upscale_params = []
                if previous_upscale:
                    all_upscale_params.extend(previous_upscale)
                if enable_upscale:
                    all_upscale_params.append({
                        "upscale_factor": upscale_factor,
                        "upscale_model": upscale_model,
                        "compression": compression,
                        "blend": blend
                    })
                
                # 构建upscale滤镜链
                upscale_filters = []
                for params in all_upscale_params:
                    if params["upscale_model"] == "auto":
                        upscale_filters.append(f"tvai_up=scale={params['upscale_factor']}")
                    else:
                        upscale_filters.append(
                            f"tvai_up=model={params['upscale_model']}"
                            f":scale={params['upscale_factor']}"
                            f":estimate=8"
                            f":compression={params['compression']}"
                            f":blend={params['blend']}"
                        )
                
                if upscale_filters:
                    filter_chain = ','.join(upscale_filters)
                    logger.info(f"Applying upscale filter chain: {filter_chain}")
                    cmd = [
                        "ffmpeg", "-y",
                        "-hwaccel", "auto",
                        "-i", current_input,
                        "-vf", filter_chain,
                        "-c:v", "mpeg4",
                        "-q:v", "2",
                        current_output
                    ]
                    
                    logger.debug(f"Running FFmpeg upscale command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise RuntimeError(f"FFmpeg upscale error: {result.stderr}")
                    
                    current_input = current_output
                    current_output = output_video
            
            # 然后处理interpolation
            if enable_interpolation:
                logger.info(f"Applying interpolation with target fps {target_fps}")
                # 确保目标帧率是有效的正整数
                if target_fps <= 0:
                    raise ValueError("Target FPS must be greater than 0")
                
                # 构建interpolation命令
                interpolation_filter = (f"tvai_fi=model={interpolation_model}:fps={target_fps}"
                                     if interpolation_model != "auto"
                                     else f"tvai_fi=fps={target_fps}")
                
                cmd = [
                    "ffmpeg", "-y",
                    "-hwaccel", "auto",
                    "-i", current_input,
                    "-vf", interpolation_filter,
                    "-c:v", "mpeg4",
                    "-q:v", "2",
                    current_output
                ]
                
                logger.debug(f"Running FFmpeg interpolation command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg interpolation error: {result.stderr}")
            else:
                # 如果不需要interpolation，且当前输入不是最终输出，则复制到最终输出
                if current_input != output_video:
                    shutil.copy2(current_input, current_output)
            
            logger.info("Converting final video back to image batch...")
            output_frames = self._video_to_batch(current_output, use_gpu)
            
            return (output_frames,)
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

NODE_CLASS_MAPPINGS = {
    "TopazVideoAI": TopazVideoAINode,
    "TopazUpscaleParams": TopazUpscaleParamsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopazVideoAI": "Topaz Video AI (Upscale & Frame Interpolation)",
    "TopazUpscaleParams": "Topaz Upscale Parameters"
}
