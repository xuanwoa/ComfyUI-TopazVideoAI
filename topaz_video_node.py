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

class TopazVideoAINode:
    def __init__(self):
        self.base_temp_dir = tempfile.gettempdir()
        self.output_dir = os.path.join(self.base_temp_dir, "comfyui_topaz_temp")
        os.makedirs(self.output_dir, exist_ok=True)
        self.temp_files = []
        logger.debug(f"Initialized temp directory at: {self.output_dir}")
        if not CUPY_AVAILABLE:
            logger.warning("CuPy not available. Some GPU operations will be disabled.")

    def __del__(self):
        for file in self.temp_files:
            if os.path.exists(file):
                os.remove(file)
                logger.debug(f"Cleaned up temporary file: {file}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enable_upscale": ("BOOLEAN", {"default": False}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "upscale_model": (["auto", "aaa-9", "ahq-12", "alq-13", "alqs-2", "amq-13", "amqs-2", "ghq-5", "iris-3", "nyx-3", "prob-4", "thm-2", "rhea-1", "rxl-1", "thm-2"], {"default": "auto"}),
                "enable_interpolation": ("BOOLEAN", {"default": False}),
                "target_fps": ("INT", {"default": 60, "min": 1, "max": 240}),
                "interpolation_model": (["auto", "apo-8", "apf-1", "chr-2", "chf-3", "chr-2"], {"default": "auto"}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_video"
    CATEGORY = "video"

    def process_video(self, images, enable_upscale, upscale_factor, upscale_model, 
                     enable_interpolation, target_fps, interpolation_model, use_gpu):
        operation_id = str(uuid.uuid4())
        base_video = os.path.join(self.output_dir, f"{operation_id}_input.mp4")
        upscaled_video = os.path.join(self.output_dir, f"{operation_id}_upscaled.mp4")
        final_video = os.path.join(self.output_dir, f"{operation_id}_final.mp4")
        
        self.temp_files.extend([base_video, upscaled_video, final_video])
        
        try:
            logger.info("Converting image batch to video...")
            self._batch_to_video(images, base_video, use_gpu)
            
            current_input = base_video
            current_output = upscaled_video if enable_upscale else final_video

            if enable_upscale:
                logger.info(f"Applying upscale filter with factor {upscale_factor}...")
                self._apply_upscale(current_input, current_output, upscale_factor, upscale_model)
                current_input = current_output
                current_output = final_video

            if enable_interpolation:
                logger.info(f"Applying frame interpolation to {target_fps} fps...")
                self._apply_interpolation(current_input, current_output, target_fps, interpolation_model)
                current_input = current_output

            logger.info("Converting final video back to image batch...")
            output_frames = self._video_to_batch(current_input, use_gpu)
            
            return (output_frames,)
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    def _save_batch(self, frames_batch, frame_dir, start_idx):
        """Helper function to save a batch of frames"""
        for i, frame in enumerate(frames_batch):
            frame_path = os.path.join(frame_dir, f"frame_{start_idx + i:05d}.png")
            img = Image.fromarray(frame)
            img.save(frame_path)
        return [os.path.join(frame_dir, f"frame_{start_idx + i:05d}.png") for i in range(len(frames_batch))]

    def _batch_to_video(self, image_batch, output_path, use_gpu):
        # Only use GPU for initial processing if GPU acceleration is enabled
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
            # Process frames in parallel using ThreadPoolExecutor
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

    def _apply_upscale(self, input_path, output_path, scale_factor, model):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
            
        vf_param = f"tvai_up=scale={scale_factor}"
        if model != "auto":
            vf_param = f"tvai_up=model={model}:scale={scale_factor}"
        
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "auto",
            "-i", input_path,
            "-vf", vf_param,
            "-c:v", "mpeg4",  # Always use mpeg4 for upscale
            "-q:v", "2",
            output_path
        ]
        
        logger.debug(f"Running upscale command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Upscale error: {result.stderr}")
        
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Upscaled video not created: {output_path}")
        
        logger.debug(f"Upscale completed: {output_path}")

    def _apply_interpolation(self, input_path, output_path, target_fps, model):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
            
        vf_param = f"tvai_fi=fps={target_fps}"
        if model != "auto":
            vf_param = f"tvai_fi=model={model}:fps={target_fps}"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", vf_param,
            "-c:v", "mpeg4",  # Always use mpeg4 for interpolation
            "-q:v", "2",
            output_path
        ]
        
        logger.debug(f"Running interpolation command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Interpolation error: {result.stderr}")
            
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Interpolated video not created: {output_path}")
        
        logger.debug(f"Interpolation completed: {output_path}")

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
            
            # Use CuPy for GPU acceleration if available and enabled
            if use_gpu and CUPY_AVAILABLE:
                logger.debug("Using CuPy for frame processing")
                with cp.cuda.Device(0):
                    for frame_file in frame_files:
                        frame_path = os.path.join(frame_dir, frame_file)
                        # Load image into CPU numpy array first
                        img_np = np.array(Image.open(frame_path))
                        # Transfer to GPU
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

NODE_CLASS_MAPPINGS = {
    "TopazVideoAI": TopazVideoAINode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopazVideoAI": "Topaz Video AI (Upscale & Frame Interpolation)"
}
