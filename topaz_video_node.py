# topaz_video_node.py

import os
import numpy as np
import torch
import subprocess
import uuid
from PIL import Image
import tempfile
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TopazVideoAI')

class TopazVideoAINode:
    """
    ComfyUI node that applies Topaz Video AI filters (upscale and frame interpolation)
    """
    
    def __init__(self):
        self.output_dir = os.path.join(tempfile.gettempdir(), "comfyui_topaz_temp")
        os.makedirs(self.output_dir, exist_ok=True)
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enable_upscale": ("BOOLEAN", {"default": False}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "enable_interpolation": ("BOOLEAN", {"default": False}),
                "target_fps": ("INT", {"default": 60, "min": 1, "max": 240}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_video"
    CATEGORY = "video"
    
    def process_video(self, images, enable_upscale, upscale_factor, enable_interpolation, target_fps):
        operation_id = str(uuid.uuid4())
        base_video = os.path.join(self.output_dir, f"{operation_id}_input.mp4")
        upscaled_video = os.path.join(self.output_dir, f"{operation_id}_upscaled.mp4")
        final_video = os.path.join(self.output_dir, f"{operation_id}_final.mp4")
        
        try:
            # 首先将图片序列转换为视频
            logger.info("Converting image batch to video...")
            self._batch_to_video(images, base_video)
            
            current_input = base_video
            current_output = upscaled_video if enable_upscale else final_video

            # 放大处理
            if enable_upscale:
                logger.info(f"Applying upscale filter with factor {upscale_factor}...")
                self._apply_upscale(current_input, current_output, upscale_factor)
                current_input = current_output
                current_output = final_video

            # 插帧处理
            if enable_interpolation:
                logger.info(f"Applying frame interpolation to {target_fps} fps...")
                self._apply_interpolation(current_input, current_output, target_fps)
                current_input = current_output

            # 转换回图片序列
            logger.info("Converting final video back to image batch...")
            output_frames = self._video_to_batch(current_input)
            
        finally:
            # 清理临时文件
            for file in [base_video, upscaled_video, final_video]:
                if os.path.exists(file):
                    os.remove(file)
                    logger.debug(f"Cleaned up temporary file: {file}")
                    
        return (output_frames,)
    
    def _batch_to_video(self, image_batch, output_path):
        """Convert tensor image batch to video file"""
        frames = image_batch.cpu().numpy()
        frames = (frames * 255).astype(np.uint8)
        
        frame_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        
        try:
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                img.save(os.path.join(frame_dir, f"frame_{i:05d}.png"))
            
            cmd = [
                "ffmpeg", "-y",
                "-i", os.path.join(frame_dir, "frame_%05d.png"),
                "-c:v", "mpeg4",
                "-q:v", "2",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stderr:
                logger.debug(f"FFmpeg output: {result.stderr}")
            
        finally:
            for file in os.listdir(frame_dir):
                os.remove(os.path.join(frame_dir, file))
            os.rmdir(frame_dir)
    
    def _apply_upscale(self, input_path, output_path, scale_factor):
        """Apply upscale filter using Topaz Video AI"""
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "auto",
            "-i", input_path,
            "-vf", f"tvai_up=scale={scale_factor}",
            "-c:v", "mpeg4",
            "-q:v", "2",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr:
            logger.debug(f"Topaz upscale filter output: {result.stderr}")
        
    def _apply_interpolation(self, input_path, output_path, target_fps):
        """Apply frame interpolation filter using Topaz Video AI"""
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"tvai_fi=fps={target_fps}",
            "-c:v", "mpeg4",
            "-q:v", "2",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr:
            logger.debug(f"Topaz frame interpolation output: {result.stderr}")
    
    def _video_to_batch(self, video_path):
        """Convert video file back to tensor image batch"""
        frame_dir = os.path.join(self.output_dir, "output_frames")
        os.makedirs(frame_dir, exist_ok=True)
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                os.path.join(frame_dir, "frame_%05d.png")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stderr:
                logger.debug(f"Frame extraction output: {result.stderr}")
            
            frames = []
            frame_files = sorted(os.listdir(frame_dir))
            
            for frame_file in frame_files:
                img = Image.open(os.path.join(frame_dir, frame_file))
                frame = np.array(img)
                frames.append(frame)
                
            frames_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
            
            return frames_tensor
            
        finally:
            for file in os.listdir(frame_dir):
                os.remove(os.path.join(frame_dir, file))
            os.rmdir(frame_dir)

# __init__.py

NODE_CLASS_MAPPINGS = {
    "TopazVideoAI": TopazVideoAINode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TopazVideoAI": "Topaz Video AI (Upscale & Frame Interpolation)"
}
