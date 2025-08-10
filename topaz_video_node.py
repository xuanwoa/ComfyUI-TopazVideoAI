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
import re
import folder_paths

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TopazVideoAI')

class TopazUpscaleParamsNode:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = ["auto", "aaa-9", "ahq-12", "alq-13", "alqs-2", "amq-13", "amqs-2", "ghq-5", "iris-2", "iris-3", "nyx-3", "prob-4", "thf-4", "thd-3", "thm-2", "rhea-1", "rxl-1"]
        return {
            "required": {
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "upscale_model": (upscale_models, {"default": "auto"}),
                "compression": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "previous_upscale": ("UPSCALE_PARAMS",),
            }
        }

    RETURN_TYPES = ("UPSCALE_PARAMS",)
    FUNCTION = "get_params"
    CATEGORY = "video"

    def get_params(self, upscale_factor=2.0, upscale_model="auto", compression=0.0, blend=0.0, previous_upscale=None):
        if upscale_model == "thm-2" and upscale_factor != 1.0:
            upscale_factor = 1.0
            logger.warning("thm-2 forces upscale_factor=1.0")
            
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
        if not CUPY_AVAILABLE:
            logger.warning("CuPy not available. Some GPU operations will be disabled.")

    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = ["auto", "aaa-9", "ahq-12", "alq-13", "alqs-2", "amq-13", "amqs-2", "ghq-5", "iris-2", "iris-3", "nyx-3", "prob-4", "thf-4", "thd-3", "thm-2", "rhea-1", "rxl-1"]
        interpolation_models = ["auto", "apo-8", "apf-1", "chr-2", "chf-3"]
        return {
            "required": {
                "images": ("IMAGE",),
                "input_fps": ("INT", {"default": 24, "min": 1, "max": 240}),
                "enable_upscale": ("BOOLEAN", {"default": False}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "upscale_model": (upscale_models, {"default": "auto"}),
                "compression": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "enable_interpolation": ("BOOLEAN", {"default": False}),
                "interpolation_multiplier": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5}),
                "interpolation_model": (interpolation_models, {"default": "auto"}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "topaz_ffmpeg_path": ("STRING", {"default": r"C:\Program Files\Topaz Labs LLC\Topaz Video AI"}),
                "force_topaz_ffmpeg": ("BOOLEAN", {"default": True}),
                "save_video": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "TopazVideo"}),
            },
            "optional": {
                "previous_upscale": ("UPSCALE_PARAMS",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "process_video"
    CATEGORY = "video"

    def _get_system_ffmpeg(self):
        return None

    def _get_topaz_ffmpeg_path(self, ffmpeg_base_path, for_topaz=False, force_topaz=True):
        return os.path.join(ffmpeg_base_path, 'ffmpeg.exe')

    def _save_batch(self, frames_batch, frame_dir, start_idx):
        for i, frame in enumerate(frames_batch):
            frame_path = os.path.join(frame_dir, f"frame_{start_idx + i:05d}.png")
            Image.fromarray(frame).save(frame_path)

    def _batch_to_video(self, image_batch, output_path, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg, input_fps=24):
        frames = (image_batch.cpu().numpy() * 255).astype(np.uint8)
        frame_dir = os.path.join(self.output_dir, f"input_frames_{uuid.uuid4()}")
        os.makedirs(frame_dir, exist_ok=True)
        
        try:
            with ThreadPoolExecutor() as executor:
                for i in range(0, len(frames), 32):
                    executor.submit(self._save_batch, frames[i:i+32], frame_dir, i)
            
            ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path, False, force_topaz_ffmpeg)
            cmd = [
                ffmpeg_exe, "-y", "-hide_banner", "-nostdin", "-strict", "2",
                "-hwaccel", "auto", "-framerate", str(input_fps),
                "-i", os.path.join(frame_dir, "frame_%05d.png"),
            ]
            
            if use_gpu:
                cmd.extend(["-c:v", "hevc_nvenc", "-profile", "main", "-preset", "medium", "-global_quality", "19", "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov"])
            else:
                cmd.extend(["-c:v", "mpeg4", "-q:v", "2"])
            
            cmd.append(output_path)
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def _video_to_batch(self, video_path, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg):
        frame_dir = os.path.join(self.output_dir, f"output_frames_{uuid.uuid4()}")
        os.makedirs(frame_dir, exist_ok=True)
        
        try:
            ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path, False, force_topaz_ffmpeg)
            cmd = [ffmpeg_exe, "-y", "-i", video_path, "-vsync", "0", os.path.join(frame_dir, "frame_%05d.png")]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
            if not frame_files:
                raise ValueError(f"No frames extracted from video: {video_path}")
            
            frames = [np.array(Image.open(os.path.join(frame_dir, f))) for f in frame_files]
            return torch.from_numpy(np.stack(frames)).float() / 255.0
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def process_video(self, images, input_fps, enable_upscale, upscale_factor, upscale_model, compression, blend,
                      enable_interpolation, interpolation_multiplier, interpolation_model, use_gpu, topaz_ffmpeg_path, 
                      force_topaz_ffmpeg, save_video=False, filename_prefix="TopazVideo", previous_upscale=None):
        
        operation_id = str(uuid.uuid4())
        input_video = os.path.join(self.output_dir, f"{operation_id}_input.mp4")
        intermediate_video = os.path.join(self.output_dir, f"{operation_id}_intermediate.mp4")
        final_video_path = os.path.join(self.output_dir, f"{operation_id}_output.mp4")
        self.temp_files.extend([input_video, intermediate_video, final_video_path])
        
        try:
            logger.info(f"Converting image batch to video with input fps {input_fps}...")
            self._batch_to_video(images, input_video, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg, input_fps)
            
            current_input = input_video
            current_output = intermediate_video
            final_fps = input_fps

            if enable_upscale:
                all_upscale_params = previous_upscale if previous_upscale else []
                all_upscale_params.append({
                    "upscale_factor": upscale_factor, "upscale_model": upscale_model,
                    "compression": compression, "blend": blend
                })
                
                upscale_filters = [
                    f"tvai_up=model={p['upscale_model']}:scale={p['upscale_factor']}:estimate=8:compression={p['compression']}:blend={p['blend']}"
                    for p in all_upscale_params
                ]
                filter_chain = ','.join(upscale_filters)
                logger.info(f"Applying upscale filter chain: {filter_chain}")

                ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path, True, force_topaz_ffmpeg)
                cmd = [
                    ffmpeg_exe, "-y", "-hide_banner", "-nostdin", "-strict", "2",
                    "-hwaccel", "auto", "-i", current_input, "-vf", filter_chain,
                ]
                
                if use_gpu:
                    cmd.extend(["-c:v", "hevc_nvenc", "-profile", "main", "-preset", "medium", "-global_quality", "19", "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov"])
                else:
                    cmd.extend(["-c:v", "mpeg4", "-q:v", "2"])
                
                cmd.append(current_output) # NO -r parameter here to preserve duration
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg upscale error: {result.stderr}")
                
                current_input = current_output
                current_output = final_video_path
            
            if enable_interpolation:
                target_fps = int(input_fps * interpolation_multiplier)
                final_fps = target_fps
                logger.info(f"Applying interpolation: target fps: {target_fps}")
                
                interpolation_filter = f"tvai_fi=model={interpolation_model}:fps={target_fps}"
                
                ffmpeg_exe = self._get_topaz_ffmpeg_path(topaz_ffmpeg_path, True, force_topaz_ffmpeg)
                cmd = [
                    ffmpeg_exe, "-y", "-hide_banner", "-nostdin", "-strict", "2",
                    "-hwaccel", "auto", "-i", current_input, "-vf", interpolation_filter,
                ]
                
                if use_gpu:
                    cmd.extend(["-c:v", "hevc_nvenc", "-profile", "main", "-preset", "medium", "-global_quality", "19", "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov"])
                else:
                    cmd.extend(["-c:v", "mpeg4", "-q:v", "2"])
                
                cmd.extend(["-r", str(target_fps), current_output]) # ADD -r parameter here
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg interpolation error: {result.stderr}")
            else:
                if current_input != current_output:
                    shutil.copy2(current_input, current_output)

            final_processed_video = current_output

            if save_video:
                output_dir = folder_paths.get_output_directory()
                full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
                
                max_counter = 0
                matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
                for f in os.listdir(full_output_folder):
                    match = matcher.fullmatch(f)
                    if match:
                        max_counter = max(max_counter, int(match.group(1)))
                
                counter = max_counter + 1
                output_file = f"{filename}_{counter:05}.mp4"
                output_path = os.path.join(full_output_folder, output_file)
                
                shutil.copy2(final_processed_video, output_path)
                logger.info(f"Saved video to: {output_path}")
                
                previews = [{"filename": output_file, "subfolder": subfolder, "type": "output", "format": "video/mp4"}]
                
                if hasattr(self, '_connected_outputs') and self._connected_outputs:
                    output_frames = self._video_to_batch(final_processed_video, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg)
                    return {"ui": {"previews": previews}, "result": (output_frames,)}
                else:
                    return {"ui": {"previews": previews}, "result": (images,)}
            else:
                output_frames = self._video_to_batch(final_processed_video, use_gpu, topaz_ffmpeg_path, force_topaz_ffmpeg)
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
