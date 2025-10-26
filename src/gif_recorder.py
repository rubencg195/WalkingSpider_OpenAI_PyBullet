"""
GIF Recorder for Walking Spider Environment

Automatically captures 10-second GIF snapshots during training for visual debugging.
Saves GIFs with timestamp filenames to the videos/ folder.

Usage:
    from gif_recorder import GifRecorder
    
    # In environment __init__:
    self.gif_recorder = GifRecorder(save_dir='videos', duration_seconds=10, fps=30)
    if render_mode:
        self.gif_recorder.start_recording()
    
    # In environment step():
    if self.envStepCounter % 2 == 0:  # Capture every other frame for 30fps
        self.gif_recorder.capture_frame()
"""

import os
import numpy as np
from datetime import datetime

try:
    import imageio
except ImportError:
    print("⚠️  imageio not installed. GIF recording disabled.")
    print("   Install with: pip install imageio")
    imageio = None


class GifRecorder:
    """
    Records simulation frames and saves them as GIF snapshots.
    
    Automatically saves a GIF when the specified duration is reached,
    then starts recording a new one.
    """
    
    def __init__(self, save_dir='videos', duration_seconds=10, fps=30, 
                 width=640, height=480, enabled=True):
        """
        Initialize GIF recorder.
        
        Args:
            save_dir: Directory to save GIF files
            duration_seconds: Length of each GIF in seconds
            fps: Frames per second for the GIF
            width: Frame width in pixels
            height: Frame height in pixels
            enabled: Enable/disable recording
        """
        self.save_dir = save_dir
        self.duration_seconds = duration_seconds
        self.fps = fps
        self.width = width
        self.height = height
        self.max_frames = duration_seconds * fps
        self.frames = []
        self.recording = False
        self.enabled = enabled and (imageio is not None)
        self.gif_count = 0
        
        if self.enabled:
            os.makedirs(save_dir, exist_ok=True)
            print(f"✅ GIF Recorder initialized: {duration_seconds}s @ {fps}fps → {save_dir}/")
        else:
            print("⚠️  GIF Recorder disabled")
    
    def start_recording(self):
        """Start recording a new GIF."""
        if not self.enabled:
            return
            
        self.frames = []
        self.recording = True
        
    def stop_recording(self):
        """Stop recording without saving."""
        self.recording = False
        self.frames = []
    
    def capture_frame(self, pybullet_instance):
        """
        Capture a frame from PyBullet simulation.
        
        Args:
            pybullet_instance: PyBullet physics client (p)
        
        Returns:
            bool: True if frame was captured, False otherwise
        """
        if not self.enabled or not self.recording:
            return False
        
        try:
            import pybullet as p
            
            # Set up camera view matrix
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=0.8,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
                physicsClientId=pybullet_instance
            )
            
            # Set up projection matrix
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, 
                aspect=float(self.width) / self.height,
                nearVal=0.1, 
                farVal=100.0,
                physicsClientId=pybullet_instance
            )
            
            # Capture image
            (_, _, px, _, _) = p.getCameraImage(
                width=self.width, 
                height=self.height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=pybullet_instance
            )
            
            # Convert to RGB array
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (self.height, self.width, 4))[:, :, :3]
            self.frames.append(rgb_array)
            
            # Auto-save when duration reached
            if len(self.frames) >= self.max_frames:
                self.save_gif()
                self.start_recording()  # Start new recording automatically
                
            return True
            
        except Exception as e:
            print(f"⚠️  Error capturing frame: {e}")
            return False
    
    def capture_frame_from_array(self, rgb_array):
        """
        Capture a frame from an existing RGB array.
        
        Args:
            rgb_array: numpy array of shape (height, width, 3)
        """
        if not self.enabled or not self.recording:
            return False
            
        self.frames.append(rgb_array)
        
        # Auto-save when duration reached
        if len(self.frames) >= self.max_frames:
            self.save_gif()
            self.start_recording()  # Start new recording
            
        return True
    
    def save_gif(self, custom_name=None):
        """
        Save the recorded frames as a GIF.
        
        Args:
            custom_name: Optional custom filename (without extension)
        
        Returns:
            str: Path to saved GIF file, or None if failed
        """
        if not self.enabled:
            return None
            
        if not self.frames:
            print("⚠️  No frames to save")
            return None
        
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if custom_name:
                filename = os.path.join(self.save_dir, f'{custom_name}_{timestamp}.gif')
            else:
                self.gif_count += 1
                filename = os.path.join(self.save_dir, f'spider_snapshot_{timestamp}.gif')
            
            # Save GIF
            imageio.mimsave(filename, self.frames, fps=self.fps, loop=0)
            
            print(f"✅ GIF saved: {filename} ({len(self.frames)} frames, {self.duration_seconds}s)")
            
            # Clear frames
            self.frames = []
            self.recording = False
            
            return filename
            
        except Exception as e:
            print(f"❌ Error saving GIF: {e}")
            return None
    
    def get_frame_count(self):
        """Get current number of recorded frames."""
        return len(self.frames)
    
    def get_progress(self):
        """
        Get recording progress as percentage.
        
        Returns:
            float: Progress from 0.0 to 1.0
        """
        if self.max_frames == 0:
            return 0.0
        return min(1.0, len(self.frames) / self.max_frames)
    
    def is_recording(self):
        """Check if currently recording."""
        return self.recording and self.enabled


# Convenience function
def create_gif_recorder(save_dir='videos', duration=10, fps=30, enabled=True):
    """
    Create a GIF recorder instance.
    
    Args:
        save_dir: Directory to save GIFs
        duration: Length of each GIF in seconds
        fps: Frames per second
        enabled: Enable/disable recording
    
    Returns:
        GifRecorder instance
    """
    return GifRecorder(
        save_dir=save_dir,
        duration_seconds=duration,
        fps=fps,
        enabled=enabled
    )

