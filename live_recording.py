import pyaudio
import numpy as np
import threading
import time
import os
from tqdm import tqdm
from scipy.io import wavfile

class LiveAudioRecorder:
    def __init__(self, output_directory="recordings", device_index=None):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.recording = False
        self.frames = []
        self.output_directory = output_directory
        self.stream = None
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Validate and set device index
        if device_index is not None:
            try:
                device_info = self.audio.get_device_info_by_index(device_index)
                if device_info['maxInputChannels'] > 0:
                    self.device_index = device_index
                else:
                    raise ValueError(f"Device {device_index} has no input channels")
            except Exception as e:
                raise ValueError(f"Invalid device index {device_index}: {str(e)}")
        else:
            # Find default input device
            default_device = None
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    default_device = i
                    break
            if default_device is None:
                raise ValueError("No input devices found")
            self.device_index = default_device
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
    
    def start_recording(self):
        """Start recording audio from the microphone"""
        if self.recording:
            return  # Already recording
            
        self.recording = True
        self.frames = []
        
        try:
            # Get device info to verify settings
            device_info = self.audio.get_device_info_by_index(self.device_index)
            
            # Open audio stream with verified device settings
            stream_params = {
                'format': self.FORMAT,
                'channels': min(self.CHANNELS, int(device_info['maxInputChannels'])),
                'rate': int(min(self.RATE, int(device_info['defaultSampleRate']))),
                'input': True,
                'input_device_index': self.device_index,
                'frames_per_buffer': self.CHUNK
            }
            
            self.stream = self.audio.open(**stream_params)
            
            # Verify stream is active
            if not self.stream.is_active():
                raise OSError("Failed to activate audio stream")
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record)
            self.record_thread.start()
            
        except Exception as e:
            self.recording = False
            if self.stream:
                self.stream.close()
                self.stream = None
            raise RuntimeError(f"Failed to start recording: {str(e)}")
        # Open audio stream
        stream_params = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.RATE,
            'input': True,
            'frames_per_buffer': self.CHUNK
        }
        
        # Add device_index if specified
        if self.device_index is not None:
            stream_params['input_device_index'] = self.device_index
            
        self.stream = self.audio.open(**stream_params)
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record)
        self.record_thread.start()
    
    def stop_recording(self):
        """Stop the recording and save the audio file"""
        self.recording = False
        self.record_thread.join()
        
        # Close the stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        print("\nProcessing audio file...")
        
        # Save the recording
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.output_directory, f"recording_{timestamp}.wav")
        
        with tqdm(total=100, desc="Transcribing audio", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            # Convert float32 data to int16
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.float32)
            
            # Normalize audio data to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            # Convert to int16 range [-32768, 32767]
            audio_data_int = (audio_data * 32767).astype(np.int16)
            pbar.update(50)  # Update progress to 50%
            
            # Save as WAV file using scipy
            wavfile.write(filename, self.RATE, audio_data_int)
            pbar.update(50)  # Complete the progress bar
        
        print(f"\nAudio file saved as: {filename}")
        return filename
    
    def _record(self):
        """Internal method to record audio data"""
        while self.recording:
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
    
    def close(self):
        """Clean up resources"""
        if self.stream:
            self.stream.close()
        self.audio.terminate()


def get_available_devices():
    """Get a list of available audio input devices"""
    audio = pyaudio.PyAudio()
    devices = []
    
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:  # Only include input devices
            devices.append({
                'index': i,
                'name': device_info['name'],
                'channels': device_info['maxInputChannels'],
                'sample_rate': int(device_info['defaultSampleRate'])
            })
    
    audio.terminate()
    return devices


# Example usage
if __name__ == "__main__":
    recorder = LiveAudioRecorder()
    
    print("Recording will start in 3 seconds...")
    time.sleep(3)
    
    print("Recording started! Speak into your microphone...")
    recorder.start_recording()
    
    # Record for 5 seconds
    time.sleep(5)
    
    print("Recording stopped!")
    filename = recorder.stop_recording()
    
    print(f"Recording saved as: {filename}")
    recorder.close()
    recorder = LiveAudioRecorder()
    recorder.start_recording()
    # ... do something while recording ...
    filename = recorder.stop_recording()
    recorder.close()