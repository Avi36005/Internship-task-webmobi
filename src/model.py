import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import logging as hf_logging

# The base checkpoint carries a pretraining-only buffer (`masked_spec_embed`)
# that is unused at inference, which triggers a noisy (but harmless) "weights
# were not initialized / you should TRAIN this model" warning. Silence it.
hf_logging.set_verbosity_error()


def _resample(audio_array: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Resample a 1D waveform to `target_sr`.

    librosa is imported lazily and only when resampling is actually needed, so a
    missing or broken librosa install (its numba/llvmlite stack is fragile) never
    breaks the common case where audio is already 16 kHz. If librosa is
    unavailable we fall back to scipy, which is a lighter and more reliable
    dependency.
    """
    if orig_sr == target_sr:
        return audio_array
    try:
        import librosa
        return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
    except Exception:
        # Fallback: high-quality polyphase resampling via scipy.
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(orig_sr), int(target_sr))
        up, down = int(target_sr) // g, int(orig_sr) // g
        return resample_poly(audio_array, up, down).astype(np.float32)

class SpeechModel:
    """
    ASR Model wrapper utilizing Wav2Vec2 for Connectionist Temporal Classification (CTC) inference.
    """
    def __init__(self, model_id: str = "facebook/wav2vec2-base-960h"):
        # Select device (CPU/CUDA/MPS)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            # Support Apple Silicon MPS for fast inference if running on macOS locally
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"Initializing {model_id} processor...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        
        print(f"Loading pretrained model {model_id} to device: {self.device}...")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
    def transcribe(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        """
        Transcribe raw audio waveform into text.
        Args:
            audio_array (np.ndarray): The 1D float array representing the audio waveform.
            sampling_rate (int): The original sampling rate of the audio file.
        Returns:
            str: Transcribed text in uppercase.
        """
        # Ensure a 1D float32 mono waveform. soundfile returns float64 and can
        # return a 2D array (samples, channels) for stereo audio, which both
        # librosa.resample and the Wav2Vec2 processor reject.
        audio_array = np.asarray(audio_array, dtype=np.float32)
        if audio_array.ndim > 1:
            # Average channels down to mono
            audio_array = audio_array.mean(axis=1)

        # Wav2Vec2 expects 16kHz sampling rate; resample anything else.
        if sampling_rate != 16000:
            audio_array = _resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000
            
        # Process inputs
        inputs = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        
        # Perform forward pass without gradients
        with torch.no_grad():
            logits = self.model(input_values).logits
            
        # Take argmax to decode tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        # Ensure standard uppercase transcript representation
        return transcription.upper().strip()
