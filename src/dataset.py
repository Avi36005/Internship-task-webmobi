import os
import io
import soundfile as sf
from datasets import load_dataset, Audio

class SpeechDataset:
    """
    Dataset loader class to retrieve public speech datasets from Hugging Face
    and decode them robustly using soundfile.
    """
    def __init__(self, dataset_name: str = "hf-internal-testing/librispeech_asr_dummy", split: str = "validation", num_samples: int = 30):
        self.dataset_name = dataset_name
        self.split = split
        self.num_samples = num_samples
        
    def load_samples(self):
        """
        Load the dataset, bypass Hugging Face's auto-decoder (to avoid torchcodec/FFmpeg issues),
        and decode raw bytes in-memory.
        
        Returns:
            List[Dict]: A list of dicts, each containing:
                        - audio_id: unique identifier
                        - ground_truth: raw transcript in uppercase
                        - audio_array: 1D numpy array
                        - sampling_rate: original sampling rate
        """
        print(f"Loading Hugging Face dataset '{self.dataset_name}' (split: '{self.split}')...")
        try:
            # Try to load with standard 'clean' configuration first
            dataset = load_dataset(self.dataset_name, "clean", split=self.split)
        except Exception as e:
            print(f"Standard configuration load failed ({e}). Loading default dataset configuration...")
            dataset = load_dataset(self.dataset_name, split=self.split)
            
        # CRITICAL: Disable Hugging Face's automatic decoding to avoid torchcodec/FFmpeg dependencies
        dataset = dataset.cast_column("audio", Audio(decode=False))
        
        total_available = len(dataset)
        num_to_take = min(self.num_samples, total_available)
        print(f"Successfully loaded. Decoding first {num_to_take} of {total_available} samples with soundfile...")
        
        samples = []
        for i in range(num_to_take):
            item = dataset[i]
            
            # Determine audio_id
            raw_id = item.get("id", item.get("file", f"sample_{i:04d}"))
            audio_id = os.path.basename(str(raw_id)).split(".")[0]
            
            # Retrieve raw audio bytes and decode manually
            audio_data = item.get("audio")
            if audio_data is None or "bytes" not in audio_data:
                print(f"Warning: Sample {i} does not contain valid audio bytes. Skipping.")
                continue
                
            try:
                audio_bytes = audio_data["bytes"]
                # sf.read handles WAV, FLAC, and other standard formats directly from bytes stream
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
            except Exception as decode_err:
                print(f"Error decoding audio for sample {i}: {decode_err}. Skipping.")
                continue
            
            # Extract and normalize ground truth
            ground_truth = item.get("text", item.get("sentence", "")).upper().strip()
            
            samples.append({
                "audio_id": audio_id,
                "ground_truth": ground_truth,
                "audio_array": audio_array,
                "sampling_rate": sampling_rate
            })
            
        return samples
