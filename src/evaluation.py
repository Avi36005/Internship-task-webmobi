import re
import numpy as np
import jiwer

def normalize_text(text: str) -> str:
    """
    Standardize text formatting by removing punctuation, converting to uppercase,
    and collapsing multiple spaces to ensure fair error rate calculations.
    """
    if not isinstance(text, str):
        return ""
    text = text.upper()
    # Remove punctuation (keep letters, numbers, and basic spaces)
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse multiple spaces or tabs into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

class Evaluator:
    """
    Evaluates prediction outputs against ground truth transcripts by computing
    ASR metrics (WER, CER) and runtime performance (latency).
    """
    def __init__(self):
        pass
        
    def evaluate(self, ground_truths, predictions, latencies):
        """
        Compute evaluation metrics.
        Args:
            ground_truths (List[str]): List of reference transcripts.
            predictions (List[str]): List of predicted transcripts.
            latencies (List[float]): List of inference times per sample in seconds.
        Returns:
            dict: Evaluated metrics.
        """
        assert len(ground_truths) == len(predictions) == len(latencies), "Length mismatch in evaluation inputs"
        
        # Apply standard normalization for robust WER/CER scoring
        norm_truths = [normalize_text(gt) for gt in ground_truths]
        norm_preds = [normalize_text(p) for p in predictions]
        
        # Compute WER and CER using jiwer
        # Handle edge case where truth might be empty
        try:
            wer = jiwer.wer(norm_truths, norm_preds)
        except Exception as e:
            print(f"Error computing WER: {e}. Defaulting to 1.0")
            wer = 1.0
            
        try:
            cer = jiwer.cer(norm_truths, norm_preds)
        except Exception as e:
            print(f"Error computing CER: {e}. Defaulting to 1.0")
            cer = 1.0
            
        # Compute individual sample metrics for details
        sample_wers = []
        sample_cers = []
        for gt, p in zip(norm_truths, norm_preds):
            if not gt:
                sample_wers.append(1.0 if p else 0.0)
                sample_cers.append(1.0 if p else 0.0)
            else:
                sample_wers.append(jiwer.wer(gt, p))
                sample_cers.append(jiwer.cer(gt, p))
                
        # Calculate latency stats in milliseconds
        latencies_ms = [l * 1000.0 for l in latencies]
        
        metrics = {
            "num_processed_samples": len(ground_truths),
            "word_error_rate": float(wer),
            "character_error_rate": float(cer),
            "latency": {
                "total_seconds": float(sum(latencies)),
                "average_ms": float(np.mean(latencies_ms)),
                "median_ms": float(np.median(latencies_ms)),
                "min_ms": float(np.min(latencies_ms)),
                "max_ms": float(np.max(latencies_ms)),
                "std_ms": float(np.std(latencies_ms))
            },
            "individual_wers": [float(w) for w in sample_wers],
            "individual_cers": [float(c) for c in sample_cers]
        }
        
        return metrics
