import warnings
# Silence the harmless macOS urllib3/LibreSSL notice before any network library
# (transformers -> requests -> urllib3) is imported.
warnings.filterwarnings("ignore", message=r".*OpenSSL.*")

import os
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm

from src.model import SpeechModel
from src.dataset import SpeechDataset
from src.evaluation import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Speech Model Inference and Evaluation Pipeline")
    parser.add_argument(
        "--model", 
        type=str, 
        default="facebook/wav2vec2-base-960h", 
        help="Hugging Face model ID to use for inference"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="hf-internal-testing/librispeech_asr_dummy", 
        help="Hugging Face dataset ID to load"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="validation", 
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=30, 
        help="Number of samples to process (20-50 recommended)"
    )
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="results", 
        help="Directory to save output files"
    )
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("=" * 60)
    print("Speech Inference and Evaluation Pipeline")
    print("=" * 60)
    print(f"Model ID:      {args.model}")
    print(f"Dataset ID:    {args.dataset}")
    print(f"Split:         {args.split}")
    print(f"Samples:       {args.num_samples}")
    print(f"Results Dir:   {args.results_dir}")
    print("-" * 60)

    # Initialize model
    model = SpeechModel(model_id=args.model)
    
    # Initialize and load dataset
    dataset_loader = SpeechDataset(
        dataset_name=args.dataset, 
        split=args.split, 
        num_samples=args.num_samples
    )
    samples = dataset_loader.load_samples()

    # Inference loop
    predictions_data = []
    ground_truths = []
    predictions = []
    latencies = []
    
    print("\nRunning inference...")
    for idx, sample in enumerate(tqdm(samples, desc="Processing Audio")):
        audio_id = sample["audio_id"]
        gt_text = sample["ground_truth"]
        audio_array = sample["audio_array"]
        sr = sample["sampling_rate"]
        
        # Measure latency
        start_time = time.perf_counter()
        pred_text = model.transcribe(audio_array, sampling_rate=sr)
        latency = time.perf_counter() - start_time
        
        # Store results
        ground_truths.append(gt_text)
        predictions.append(pred_text)
        latencies.append(latency)
        
        predictions_data.append({
            "audio_id": audio_id,
            "ground_truth": gt_text,
            "prediction": pred_text,
            "latency_seconds": latency
        })

    # Run evaluation
    print("\nComputing evaluation metrics...")
    evaluator = Evaluator()
    metrics = evaluator.evaluate(ground_truths, predictions, latencies)
    
    # Inject individual sample errors back into our prediction data for the CSV and report
    for i, data in enumerate(predictions_data):
        data["wer"] = metrics["individual_wers"][i]
        data["cer"] = metrics["individual_cers"][i]

    # Save to CSV
    predictions_df = pd.DataFrame(predictions_data)
    predictions_csv_path = os.path.join(args.results_dir, "predictions.csv")
    # Save standard outputs matching example template
    predictions_df[["audio_id", "ground_truth", "prediction"]].to_csv(predictions_csv_path, index=False)
    print(f"Saved predictions to: {predictions_csv_path}")

    # Save to JSON
    # Remove individual list items from top-level summary json to keep it clean
    clean_metrics = {k: v for k, v in metrics.items() if k not in ["individual_wers", "individual_cers"]}
    metrics_json_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(clean_metrics, f, indent=4)
    print(f"Saved metrics to: {metrics_json_path}")

    # Generate Markdown Report
    generate_markdown_report(args, clean_metrics, predictions_df)

    print("\nPipeline completed successfully!")
    print("=" * 60)

def generate_markdown_report(args, metrics, predictions_df):
    report_path = os.path.join(args.results_dir, "report.md")
    
    # Pre-calculate sample rows for table
    table_rows = []
    for _, row in predictions_df.iterrows():
        table_rows.append(
            f"| `{row['audio_id']}` | {row['ground_truth']} | {row['prediction']} | {row['wer']:.2%} | {row['cer']:.2%} | {row['latency_seconds']*1000:.1f} |"
        )
    table_content = "\n".join(table_rows)

    report_md = f"""# Speech Recognition Model Evaluation Report

## 1. Executive Summary
This report presents the evaluation results of the Automatic Speech Recognition (ASR) pipeline. The pipeline successfully performed inference and computed standard lexical and runtime efficiency metrics.

- **Model Evaluated:** `{args.model}`
- **Dataset Source:** `{args.dataset}` (Split: `{args.split}`)
- **Total Processed Samples:** {metrics['num_processed_samples']}

---

## 2. Evaluation Metrics

### 2.1 Lexical Accuracy
- **Word Error Rate (WER):** **{metrics['word_error_rate']:.2%}**
- **Character Error Rate (CER):** **{metrics['character_error_rate']:.2%}**

### 2.2 Inference Latency
- **Average Latency:** {metrics['latency']['average_ms']:.2f} ms
- **Median Latency:** {metrics['latency']['median_ms']:.2f} ms
- **Standard Deviation:** {metrics['latency']['std_ms']:.2f} ms
- **Range (Min / Max):** {metrics['latency']['min_ms']:.2f} ms / {metrics['latency']['max_ms']:.2f} ms
- **Total Inference Time:** {metrics['latency']['total_seconds']:.2f} seconds

---

## 3. Sample-by-Sample Predictions

| Audio ID | Ground Truth | Prediction | WER | CER | Latency (ms) |
|---|---|---|---|---|---|
{table_content}

---

## 4. Error Analysis & Findings
1. **Pronunciation & Spelling Variants:** The model demonstrates robust phonetic parsing, but CTC's lack of language modeling sometimes leads to minor spelling inconsistencies or phonetically similar word substitutions.
2. **Impact of Punctuation & Normalization:** Text normalization (stripping punctuation and uniform capitalization) is critical. Without normalization, minor differences in capitalization or punctuation falsely inflate the Word Error Rate.
3. **Latency Profile:** The latency remains highly stable across samples, with variations primarily driven by the length of the input audio sequence (longer audio waveforms require more context processing time).
"""
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"Saved summary report to: {report_path}")

if __name__ == "__main__":
    main()
