# Research Summary: Wav2Vec 2.0

**Paper Title:** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*
**Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli
**Venue:** NeurIPS 2020 (Facebook AI Research)

---

## 1. What Problem Does It Solve?

Historically, Automatic Speech Recognition (ASR) systems required massive amounts of labeled audio data (transcribed speech) to achieve acceptable accuracy. Transcribing speech is extremely labor-intensive, time-consuming, and expensive, creating a significant bottleneck for low-resource languages that lack extensive annotated corpora.

Wav2Vec 2.0 solves this data-efficiency problem through **Self-Supervised Learning (SSL)**. It learns high-quality speech representations directly from raw, unannotated audio. By pretraining on vast amounts of unlabeled speech, the model learns the phonetic structure of language. It can then be fine-tuned on a tiny fraction of labeled data (as little as 10 minutes) to achieve highly competitive Word Error Rate (WER) scores — democratizing ASR for low-resource languages.

## 2. How Does the Architecture Work?

Wav2Vec 2.0 is composed of three primary building blocks:

```
                  ┌───────────────────────────────┐
                  │      Transformer Context      │ ──> Context Representations (C)
                  │            Network            │
                  └───────────────┬───────────────┘
                                  ▲
                            [Masked Inputs]
                                  │
                  ┌───────────────────────────────┐      ┌─────────────────────────┐
Raw Waveform ──>  │     CNN Feature Encoder       │ ──>  │   Quantization Module   │ ──> Codebook Targets (Q)
                  └───────────────────────────────┘      └─────────────────────────┘
```

1. **Feature Encoder (CNN):**
   - The input is raw audio waveform *X* sampled at 16 kHz.
   - A multi-layer temporal convolutional network (CNN) processes the waveform through temporal downsampling.
   - It outputs local latent feature representations *Z = (z₁, z₂, ..., z_T)* at a frame rate of 50 Hz (every 20 ms of audio).

2. **Context Network (Transformer):**
   - The latent representations *Z* are mapped to a higher dimension and added to relative positional embeddings.
   - During pretraining, a proportion of the feature vector time steps (around 50%) are masked.
   - The masked vectors are fed into a standard Transformer architecture to capture long-range contextual relationships, outputting context representations *C = (c₁, c₂, ..., c_T)*.

3. **Quantization Module (Vector Quantization):**
   - To define a target task for self-supervised training, the continuous encoder outputs *Z* are discretized into codebook representations *Q = (q₁, q₂, ..., q_T)*.
   - It uses **Gumbel-Softmax** to select discrete codebook entries from product-quantization codebooks. Specifically, it employs *G* groups (typically *G = 2*) with *V* entries (typically *V = 320*) each. The chosen entries are concatenated and projected to form *q_t*.

4. **Pretraining Objective:**
   - **Contrastive Loss:** Given a masked frame *t*, the context network output *c_t* is trained to identify the true quantized target *q_t* among a set of distractor representations drawn from other masked frames in the same sequence.
   - **Diversity Loss:** Encourages the model to use all codebook entries uniformly, preventing "codebook collapse" where only a small subset of vectors are selected.

5. **Fine-Tuning:**
   - For speech recognition, the quantization module is discarded.
   - A linear projection layer is added on top of the Transformer Context Network.
   - The model is fine-tuned using **Connectionist Temporal Classification (CTC)** loss on labeled transcripts, learning the mapping from context vectors to character sequences.

## 3. Why Is It Better Than Previous Approaches?

- **Direct Raw Waveform Processing:** Unlike traditional systems that rely on hand-crafted log-mel filterbank or MFCC features, Wav2Vec 2.0 extracts optimal features directly from the raw audio signal.
- **End-to-End Joint Representation Learning:** Previous systems, such as *vq-wav2vec*, learned quantization codes in a separate, disjoint step before training the context network. Wav2Vec 2.0 learns quantization targets and context representations *jointly* in one end-to-end framework, improving representation quality.
- **Unprecedented Data Efficiency:** Wav2Vec 2.0 set new benchmarks in semi-supervised training. Fine-tuning a pretrained model on just **10 minutes** of LibriSpeech audio achieved a WER of **4.8%** on the clean test set — matching or outperforming models trained on 100× more data in prior years.

## 4. What Datasets Were Used?

- **Pretraining Data:**
  - **LibriSpeech:** 960 hours of unannotated audiobooks (clean and noisy splits).
  - **Libri-Light:** 60,000 hours of unannotated speech extracted from LibriVox audiobooks.
- **Fine-Tuning Data:**
  - **LibriSpeech train splits:** configured in different sizes (10 minutes, 1 hour, 10 hours, 100 hours, and the full 960 hours) to demonstrate performance under various resource constraints.

## 5. What Are Its Limitations?

- **CTC Independence Assumption:** CTC assumes label predictions at different time frames are conditionally independent. Consequently, Wav2Vec 2.0 does not naturally learn a strong language model over the text output and benefits from an external n-gram language model during decoding for optimal word-level accuracy.
- **Domain Mismatch Vulnerability:** When pretrained on clean audiobook speech, performance degrades significantly on accented, noisy, or domain-specific conversational speech.
- **Computational Cost:** Pretraining is extremely compute-heavy, requiring large GPU clusters (e.g., 64 V100 GPUs) for several days. This makes custom pretraining inaccessible to many researchers, leaving them dependent on available pretrained checkpoints.
- **Fixed Sampling Rate Requirement:** Wav2Vec 2.0 requires audio sampled at exactly 16 kHz. Audio recorded at other rates must be resampled, adding preprocessing overhead and potential quality degradation.

---

*This summary corresponds to the model used in this repository's pipeline: [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) — the Wav2Vec 2.0 base model fine-tuned on 960 hours of LibriSpeech.*
