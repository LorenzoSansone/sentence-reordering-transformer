# Sentence Reconstruction

Reconstruct the original English sentence from a randomly shuffled sequence of its words, using a Transformer model trained from scratch.

This repository contains the notebook `Sentence_Reordering.ipynb`, developed for a Deep Learning course project.

## Task

Given a random permutation of the words of an English sentence, the goal is to recover the original word ordering. The output can be produced either in a single shot or autoregressively (one token at a time); this project uses the **autoregressive** approach.

## Constraints

The project must respect the following rules:

- No pretrained models.
- The model must have **fewer than 20M parameters**.
- No postprocessing (e.g., no beam search).
- No additional training data.


## Dataset

Sentences are taken from the **`generics_kb`** dataset (Hugging Face). The data is prepared as follows:

- The vocabulary is restricted to the **10,000 most frequent words**; only sentences using exclusively this vocabulary are kept.
- Only sentences **longer than 8 words** are retained.
- Commas are replaced with a `<comma>` token, and every sentence is wrapped with `<start>` and `<end>` markers.
- Text is tokenized with Keras `TextVectorization` (max 10,000 tokens); sentences containing out-of-vocabulary tokens are discarded.
- A custom `DataGenerator` produces, for each ordered sentence, a version with its inner words randomly shuffled (keeping `<start>`/`<end>` in place).

**Split (sentences):** ~220000 training / 17984 validation / 3232 test.

## Preprocessing for the model

Three aligned views of the data are built:

- **origin_data**: collection of ordered phrase
- **shuffle_data**: collection of shuffled phrase
- **shift_data**: collection of ordered phrase without the *\<start>* token. So it is similar to original data but is shifted by one. This data collection is used in order to implement *Teacher Forcing* and try to increase the quality of the result.

All sequences are padded to length 28, and the corresponding padding masks are computed.

## Model

The model is an **encoder–decoder Transformer**, built with `keras_nlp` layers following *Attention Is All You Need*.

| Hyperparameter | Value |
| --- | --- |
| Embedding dimension | 256 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| Attention heads | 25 |
| Feed-forward (intermediate) dim | 1024 |
| Dropout | 0.2 |
| Vocabulary size | 10,000 |
| Sequence length | 28 |
| **Total parameters** | **~15M** (15,003,192) — within the 20M limit |

## Training

- **Optimizer:** Adam (β₁ = 0.9, β₂ = 0.98, ε = 1e-9) with a Transformer-style **warmup learning-rate schedule** (`CustomSchedule`).
- **Loss:** categorical cross-entropy — **Metric:** categorical accuracy.
- **Batch size:** 256 — **Max epochs:** 100 with **EarlyStopping** (patience = 5).
- To reduce overfitting, the shuffled targets are **re-permuted at every epoch** (`shuffleTarget`).

Training stopped early after **13 epochs**, reaching a validation categorical accuracy of about **0.91**. The trained weights are saved to `final_model.h5 (generated after the first run of the notebook)`.

## Inference

Predictions are generated **autoregressively** (`predictSequences`): starting from `<start>`, the model predicts one token at a time and feeds it back, until `<end>` (or padding) is produced for the whole batch. The output is then cleaned of special tokens (`cleanData`).

## Evaluation metric

Quality is measured by the **longest common substring** between the source sentence `s` and the prediction `p`:

```
score = |longest_common_substring(s, p)| / max(|s|, |p|)
```

It is computed with `difflib.SequenceMatcher`.

## Results

- **Mean score: ≈ 0.523** (over 3,000 test samples).

An initial attempt with an **LSTM encoder–decoder** produced a much lower score, which motivated the switch to the Transformer architecture.

## Usage

The notebook is designed to run on Google Colab. Main dependencies:

```bash
pip install datasets keras-nlp
```

Open `Sentence_Reordering.ipynb` and run the cells in order. The trained weights (`final_model.h5`) are available on after the first run.
