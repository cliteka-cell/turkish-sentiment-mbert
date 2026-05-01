# Cross-Lingual Transfer: Turkish Sentiment Analysis

🔗 **[Try the live demo](https://huggingface.co/spaces/Reshiman/turkish-sentiment-mbert)**

A cross-lingual transfer experiment investigating how well NLP models generalize
across languages. Trained an English sentiment model and tested it on Turkish —
then fine-tuned mBERT on Turkish data to measure the recovery gap.

## The Experiment

**Question:** Can a model trained only on English sentiment understand Turkish?

| Model | Accuracy | Notes |
|-------|----------|-------|
| English DistilBERT (zero-shot) | 0.4171 | Worse than random |
| Random baseline | 0.5000 | Coin flip |
| mBERT fine-tuned on Turkish | **0.8436** | +0.4265 recovery |

**The English model didn't just fail — it actively predicted the wrong label
more often than chance.** This means it developed a systematic bias from
Turkish tokens it couldn't read, flipping predictions in the wrong direction.

## Dataset

[Turkish Sentiment Dataset](https://huggingface.co/datasets/sepidmnorozy/Turkish_sentiment)
— 4,486 Turkish restaurant reviews annotated with positive/negative sentiment.

| Split | Negative | Positive |
|-------|----------|----------|
| Train | 2,175 (48.5%) | 2,311 (51.5%) |
| Validation | 105 | — |
| Test | 211 | — |

## Models

| Model | Purpose |
|-------|---------|
| `distilbert-base-uncased-finetuned-sst-2-english` | Zero-shot baseline |
| `bert-base-multilingual-cased` (mBERT) | Fine-tuned on Turkish |

mBERT was pretrained on 104 languages including Turkish, making it capable
of cross-lingual transfer with minimal fine-tuning data.

## Results

### mBERT Fine-tuned Performance

| Epoch | Training Loss | Val Loss | Accuracy | F1 |
|-------|--------------|----------|----------|----|
| 1 | — | 0.5810 | 0.6952 | 0.6863 |
| 2 | — | 0.5647 | 0.7333 | 0.7500 |
| 3 | — | 0.5864 | 0.7333 | 0.7627 |

**Test set: Accuracy = 0.8436 / F1 = 0.8736**

## Linguistic Observations

**Worse than random is more dangerous than random:** The English model at 41.7%
isn't just ignorant of Turkish — it's confidently wrong. Turkish tokens
get split into meaningless subword pieces by the English tokenizer, which
the model interprets as systematic signal, producing inverted predictions.
A model that says "I don't know" is safer than one that says the wrong thing
with confidence.

**Multilingual pretraining + small data = strong recovery:** mBERT reached
0.84 accuracy with only 4,486 training examples. This shows that cross-lingual
transfer works not through brute-force multilingual training data, but through
shared subword representations across related scripts.

**Within-language domain mismatch:** The training data consists of restaurant
reviews. Testing on general Turkish text (e.g. weather observations, news) shows
further degradation — domain mismatch compounds cross-lingual mismatch as
separate failure modes.

**Turkish morphology as a challenge:** Turkish is an agglutinative language —
a single word can carry the meaning of an entire English sentence
(*"yapamayacaklardandım"* = "I was among those who would not be able to do it").
WordPiece tokenization splits these long words into subword fragments,
losing morphological coherence that a linguist would preserve.

## How to Run

Open the notebook in Google Colab and run all cells.

## Tech Stack

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace Datasets](https://github.com/huggingface/datasets)
- Python 3.x, PyTorch, Gradio
