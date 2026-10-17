# Lightweight fine-tuning of GPT-2 with LoRA

Turning GPT-2, a 124M-parameter text generator, into a four-class news topic classifier by training **0.24% of its weights** with LoRA through the Hugging Face [`peft`](https://huggingface.co/docs/peft) library.

| | Accuracy | Macro-F1 | Trained parameters | Stored on disk |
|---|---:|---:|---:|---:|
| GPT-2 before fine-tuning | 27.0% | 0.115 | 0 | - |
| **GPT-2 + LoRA (r = 8)** | **89.1%** | **0.891** | **297,984** | **1.15 MB** |

Both rows are measured on the same 1,000 held-out AG News items with the same evaluation code. The fine-tuned row is the adapter **reloaded from disk**, and it reproduces the in-memory model's predictions exactly.

![Confusion matrices before and after fine-tuning](assets/before-after.png)

## What is in this repository

```
LightweightFineTuning.ipynb     the notebook: load, evaluate, fine-tune, save, reload, re-evaluate
gpt2-lora-ag-news/              the saved PEFT weights (LoRA adapters + classification head)
  adapter_config.json
  adapter_model.safetensors
assets/                         figures used in this README
requirements.txt
```

## How to run it

**Locally** (tested with Python 3.13 on an Apple Silicon Mac; any machine with a recent PyTorch works):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab LightweightFineTuning.ipynb
```

Then run all cells. The whole notebook takes about six minutes on a laptop GPU: two minutes for the main fine-tuning run, and the rest for the optional comparison of configurations at the end. GPT-2 and the AG News dataset download from the Hugging Face Hub on first use.

**In the course workspace**, open the notebook and run all cells. The code adapts to older library versions on its own (the three places that need it are marked "Compatibility 1/2/3 of 3" in the notebook).

**Just the saved model**, without re-training:

```python
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer

labels = ["World", "Sports", "Business", "Sci/Tech"]
model = AutoPeftModelForSequenceClassification.from_pretrained("gpt2-lora-ag-news", num_labels=4)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id   # GPT-2 needs this to find the last real token

inputs = tokenizer("Striker scores twice in stoppage time to win the final", return_tensors="pt")
print(labels[model(**inputs).logits.argmax(-1).item()])   # Sports
```

## Approach

### 1. Prepare the foundation model

- **Model:** `openai-community/gpt2` loaded with `AutoModelForSequenceClassification`, which keeps the pre-trained transformer and adds a new, randomly initialized 4-way linear layer (`score`).
- **Dataset:** [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (World, Sports, Business, Sci/Tech). A random 4,000 items for training and 1,000 for testing, all four classes roughly balanced.
- **Tokenizer:** GPT-2's own. GPT-2 has no padding token, so the end-of-text token is reused for padding and the model is told which id that is.
- **Baseline evaluation:** `Trainer.evaluate()` with accuracy and macro-F1.

### 2. Lightweight fine-tuning

`LoraConfig(task_type=SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["c_attn"], fan_in_fan_out=True)`, wrapped with `get_peft_model`, trained for 2 epochs with the Hugging Face `Trainer` (learning rate 5e-4, batch size 16), and saved with `save_pretrained`.

LoRA keeps every original weight $W_0$ frozen and learns a small update next to it, $W_0 + \frac{\alpha}{r}BA$, where $A$ has $r$ rows and $B$ has $r$ columns. Where the 297,984 trained parameters come from, step by step:

![Counting the trained LoRA parameters on GPT-2, step by step](assets/lora-parameter-count.gif)

| Step | Calculation | Result |
|---|---|---:|
| `c_attn` (query, key, value together) maps 768 to | $3 \times 768$ | 2,304 |
| $A$: $r$ rows × 768 columns | $8 \times 768$ | 6,144 |
| $B$: 2,304 rows × $r$ columns | $2304 \times 8$ | 18,432 |
| One layer's adapter, $A + B$ | $6{,}144 + 18{,}432$ | 24,576 |
| All 12 layers | $12 \times 24{,}576$ | 294,912 |
| Classification head (4 classes, no bias) | $4 \times 768$ | 3,072 |
| **Total trained** | $294{,}912 + 3{,}072$ | **297,984** |
| Share of GPT-2 | $297{,}984 \;/\; 124{,}442{,}880$ | 0.24% |

![Training loss](assets/training-curve.png)

### 3. Inference with the fine-tuned model

All model objects are deleted, the adapter is reloaded from `gpt2-lora-ag-news/` with `AutoPeftModelForSequenceClassification`, and the evaluation from step 1 is repeated on the same test set. Two details matter when reloading: `num_labels` has to be passed again, and `pad_token_id` has to be set again. Without either, a reloaded model scores differently from the one that was saved.

The notebook also merges the adapter into the base weights (`merge_and_unload`) and confirms the merged model gives identical predictions, so serving it costs nothing extra.

## Results

**Before vs after.** Accuracy goes from 27.0% to 89.1% (95% interval roughly 87% to 91%). The untrained model answers "World" for almost everything. After fine-tuning, Sports is nearly solved (98% recall) and most remaining errors sit between Business and Sci/Tech, where news about technology companies genuinely belongs to both.

**Comparing PEFT configurations.** To separate what the adapters contribute from what any trained head would give, I trained five configurations for one epoch each under identical settings:

![Accuracy against LoRA rank](assets/rank-sweep.png)

| Configuration | Trained parameters | Accuracy after 1 epoch |
|---|---:|---:|
| Head only (GPT-2 frozen) | 3,072 | 83.7% |
| LoRA r = 1 | 39,936 | 86.9% |
| LoRA r = 2 | 76,800 | 87.3% |
| LoRA r = 8 | 297,984 | 87.8% |
| LoRA r = 32 | 1,182,720 | 88.2% |

- The frozen GPT-2 features already separate topics well: training only the head reaches 83.7%.
- The first LoRA rank is worth more than the next thirty-one: rank 1 adds 3.2 points, and going from rank 1 to rank 32 adds only 1.3 more.
- Rank does not change training time (about 42 seconds for every rank, against 22 for head-only). The cost is the backward pass through the network, not the adapter size.
- With 1,000 test items each accuracy carries about ±2 points of sampling error, so neighbouring ranks cannot be told apart. For this task, rank 1 or 2 would do almost as well as 8 with a much smaller adapter.

## Limitations

- The "before" model has a randomly initialized head, so it is a floor, not a competitor. The head-only row (83.7%) is the fairer measure of what LoRA itself adds: about 3 to 5 points.
- One seed and a 4,000-item training subset. More data would raise every number.
- QLoRA (4-bit quantization with `bitsandbytes`) was not attempted: it needs a CUDA GPU, and this was developed on Apple Silicon.

## References

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Aghajanyan, Zettlemoyer & Gupta (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* [arXiv:2012.13255](https://arxiv.org/abs/2012.13255)
- Hugging Face PEFT: [LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)
- Dataset: [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news)
