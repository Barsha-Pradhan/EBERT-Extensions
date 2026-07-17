# EBERT
This repository serves as the official code release of the paper [EBERT: Efficient BERT Inference with Dynamic Structured Pruning](https://aclanthology.org/2021.findings-acl.425/) (pubilished at Findings of ACL 2021).

<div align=center>
<img src=EBERT.png>
</div>

EBERT is a dynamic structured pruning algorithm for efficient BERT inference. Unlike previous methods that randomly prune the model weights for static inference, EBERT dynamically determines and prunes the unimportant heads in multi-head self-attention layers and the unimportant structured computations in feed-forward network for each input sample at run-time.

## Prerequisites

The code has the following dependencies:
* python >= 3.8.5
* pytorch >= 1.4.0
* transformers = 3.3.1
As transformers v3.3.1 has a bug when the evaluation strategy is `epoch`, you need to make the following changes in the transformers library:
```
--- a/src/transformers/training_args.py
+++ b/src/transformers/training_args.py
@@ -323,7 +323,7 @@ class TrainingArguments:
     def __post_init__(self):
         if self.disable_tqdm is None:
             self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN
-        if self.evaluate_during_training is not None:
+        if self.evaluate_during_training:
             self.evaluation_strategy = (
                 EvaluationStrategy.STEPS if self.evaluate_during_training else EvaluationStrategy.NO
             )
```

## Usages
We provide script files for training and validation in the `scripts` folder, and users can run these script from the repo root, e.g. `bash scripts/eval_glue.sh`.
In each scripts, there are several arguments to modify before running:
* `--data_dir`: path to dataset：[GLUE](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e), [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/). 
* `MODEL_PATH` or `--model_name_or_path`: path to trained model folder
* `TASK_NAME`: task name in GLUE (SST-2, MNLI, ...)
* `RUN_NAME`: name of the current experiment, which influence the save path and log name for wandb.
* other hyper-parameters, e.g., `head_mask_mode`

You can download the original pretrained model of [BERT](https://huggingface.co/bert-base-uncased) and [RoBERTa](https://huggingface.co/roberta-base) from HuggingFace. 

## ELECTRA extension (not in the official release)
This fork extends the official BERT/RoBERTa implementation to **ELECTRA-base**, since the upstream repository only ships `modeling_ebert.py` (BERT) and `modeling_eroberta.py` (RoBERTa).

* `models/modeling_eelectra.py` is derived from `transformers==3.3.1`'s `modeling_electra.py`, using the exact same transformation the authors applied to go from BERT to RoBERTa:
  * `ElectraSelfAttention` gets a `predictor` MLP + `GumbelSoftmax` (or top-k `Binarize_STE`) gate that dynamically prunes attention heads per-sample, threaded with a `layer_id` (layer 0 uses mean-pooled hidden states, deeper layers use the `[CLS]` token) exactly like `BertSelfAttention`/`RobertaSelfAttention`.
  * `ElectraIntermediate` gets the analogous dynamic FFN-neuron mask.
  * `ElectraLayer`/`ElectraEncoder`/`ElectraModel` propagate `(dynamic_head_masks, dynamic_ffn_masks)` through the forward pass the same way `BertLayer`/`BertEncoder`/`BertModel` do.
  * `ElectraForSequenceClassification` adds the same sparsity-regularized loss term (`loss_lambda`, `sparsity_target`) as `BertForSequenceClassification`, and unpacks/repacks the two extra tuple elements in every other `Electra*` head (`ElectraForPreTraining`, `ElectraForMaskedLM`, `ElectraForTokenClassification`, `ElectraForQuestionAnswering`, `ElectraForMultipleChoice`) so the module still imports and runs even though only sequence classification is wired into `run_dy_glue.py`.
  * `custom_utils/custom_trainer.py` and `custom_utils/cutils.py` needed **no changes** — the predictor-parameter-group optimizer and sparsity/FLOPs accounting are both name- and buffer-based (`"predictor" in n`, `head_nopruned_times`, `ffn_nopruned_times`), so they pick up ELECTRA's submodules automatically.
* `run_dy_glue.py`'s `MODELS` dict now also maps `'electra': modeling_eelectra.ElectraForSequenceClassification`, keyed off `config.model_type` from `AutoConfig`, so `--model_name_or_path` pointing at an ELECTRA checkpoint routes correctly.
* `run_glue.py` (the plain fine-tuning baseline, no dynamic masking) needed no changes since it already uses `AutoModelForSequenceClassification`, which supports ELECTRA natively.
* `scripts/train_electra_glue.sh` / `scripts/train_eelectra_glue.sh` add ELECTRA-base training scripts for the baseline fine-tune (stage 1) and the EBERT dynamic-masking fine-tune (stage 2), configured for: seed 42, AdamW, weight decay 0.01, batch size 16, learning rate 2e-5, 10% linear warmup, linear LR decay, 5 epochs, max sequence length 128, CUDA.
  * `transformers==3.3.1`'s `TrainingArguments` only exposes `--warmup_steps` (an absolute count), not `--warmup_ratio`, so `scripts/compute_warmup_steps.py` converts the 10% ratio into a step count per GLUE task before the scripts launch training.
* You can download the original pretrained [ELECTRA-base discriminator](https://huggingface.co/google/electra-base-discriminator) from HuggingFace, the same way as BERT/RoBERTa above.

**Verified**: `ElectraForSequenceClassification` was smoke-tested end-to-end (forward + backward, both `gumbel` and `topk` head/FFN mask modes, in both train and eval mode) with a small randomly-initialized config, and the `head_nopruned_times`/`ffn_nopruned_times` sparsity buffers were confirmed to populate correctly during a dummy forward pass, matching the reporting mechanism `custom_utils/cutils.py` relies on. Full-scale GLUE fine-tuning with real ELECTRA-base weights and data has not been run here — that step needs your lab's GPU, GLUE data, and pretrained checkpoint.

## DistilBERT extension (not in the official release)
This fork also extends the official implementation to **DistilBERT-base**. DistilBERT's architecture differs more from BERT than RoBERTa/ELECTRA do, so the port isn't a pure copy-paste of the BERT diff — the underlying idea (predictor MLP + Gumbel/top-k gate, applied to attention weights and FFN activations) is the same, but it had to be re-attached to DistilBERT's own module shapes:

* `models/modeling_edistilbert.py` is derived from `transformers==3.3.1`'s `modeling_distilbert.py`. Structural differences from BERT that the port accounts for:
  * DistilBERT has **no separate `SelfOutput`/`Intermediate`/`Output` submodules** — `MultiHeadSelfAttention` computes attention end-to-end (including the output projection) and `FFN` computes the whole feed-forward block (including both linear layers) in one class each, with the residual + LayerNorm living in `TransformerBlock` itself. The dynamic head mask is therefore applied directly inside `MultiHeadSelfAttention.forward` (mirroring where `BertSelfAttention` applies it), and the dynamic FFN mask is applied inside `FFN.ff_chunk`, right after the first linear layer and before the activation (mirroring `BertIntermediate`).
  * DistilBERT layers were originally built once and `copy.deepcopy`'d; since each layer's predictor now needs to know its own `layer_id` (layer 0 pools over the sequence, deeper layers use the `[CLS]` position — same convention as BERT/RoBERTa/ELECTRA), `Transformer.__init__` now instantiates each `TransformerBlock(config, i)` directly instead of deep-copying a single instance.
  * DistilBERT's own attention mask is a raw `(bs, seq_length)` padding mask consumed inside `MultiHeadSelfAttention`, not a precomputed additive mask like BERT/RoBERTa/ELECTRA — left untouched; the dynamic head mask is multiplied in as a second, independent gate after the existing static `head_mask` step.
  * `DistilBertConfig` names its dimensions differently (`dim`, `n_heads`, `hidden_dim`, `n_layers` instead of `hidden_size`, `num_attention_heads`, `intermediate_size`, `num_hidden_layers` — though `DistilBertConfig` does already expose the first three as read-only properties). Since `custom_utils/cutils.py`'s FLOPs/sparsity-loss helpers and the predictor's `BatchNorm1d` eps were written once against BERT's naming, a small `add_ebert_config_aliases()` helper sets the two missing aliases (`intermediate_size`, `layer_norm_eps`) onto the config the first time a DistilBERT model is built, rather than duplicating that logic per architecture.
  * `DistilBertForSequenceClassification` gets the same sparsity-regularized loss term as `BertForSequenceClassification`, and the other DistilBERT heads (`ForMaskedLM`, `ForQuestionAnswering`, `ForTokenClassification`, `ForMultipleChoice`) have their output-tuple slicing patched the same way as ELECTRA's, so the module still imports and runs even though only sequence classification is wired into `run_dy_glue.py`.
  * As with ELECTRA, `custom_utils/custom_trainer.py` and `cutils.py` needed no changes — same name-based (`"predictor" in n`) and buffer-based (`head_nopruned_times`, `ffn_nopruned_times`) mechanisms pick DistilBERT up automatically.
  * No special handling was needed for GLUE's `token_type_ids`: `DistilBertTokenizer.model_input_names` doesn't include `token_type_ids`, so `glue_convert_examples_to_features` already omits it for DistilBERT the same way it does in vanilla `transformers` — `run_dy_glue.py`/`GlueDataset` need no DistilBERT-specific branch.
* `run_dy_glue.py`'s `MODELS` dict now also maps `'distilbert': modeling_edistilbert.DistilBertForSequenceClassification`.
* `scripts/train_distilbert_glue.sh` / `scripts/train_edistilbert_glue.sh` add DistilBERT-base training scripts (baseline stage 1, EBERT dynamic-masking stage 2), same config as ELECTRA above: seed 42, AdamW, weight decay 0.01, batch size 16, learning rate 2e-5, 10% linear warmup, linear LR decay, 5 epochs, max sequence length 128, CUDA.
* You can download the original pretrained [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased) from HuggingFace, the same way as BERT/RoBERTa/ELECTRA above.

**Verified**: `DistilBertForSequenceClassification` was smoke-tested the same way as ELECTRA above — forward + backward in both `gumbel` and `topk` modes, train and eval mode, plus confirmation that `head_nopruned_times`/`ffn_nopruned_times` populate correctly. Full-scale GLUE fine-tuning with real DistilBERT-base weights has not been run here.

## Citation
If you found the library useful for your work, please kindly cite our work:
```
@inproceedings{liu-etal-2021-ebert,
    title = "{EBERT}: Efficient {BERT} Inference with Dynamic Structured Pruning",
    author = "Liu, Zejian  and
              Li, Fanrong  and
              Li, Gang  and
              Cheng, Jian",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.425",
    doi = "10.18653/v1/2021.findings-acl.425",
    pages = "4814--4823",
}
```
