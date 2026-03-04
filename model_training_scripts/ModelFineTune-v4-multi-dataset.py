"""
mBART fine-tuning (multi-dataset) as a standalone script.

Supports:
- NVIDIA CUDA
- AMD ROCm (appears as torch cuda backend)
- Optional DirectML on Windows AMD GPUs (requires torch-directml)

Background run on Windows PowerShell:
    Start-Process -FilePath ".venv\Scripts\python.exe" `
      -ArgumentList "model_training_scripts/ModelFineTune-v4-multi-dataset.py --hf-token hf_xxx" `
      -RedirectStandardOutput "logs/finetune_v4.out.log" `
      -RedirectStandardError "logs/finetune_v4.err.log"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login, whoami
from huggingface_hub.utils import HfHubHTTPError
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


DEFAULT_DATASET_IDS = [
    # "SPEAK-PP/sinhala-spelling-correction-already-corrected-pairs",
    "SPEAK-PP/openslr-sinhala-spelling-correction-prediction-reference",
    # "SPEAK-PP/sinhala-itn-dataset",
]

TARGET_FEATURES = Features(
    {
        "dyslexic_sentence": Value("large_string"),
        "clean_sentence": Value("large_string"),
    }
)

load_dotenv()

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune mBART on multiple Sinhala spelling datasets"
    )
    parser.add_argument("--model-name", default="facebook/mbart-large-50")
    parser.add_argument("--dataset-ids", nargs="+", default=DEFAULT_DATASET_IDS)
    parser.add_argument("--source-lang", default="si_LK")
    parser.add_argument("--target-lang", default="si_LK")
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/mbart-model-v4-multi")
    parser.add_argument(
        "--hub-model-id", default="SPEAK-PP/mBART-large-50-si-spelling-v4-multi"
    )
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--prefer-directml", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--log-dir", default="logs")
    return parser.parse_args()


def setup_logging(log_dir: str | Path) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"finetune_v4_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger.info("Logging to: %s", log_file)


def select_device(prefer_directml: bool = False):
    if prefer_directml:
        try:
            import torch_directml

            device = torch_directml.device()
            logger.info("Device backend: DirectML")
            return device, "directml"
        except Exception as exc:
            logger.warning("DirectML not available (%s). Falling back to torch backend.", exc)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        hip_ver = getattr(torch.version, "hip", None)
        cuda_ver = getattr(torch.version, "cuda", None)
        if hip_ver:
            logger.info("Device backend: ROCm | GPU: %s | HIP: %s", device_name, hip_ver)
        else:
            logger.info("Device backend: CUDA | GPU: %s | CUDA: %s", device_name, cuda_ver)
        return device, "cuda"

    logger.info("Device backend: CPU")
    return torch.device("cpu"), "cpu"


def login_hf(hf_token: str) -> bool:
    token = hf_token.strip() or os.environ.get("HF_TOKEN", "").strip()
    if not token:
        logger.warning("HF token not found. Set --hf-token or HF_TOKEN env var.")
        return False

    try:
        print("token:", token)
        login(token=token, add_to_git_credential=True)
        # user_info = whoami()
        # logger.info("Hugging Face logged in as: %s", user_info.get('name', 'unknown'))
        logger.info("Hugging Face authentication successful.")
        return True
    except HfHubHTTPError as exc:
        logger.error("HF authentication failed: %s", exc)
        return False


def get_eval_split(ds_dict: DatasetDict) -> Dataset | None:
    if "eval" in ds_dict:
        return ds_dict["eval"]
    if "validation" in ds_dict:
        return ds_dict["validation"]
    return None


def normalize_pair_columns(ds_split: Dataset) -> Dataset:
    cols = ds_split.column_names

    src_col = (
        "dyslexic_sentence"
        if "dyslexic_sentence" in cols
        else "input_text"
        if "input_text" in cols
        else "textual_format"
        if "textual_format" in cols
        else None
    )

    tgt_col = (
        "clean_sentence"
        if "clean_sentence" in cols
        else "corrected_text"
        if "corrected_text" in cols
        else "numerical_format"
        if "numerical_format" in cols
        else None
    )

    if src_col is None or tgt_col is None:
        raise ValueError(f"Could not map columns for split. Found: {cols}")

    if src_col != "dyslexic_sentence":
        ds_split = ds_split.rename_column(src_col, "dyslexic_sentence")
    if tgt_col != "clean_sentence":
        ds_split = ds_split.rename_column(tgt_col, "clean_sentence")

    keep_cols = ["dyslexic_sentence", "clean_sentence"]
    drop_cols = [col for col in ds_split.column_names if col not in keep_cols]
    if drop_cols:
        ds_split = ds_split.remove_columns(drop_cols)

    return ds_split.cast(TARGET_FEATURES)


def load_and_merge_datasets(dataset_ids: list[str], seed: int) -> DatasetDict:
    logger.info("[STEP 1] Loading + combining datasets")

    train_parts, eval_parts, test_parts = [], [], []

    for dataset_id in dataset_ids:
        ds = load_dataset(dataset_id)
        logger.info("Loaded %s splits: %s", dataset_id, list(ds.keys()))

        if "train" in ds:
            train_parts.append(normalize_pair_columns(ds["train"]))
        if "test" in ds:
            test_parts.append(normalize_pair_columns(ds["test"]))

        eval_split = get_eval_split(ds)
        if eval_split is not None:
            eval_parts.append(normalize_pair_columns(eval_split))

    if not train_parts or not test_parts:
        raise ValueError("Missing required splits. Need at least train and test across datasets.")

    train_merged = concatenate_datasets(train_parts)
    test_merged = concatenate_datasets(test_parts)

    if eval_parts:
        eval_merged = concatenate_datasets(eval_parts)
    else:
        split = train_merged.train_test_split(test_size=0.1, seed=seed)
        train_merged = split["train"]
        eval_merged = split["test"]

    merged = DatasetDict(
        {
            "train": train_merged.shuffle(seed=seed),
            "eval": eval_merged.shuffle(seed=seed),
            "test": test_merged.shuffle(seed=seed),
        }
    )

    logger.info("✓ Combined splits:")
    for split_name in ["train", "eval", "test"]:
        logger.info("  %s: %d", split_name, len(merged[split_name]))
    logger.info("Columns: %s", merged["train"].column_names)

    return merged


def preprocess_and_tokenize(dataset: DatasetDict, tokenizer, args) -> tuple[Dataset, Dataset, Dataset]:
    logger.info("[STEP 2] Tokenizing & preparing datasets...")

    def preprocess_function(examples):
        input_texts, target_texts = [], []

        for source_text, target_text in zip(
            examples["dyslexic_sentence"], examples["clean_sentence"]
        ):
            if (
                source_text
                and target_text
                and str(source_text).strip()
                and str(target_text).strip()
            ):
                input_texts.append(str(source_text))
                target_texts.append(str(target_text))

        if not input_texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        model_inputs = tokenizer(
            input_texts,
            max_length=args.max_input_length,
            padding="max_length",
            truncation=True,
        )

        labels = tokenizer(
            text_target=target_texts,
            max_length=args.max_target_length,
            padding="max_length",
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_columns = dataset["train"].column_names
    dataset_tok = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=100,
        remove_columns=remove_columns,
        desc="Tokenizing",
    )

    def keep_nonempty(ex):
        return (
            ex["input_ids"] is not None
            and len(ex["input_ids"]) > 0
            and ex["labels"] is not None
            and len(ex["labels"]) > 0
        )

    dataset_tok = dataset_tok.filter(keep_nonempty)

    train_dataset = dataset_tok["train"]
    eval_dataset = dataset_tok["eval"]
    test_dataset = dataset_tok["test"]

    logger.info("✓ Tokenized split sizes:")
    logger.info("  Train: %d", len(train_dataset))
    logger.info("  Eval : %d", len(eval_dataset))
    logger.info("  Test : %d", len(test_dataset))

    return train_dataset, eval_dataset, test_dataset


def get_precision_flags(device, backend: str) -> tuple[bool, bool]:
    if backend != "cuda":
        return False, False

    try:
        device_name = torch.cuda.get_device_name(0).lower()
        is_nvidia = "nvidia" in device_name
        if not is_nvidia:
            return False, False

        major_capability = torch.cuda.get_device_capability(0)[0]
        use_bf16 = major_capability >= 8
        use_fp16 = not use_bf16
        return use_fp16, use_bf16
    except Exception:
        return False, False


def build_compute_metrics(tokenizer):
    sacrebleu = evaluate.load("sacrebleu")

    def postprocess_text(predictions, labels):
        predictions = [text.strip() for text in predictions]
        labels = [[text.strip()] for text in labels]
        return predictions, labels

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_predictions, decoded_labels = postprocess_text(
            decoded_predictions, decoded_labels
        )

        bleu = sacrebleu.compute(
            predictions=decoded_predictions, references=decoded_labels
        )["score"]
        exact = np.mean(
            [pred == label[0] for pred, label in zip(decoded_predictions, decoded_labels)]
        )

        return {"bleu": bleu, "exact_match": float(exact)}

    return compute_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    setup_logging(args.log_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, backend = select_device(prefer_directml=args.prefer_directml)

    hf_logged_in = login_hf(args.hf_token)

    use_wandb = args.use_wandb
    if use_wandb and not os.environ.get("WANDB_API_KEY", "").strip():
        logger.warning("WANDB_API_KEY not found. Disabling W&B logging.")
        use_wandb = False

    dataset = load_and_merge_datasets(args.dataset_ids, seed=args.seed)

    logger.info("[STEP 3] Loading tokenizer + model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.src_lang = args.source_lang
    tokenizer.tgt_lang = args.target_lang

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    lang_token_id = tokenizer.convert_tokens_to_ids([args.target_lang])[0]
    model.config.decoder_start_token_id = lang_token_id
    model = model.to(device)

    train_dataset, eval_dataset, test_dataset = preprocess_and_tokenize(
        dataset, tokenizer, args
    )

    compute_metrics = build_compute_metrics(tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_fp16, use_bf16 = get_precision_flags(device, backend)
    report_to = ["wandb"] if use_wandb else []

    logger.info("Precision settings -> fp16: %s, bf16: %s", use_fp16, use_bf16)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        # eval_steps=args.eval_steps,
        save_strategy="epoch",
        # save_steps=args.save_steps,
        save_total_limit=3,
        push_to_hub=hf_logged_in,
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        fp16=use_fp16,
        bf16=use_bf16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        report_to=report_to,
        dataloader_num_workers=4,
        dataloader_pin_memory=(backend == "cuda"),
        remove_unused_columns=True,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )

    try:
        trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=tokenizer)

    logger.info("[STEP 4] Training...")
    train_result = trainer.train()
    logger.info("✓ Training done")
    logger.info("Training loss: %s", train_result.training_loss)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Saved to: %s", output_dir)

    logger.info("[STEP 5] Testing...")
    prediction_output = trainer.predict(test_dataset)
    decoded_predictions = tokenizer.batch_decode(
        prediction_output.predictions, skip_special_tokens=True
    )

    labels = np.where(
        prediction_output.label_ids != -100,
        prediction_output.label_ids,
        tokenizer.pad_token_id,
    )
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact = np.mean(
        [pred.strip() == label.strip() for pred, label in zip(decoded_predictions, decoded_labels)]
    )
    logger.info("Exact match (test): %.2f %%", round(float(exact) * 100, 2))

    if hf_logged_in:
        logger.info("Pushing to hub: %s", args.hub_model_id)
        try:
            trainer.push_to_hub(
                language="si",
                finetuned_from=args.model_name,
                model_name=args.hub_model_id,
                dataset=", ".join(args.dataset_ids),
            )
            logger.info("✓ Pushed")
        except HfHubHTTPError as exc:
            logger.error("✗ Push failed with Hugging Face API error.")
            logger.error("Reason: %s", exc)
    else:
        logger.warning("Skipping push_to_hub (not logged in).")


if __name__ == "__main__":
    main()
