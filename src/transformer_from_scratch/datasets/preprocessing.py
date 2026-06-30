from pathlib import Path

from datasets import load_from_disk

from transformer_from_scratch.datasets.tokenizer import (
    encode,
    bos_id,
    eos_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATASET_DIR = DATA_DIR / "raw" / "iwslt"
PROCESSED_DATASET_DIR = DATA_DIR / "processed" / "iwslt"

MAX_SEQ_LEN = 128


def preprocess(example):
    """
    Convert one translation example into Transformer inputs.
    """

    # English sentence
    src_text = example["translation"]["en"]

    # German sentence
    tgt_text = example["translation"]["de"]

    src_ids = encode(src_text)
    tgt_ids = encode(tgt_text)

    encoder_input = src_ids
    decoder_input = [bos_id()] + tgt_ids

    labels = tgt_ids + [eos_id()]

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "labels": labels,
    }


def filter_long_sequences(example):
    """
    Remove examples longer than MAX_SEQ_LEN.
    """

    return (
        len(example["encoder_input"]) <= MAX_SEQ_LEN
        and len(example["decoder_input"]) <= MAX_SEQ_LEN
        and len(example["labels"]) <= MAX_SEQ_LEN
    )


def preprocess_dataset():

    print("Loading dataset...")

    dataset = load_from_disk(str(RAW_DATASET_DIR))

    print(dataset)

    print("Tokenizing dataset...")

    processed = dataset.map(
        preprocess,
        remove_columns=dataset["train"].column_names, # type: ignore
    )

    print("Filtering long sequences...")

    processed = processed.filter(filter_long_sequences)

    PROCESSED_DATASET_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.save_to_disk(str(PROCESSED_DATASET_DIR))

    print("Processed dataset saved.")

    return processed


if __name__ == "__main__":

    dataset = preprocess_dataset()

    print(dataset)

    print(dataset["train"][0])