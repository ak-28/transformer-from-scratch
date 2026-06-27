import sentencepiece as spm
from pathlib import Path
from datasets import load_from_disk
from typing import List, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PREFIX = DATA_DIR / "tokenizer"/ "iwslt"
MODEL_PATH = DATA_DIR / "tokenizer" / "iwslt.model"

def sentence_iterator(dataset):
    """Generator to yield sentences from the training split for both languages."""
    for example in dataset["train"]:
        yield example['translation']['en']
        yield example['translation']['de']

def train_tokenizer(
    vocab_size=16000,
    model_prefix=str(MODEL_PREFIX)
    ):
    
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading local Apache Arrow datasets...")
    dataset = load_from_disk(str(DATA_DIR / "raw" / "iwslt"))
    print(f"Dataset loaded successfully: {dataset}")

    print("Starting SentencePiece BPE Tokenizer training...")
    spm.SentencePieceTrainer.train(  # type: ignore
        sentence_iterator=sentence_iterator(dataset),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",   
        pad_id=0,                 
        unk_id=1,
        bos_id=2,
        eos_id=3,
        byte_fallback=True 
    )
    print(f"Tokenizer saved to {model_prefix}.model")
    print(f"Vocabulary saved to {model_prefix}.vocab")

_SP_PROCESSOR = None

def load_tokenizer(model_path: str = str(MODEL_PATH)):
    """
    Initializes and caches the SentencePiece processor.
    """
    global _SP_PROCESSOR
    if _SP_PROCESSOR is None:
        _SP_PROCESSOR = spm.SentencePieceProcessor(model_file=model_path) # type: ignore
    return _SP_PROCESSOR


def encode(text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
    """
    Converts a single raw sentence or a list of sentences into token IDs.
    
    Args:
        text: A single string or a list of strings to tokenize.
        
    Returns:
        A list of integer IDs (or a nested list of lists if batch input).
    """
    sp = load_tokenizer()
    return sp.encode(text, out_type=int) # type: ignore


def decode(ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
    """
    Converts a list of token IDs (or a batch of lists) back into human-readable text.
    
    Args:
        ids: A list of integer IDs or a list of lists of integer IDs.
        
    Returns:
        The decoded text string or a list of decoded strings.
    """
    sp = load_tokenizer()
    return sp.decode(ids) # type: ignore

def pad_id() -> int:
    return load_tokenizer().pad_id()


def unk_id() -> int:
    return load_tokenizer().unk_id()


def bos_id() -> int:
    return load_tokenizer().bos_id()


def eos_id() -> int:
    return load_tokenizer().eos_id()


def vocab_size() -> int:
    return load_tokenizer().vocab_size()

def piece_to_id(piece: str):
    return load_tokenizer().piece_to_id(piece)  # type: ignore


def id_to_piece(idx: int):
    return load_tokenizer().id_to_piece(idx) # type: ignore

if __name__ == "__main__":
    train_tokenizer()
    assert MODEL_PATH.exists(), "Tokenizer model was not created."
    
    texts = [
        "I love transformers.",
        "Deep learning is fun."
    ]

    ids = encode(texts)

    print(ids)
    print(decode(ids))