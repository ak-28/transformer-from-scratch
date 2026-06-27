from pathlib import Path
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

def load_iwslt():
    """
    Load the IWSLT 2017 English-German dataset.
    """
    dataset = load_dataset(
        "iwslt2017",
        "iwslt2017-en-de",
    )
    return dataset


def save_dataset(dataset, save_path: str):
    """
    Save the dataset to disk.
    """
    save_path = Path(save_path) # type: ignore
    save_path.parent.mkdir(parents=True, exist_ok=True) # type: ignore
    dataset.save_to_disk(str(save_path))


if __name__ == "__main__":
    dataset = load_iwslt()

    print(dataset)
    print(dataset["train"][0]) # type: ignore

    save_dataset(dataset, str(DATA_DIR/"iwslt"))