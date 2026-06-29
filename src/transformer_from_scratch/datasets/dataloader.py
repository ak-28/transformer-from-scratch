from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATASET_DIR = DATA_DIR / "processed" / "iwslt"


class TranslationDataset(Dataset):

    def __init__(self, split:str = 'train', device:str = 'cpu'):
        dataset = load_from_disk(str(PROCESSED_DATASET_DIR))
        self.dataset = dataset[split]
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample =  self.dataset[idx]
        
        return {
            "encoder_input": torch.tensor(
                sample["encoder_input"],
                dtype=torch.long,
                device=self.device
            ),
            "decoder_input": torch.tensor(
                sample["decoder_input"],
                dtype=torch.long,
                device=self.device,
            ),
            "labels": torch.tensor(
                sample["labels"],
                dtype=torch.long,
                device=self.device
            )
        }
        
def get_dataloader(
    split="train",
    batch_size=1,
    shuffle=True,
    device='cpu'
    ):
    dataset = TranslationDataset(split=split, device=device)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    
if __name__=="__main__":

    loader = get_dataloader(
        split="train",
        batch_size=1,
    )

    batch = next(iter(loader))

    print(batch["encoder_input"].shape)
    print(batch["decoder_input"].shape)
    print(batch["labels"].shape)

    print(batch)