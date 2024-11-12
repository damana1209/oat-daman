from oat.exploration import CompletionDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class Args:
    def __init__(self):
        self.apply_chat_template = False
        self.prompt_max_length = 10
        self.train_batch_size_per_device = 2

args = Args()
dataset = CompletionDataset(
    prompt="Daman",
    completions=[" is a good boy", " is a bad boy but likes cats."],
    tokenizer=AutoTokenizer.from_pretrained('trl-lib/pythia-1b-deduped-tldr-sft'),
    args=args
)

dataloader = DataLoader(
    dataset,
    batch_size=args.train_batch_size_per_device,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)
print(len(dataloader))

for data in dataloader:
    print(data)