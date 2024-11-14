from oat.exploration import CompletionDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import vllm
import torch

class Args:
    def __init__(self):
        self.apply_chat_template = False
        self.prompt_max_length = 10
        self.train_batch_size_per_device = 2


def test_dataset():
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

def test_vllm():

    prompt = 'Daman is a'

    llm = vllm.LLM(**{
        "model": 'trl-lib/pythia-1b-deduped-tldr-sft',
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "dtype": "bfloat16",
        "enable_prefix_caching": False,
        "max_model_len": 1024,
    })
    vllm_response = llm.generate(
        prompts=[prompt], 
        sampling_params=vllm.SamplingParams(
            temperature=1,
            top_p=1.0,
            top_k=-1,
            max_tokens=5,
            n=5,
            logprobs=1
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        'trl-lib/pythia-1b-deduped-tldr-sft',
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0'
    )

    tokenizer = AutoTokenizer.from_pretrained('trl-lib/pythia-1b-deduped-tldr-sft')
    inp = tokenizer.encode_plus(prompt + vllm_response[0].outputs[0].text, return_tensors="pt")
    input_ids, attention_mask = inp['input_ids'].to(model.device), inp['attention_mask'].to(model.device)
    output = model(input_ids, attention_mask=attention_mask)
    logits = output['logits']

    # Start from the first position
    labels = input_ids[:, 1:].clone()
    # labels = input_ids.clone()
    logits = logits[:, :-1, :].float()

    loss_masks = attention_mask.clone().bool()
    # mask prompts
    source_len = len(tokenizer.encode(prompt))
    for mask in loss_masks:
        mask[:source_len] = False
    loss_masks = loss_masks[:, 1:]

    # dummy token; we'll ignore the losses on these tokens later
    labels[loss_masks == False] = 0
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    logp = (per_token_logps * loss_masks).sum(-1)
    print(logp.cpu().item(), vllm_response[0].outputs[0].cumulative_logprob)

test_vllm()