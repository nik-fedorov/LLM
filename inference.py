import torch


def generate(model, prompt, tokenizer, max_len, device):
    encoded = [tokenizer.bos_id] + tokenizer.encode(prompt)
    encoded = torch.tensor(encoded).unsqueeze(0).to(device)

    while encoded[0, -1] != tokenizer.eos_id and encoded.size(1) < max_len:
        pred = model(encoded, torch.tensor([encoded.size(1)]).to(device))['logits']

        new_tokens = torch.argmax(pred[:, -1], dim=1)
        encoded = torch.cat((encoded, new_tokens.unsqueeze(-1)), dim=1)

    return tokenizer.decode(encoded.cpu().tolist()[0])
