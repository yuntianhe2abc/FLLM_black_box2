import torch
def calculatePPL(sentence, model, tokenizer,device):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    return torch.exp(loss).item()

def write_PPL_TOP_1000(data_path):
    with open(data_path, 'w') as f:
        for index in largest_indices:
            f.write(f"{scores[j][k]}\n")
            f.write(f"{zlib_sentences[j]}\n")
            f.write(f"{train_data[j]}\n")
            f.write("\n")
    f.close()
