from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import nltk
import numpy as np
import pickle


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


def get_parameters(model):
    """
    Get the parameters of a model.
    Args:
        model: A neural network model with parameters.

    Returns:
        A deep copy list of parameters of the model.
    """
    return [val.clone().detach().cpu().numpy() for _, val in model.state_dict().items()]


def initialise_client_parameters(server_parameters, num_of_clients):
    """
    Initialise the parameters of the clients.
    Args:
        server_parameters: The parameters of the server model.
        num_of_clients: The number of clients.

    Returns:
        A list of parameters for the clients.
    """
    client_parameters = []
    for i in range(num_of_clients):
        client_parameters.append([np.copy(p) for p in server_parameters])
    return client_parameters


def set_parameters(model, parameters):
    """
    Set the parameters of a model.
    Args:
        model: A neural network models with parameters.
        parameters: A list of parameters for the model.

    Returns:
        The model with the new parameters.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def train_epoch(model, train_loader, device):
    """
    Train a client model on local data for a complete epoch.
    Args:
        model: The model to be trained
        train_loader: The training data loader
        device: The device to run the model on

    Returns:
        The average training loss
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    train_loop = tqdm(train_loader, leave=False)
    epoch_train_loss = 0
    for batch in train_loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        train_loss = train_outputs.loss
        train_loss.backward()
        optimizer.step()
        epoch_train_loss += train_loss.item()
        train_loop.set_description(f"Training loss: {train_loss.item()}")
    average_epoch_loss = epoch_train_loss / len(train_loop)
    print(f"Epoch average training loss: {average_epoch_loss}")
    return average_epoch_loss


def train_batch(model, batch, device):
    """
    Train a client model on local data for a single batch.
    Args:
        model: The model to be trained
        batch: The training batch
        device: The device to run the model on

    Returns:
        The batch training loss and the attention scores
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True,
                          output_hidden_states=True)
    attentions = train_outputs.attentions
    batch_loss = train_outputs.loss
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item(), attentions, train_outputs.hidden_states, train_outputs.logits


def test(model, test_loader, device):
    """
    Test the server model after aggregation.
    Args:
        model: The server model to be tested
        test_loader: The testing data loader

    Returns:
        The average testing loss
    """
    model.eval()
    test_loop = tqdm(test_loader, leave=False)
    epoch_test_loss = 0
    with torch.no_grad():
        for batch in test_loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            test_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            test_loss = test_outputs.loss
            epoch_test_loss += test_loss.item()
            test_loop.set_description(f"Test loss: {test_loss.item()}")
        average_epoch_loss = epoch_test_loss / len(test_loop)
        print(f"Epoch Average test loss: {average_epoch_loss}")
    return average_epoch_loss


def test_batch(model, batch, device):
    """
    Test the model with batch data.
    Args:
        model: The model to be tested
        batch: The batch data
        device: The device to run the model on

    Returns:
        The batch testing loss and the attention scores
    """
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    test_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                         output_attentions=True)
    test_loss = test_outputs.loss
    return test_loss.item(), test_outputs.attentions


def aggregate_parameters(server, client_parameters):
    aggregated_parameters = []
    for param in get_parameters(server):
        aggregated_parameters.append(torch.zeros(param.shape))

    for j in range(len(client_parameters)):
        single_client_parameter = client_parameters[j]
        for k, param in enumerate(single_client_parameter):
            aggregated_parameters[k] += torch.Tensor(param)

    for j in range(len(aggregated_parameters)):
        aggregated_parameters[j] /= len(client_parameters)
    return aggregated_parameters


def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.
    Args:
        vector_a: A vector.
        vector_b: Another vector.

    Returns:
        The cosine similarity between the two vectors.
    """
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))


def generate2(model, tokenizer, device, seq_len, batch_size, num_samples, prompt, top_k=50, temperature=0.8):
    samples = []
    num_batches = int(np.ceil(num_samples / batch_size))
    with tqdm(total=num_samples) as pbar:
        for i in range(num_batches):
            prompts = [prompt] * batch_size
            input_len = 1
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            output_sequences = model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True,
                # top_p=1,
                top_k=top_k,
                temperature=temperature

            )
            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            for text in texts:
                samples.append(text)
            pbar.update(batch_size)
    return samples


def encode(tokenizer, sentences):
    encodings = []
    for a in sentences:
        encoding = tokenizer(a, return_tensors="pt", padding=False, truncation=True, max_length=128)
        encodings.append(encoding["input_ids"][0].tolist())
    return encodings


def get_k_token_set(sentence_encoding, k=3):
    k_tokens_list = []
    n = len(sentence_encoding)
    for i in range(n - k + 1):
        k_tokens = sentence_encoding[i:i + k]
        k_tokens_list.append(k_tokens)
    return k_tokens_list


def load_clients_train_data(clients, folder_path):
    clients_data = []
    start, end = clients
    for i in range(start, end + 1):
        file_name = f"client_{i}.txt"
        file_path = folder_path + file_name
        file = open(file_path, 'r')
        a = file.readlines()
        clients_data.extend(a)
    return clients_data


def read_perplexity_ranking(file_path):
    sentences = []
    file = open(file_path, 'r')
    result1 = file.readlines()
    for i in range(1000):
        sentences.append(result1[i * 3 + 2])
    return sentences


def get_intersection(k_set1, k_set2):
    """
    given k-tokens-set for two sentences, count how many k-tokens set do they have in common
    Args:
    Returns:
        Count of common and unique k-token-set
    """
    common_list = []
    count = 0
    for a in k_set1:
        for b in k_set2:
            if a == b:
                if a not in common_list:
                    count += 1
                    common_list.append(a)

    return count, common_list


def calculate_scores(generated_data, email_data, tokenizer, k=3):
    """
    given k-tokens-set for two sentences, count how many k-tokens set do they have in common
    Args:
    Returns:
        scores np array
    """
    generated_encodings = encode(tokenizer, generated_data)
    email_encodings = encode(tokenizer, email_data)
    print("finish encoding")
    m = len(generated_encodings)
    n = len(email_encodings)
    count = 0
    scores = np.zeros((m, n), dtype=float)
    for i in range(m):
        count += 1
        if count % 40 == 0:
            print(count)
        x = generated_encodings[i]
        k_set1 = get_k_token_set(x)
        for j in range(n):
            y = email_encodings[j]
            k_set2 = get_k_token_set(y)
            score, _ = get_intersection(k_set1, k_set2)
            scores[i][j] = score
    return scores


def write_ground_truth(scores, generated_data, email_data, path, top_n=1000):
    scores_flatten = np.ravel(scores)
    sorted_indices = np.argsort(scores_flatten)
    reverse_sorted_indices = sorted_indices[::-1]
    largest_indices = reverse_sorted_indices[:top_n]
    with open(path, 'w') as f:
        for index in largest_indices:
            j, k = np.unravel_index(index, scores.shape)
            f.write(f"{scores[j][k]}\n")
            f.write(f"{generated_data[j]}\n")
            f.write(f"{email_data[j]}\n")
            f.write("\n")
    f.close()


def load_all_train_data(path):
    file = open(path, 'r')
    file_data = file.readlines()
    file.close()
    return file_data


def search_train_data(string, all_train_data):
    count = 0
    for x in all_train_data:
        if string in x:
            count += 1
    return count


def score_statistics(scores, store_path):
    score_count = {}
    scores_flatten = np.ravel(scores)
    sorted_indices = np.argsort(scores_flatten)
    reverse_sorted_indices = sorted_indices[::-1]
    for index in reverse_sorted_indices:
        j, k = np.unravel_index(index, scores.shape)
        score = scores[j][k]
        score_count[score] = score_count.get(score, 0) + 1
    with open(store_path, 'wb') as fp:
        pickle.dump(score_count, fp)
        print('Score statistics saved successfully!')
    return score_count


def write_top_scores_dict(scores, store_path, top_n=2000):
    top_scores = {}
    common_k_set=[]
    scores_flatten = np.ravel(scores)
    sorted_indices = np.argsort(scores_flatten)
    reverse_sorted_indices = sorted_indices[::-1]
    count=0
    for index in reverse_sorted_indices:
        if count<top_n:

            j, k = np.unravel_index(index, scores.shape)

            key = (j, k)
            top_scores[key] = scores[j][k]

    with open(store_path, 'wb') as fp:
        pickle.dump(top_scores, fp)
        print('dictionary saved successfully to file')
