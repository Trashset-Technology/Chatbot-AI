from django.shortcuts import render
from django.http import JsonResponse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .nltk_utils import bag_of_words, tokenize, stem
from .model import NeuralNet


class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def load_intents(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def train_chatbot(request):
    intents = load_intents("chatbot/intents.json")
    if intents is None:
        return JsonResponse({"error": "Intents file not found."}, status=404)

    all_words = []
    tags = []
    xy = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ["?", ".", "!"]
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train, y_train = [], []
    for pattern_sentence, tag in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001
    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)

    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    return JsonResponse({"message": "Training complete!", "file_saved": FILE})


# Load the model once when the server starts
model_data = torch.load("chatbot/data.pth")
model = NeuralNet(
    model_data["input_size"], model_data["hidden_size"], model_data["output_size"]
)
model.load_state_dict(model_data["model_state"])
model.eval()


def chat(request):
    user_input = request.GET.get("message")  # Get the message from the query parameters
    intents = load_intents("chatbot/intents.json")  # Load intents

    if intents is None:
        return JsonResponse({"error": "Intents file not found."}, status=404)

    # Tokenize and create bag of words for user input
    tokenized_input = tokenize(user_input)
    input_bow = bag_of_words(tokenized_input, model_data["all_words"])

    # Convert to tensor
    input_tensor = torch.tensor(input_bow).float().unsqueeze(0)

    # Get model predictions
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, dim=1)

    tag = model_data["tags"][predicted.item()]

    # Find the corresponding intent for the predicted tag
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = np.random.choice(intent["responses"])  # Choose a random response
            return JsonResponse({"response": response})

    return JsonResponse({"response": "I'm sorry, I didn't understand that."})
