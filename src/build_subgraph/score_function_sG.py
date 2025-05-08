import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import random
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from src.utils import load_all_graphs
from ..config.config import SUBGRAPH_CONFIG


# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ============================
# 1. Data Preparation
# ============================

class KGLinkPredictionDataset(Dataset):
    """
    Dataset for Link Prediction in Knowledge Graphs.
    Each sample consists of a triple represented as a string and a label indicating its validity.
    """
    def __init__(self, triples, labels, tokenizer, max_length=512):
        self.triples = triples
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        triple = self.triples[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            triple,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),  # Tensor of shape (max_length)
            'attention_mask': encoding['attention_mask'].flatten(),  # Tensor of shape (max_length)
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(G: nx.DiGraph, tokenizer, negative_ratio=1.0, max_samples=None):
    """
    Prepare positive and negative samples for link prediction.

    Parameters:
    - G (nx.DiGraph): The raw knowledge graph.
    - tokenizer: The tokenizer to convert triples into text.
    - negative_ratio (float): Ratio of negative to positive samples.
    - max_samples (int or None): Maximum number of positive samples to use.

    Returns:
    - triples (List[str]): List of triple strings.
    - labels (List[int]): Corresponding labels (1 for positive, 0 for negative).
    """
    # Extract all positive triples
    positive_triples = [(u, d['relation'], v) for u, v, d in G.edges(data=True)]
    
    if max_samples:
        positive_triples = positive_triples[:max_samples]
    
    triples = []
    labels = []
    
    # Positive samples
    for triple in positive_triples:
        e1, r, e2 = triple
        triple_str = f"{e1} {r} {e2}"
        triples.append(triple_str)
        labels.append(1)
    
    # Negative samples
    num_negative = int(len(positive_triples) * negative_ratio)
    entities = list(G.nodes())
    relations = list(set(nx.get_edge_attributes(G, 'relation').values()))
    
    negative_triples = set()
    while len(negative_triples) < num_negative:
        e1, r, e2 = random.choice(entities), random.choice(relations), random.choice(entities)
        if not G.has_edge(e1, e2):
            negative_triples.add((e1, r, e2))
    
    for triple in negative_triples:
        e1, r, e2 = triple
        triple_str = f"{e1} {r} {e2}"
        triples.append(triple_str)
        labels.append(0)
    
    return triples, labels

# ============================
# 2. Model Training
# ============================

def train_model(G: nx.DiGraph, model_save_path: str = SUBGRAPH_CONFIG["link_prediction_model_path"]):
    """
    Train a DistilBERT model for link prediction on the given knowledge graph.

    Parameters:
    - G (nx.DiGraph): The raw knowledge graph.
    - model_save_path (str): Directory to save the trained model.

    Returns:
    - tokenizer: The tokenizer used.
    - model: The trained model.
    """
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    
    # Prepare dataset
    triples, labels = prepare_data(G, tokenizer, negative_ratio=1.0)
    
    # Split into train and validation
    triples_train, triples_val, labels_train, labels_val = train_test_split(
        triples, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = KGLinkPredictionDataset(triples_train, labels_train, tokenizer)
    val_dataset = KGLinkPredictionDataset(triples_val, labels_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Move model to device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # Define optimizer and loss
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)
        val_accuracy = correct / total
        print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Save the model
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return tokenizer, model

# ============================
# 3. Scoring Function s_G
# ============================

class TripleScorer:
    """
    Scoring function s_G that uses a trained language model to score triples.
    """
    def __init__(self, model_path: str = SUBGRAPH_CONFIG["link_prediction_model_path"]):
        os.makedirs(model_path, exist_ok=True)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def score(self, e1: str, r: str, e2: str) -> float:
        """
        Score a triple (e1, r, e2) using the trained model.

        Parameters:
        - e1 (str): Head entity.
        - r (str): Relation.
        - e2 (str): Tail entity.

        Returns:
        - float: Probability score in [0, 1].
        """
        triple_str = f"{e1} {r} {e2}"
        encoding = self.tokenizer.encode_plus(
            triple_str,
            add_special_tokens=True,
            max_length=50,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            # Assuming label 1 is the positive class
            prob_positive = probabilities[:, 1].item()
        return prob_positive
 
 
def train_link_prediction_model(G: nx.DiGraph):
    tokenizer, model = train_model(G, model_save_path=SUBGRAPH_CONFIG["link_prediction_model_path"])
    return tokenizer, model
    
    
if __name__ == "__main__":
    # ----------------------------
    # Load data and preprocess
    # ----------------------------
    pruned_ppr_graphs = load_all_graphs(SUBGRAPH_CONFIG["pruned_ppr_dir"])
    pruned_ppr_graphs = [g["graph"] for g in pruned_ppr_graphs]
    G_all = nx.compose_all(pruned_ppr_graphs)
    
    # ----------------------------
    # Train the Link Prediction Model
    # ----------------------------
    if not os.path.exists(SUBGRAPH_CONFIG["link_prediction_model_path"]):
        print("Training the link prediction model...")
        tokenizer, model = train_link_prediction_model(G_all)