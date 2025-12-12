#!/usr/bin/env python
"""
Train GCN and GIN baselines for comparison.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.gcn import create_gcn_model
from baselines.gin import create_gin_model
from dataset.datasets import generate_synthetic_dataset, BA2Motif
from torch_geometric.data import DataLoader as PyGDataLoader


def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Train a GCN/GIN model."""
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                preds = logits.argmax(dim=1)
                val_preds.append(preds.cpu())
                val_labels.append(batch.y.cpu())
        
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_acc = accuracy_score(val_labels, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Acc={val_acc:.4f}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train GCN/GIN Baselines')
    parser.add_argument('--model', type=str, choices=['gcn', 'gin'], required=True)
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'ba2motif'])
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='chkpts/baselines')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    if args.dataset == 'synthetic':
        print("Generating synthetic datasets...")
        train_graphs = generate_synthetic_dataset(n_graphs=args.n_train, motif_types=['house', 'cycle', 'star'])
        val_graphs = generate_synthetic_dataset(n_graphs=args.n_val, motif_types=['house', 'cycle', 'star'])
        test_graphs = generate_synthetic_dataset(n_graphs=args.n_test, motif_types=['house', 'cycle', 'star'])
        num_classes = 3
        # Get input dim from first graph
        input_dim = train_graphs[0].x.size(1) if len(train_graphs) > 0 else 4
    else:
        print("Loading BA-2Motif datasets...")
        train_dataset = BA2Motif(args.data_dir, mode='train')
        val_dataset = BA2Motif(args.data_dir, mode='valid')
        test_dataset = BA2Motif(args.data_dir, mode='test')
        train_graphs = [train_dataset[i] for i in range(len(train_dataset))]
        val_graphs = [val_dataset[i] for i in range(len(val_dataset))]
        test_graphs = [test_dataset[i] for i in range(len(test_dataset))]
        num_classes = 2
        input_dim = train_graphs[0].x.size(1) if len(train_graphs) > 0 else 10
    
    train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    if args.model == 'gcn':
        model = create_gcn_model(input_dim, args.hidden_dim, num_classes, args.num_layers).to(device)
    else:
        model = create_gin_model(input_dim, args.hidden_dim, num_classes, args.num_layers).to(device)
    
    print(f"Training {args.model.upper()}...")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model, best_val_acc = train_model(model, train_loader, val_loader, device, args.epochs, args.lr)
    
    # Test
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=1)
            test_preds.append(preds.cpu())
            test_labels.append(batch.y.cpu())
    
    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    
    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'{args.model}_{args.dataset}_best.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
