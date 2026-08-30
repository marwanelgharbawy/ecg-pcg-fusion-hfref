import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    # Weighted loss for class imbalance[cite: 1]
    # Adjust pos_weight based on the ratio of Negative/Positive samples in your dataset
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for ecg, pcg, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            ecg = ecg.to(device)
            pcg = pcg.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(ecg, pcg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * ecg.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validation evaluation step goes here