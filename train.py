import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model import DetermineSpeaker

def train():
    data = pd.read_csv('data/cleaned_dataset.csv')
    X = data[['caps_ratio', 'punctuation_count', 'message_length']].values
    y = data['sender_id'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = DetermineSpeaker(input_dim=X_train.shape[1], num_classes=len(set(y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=100)

    for epoch in range(2000):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:04d} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test_t), dim=1)
        acc = (preds == y_test_t).float().mean()
        print(f"\nFinal Test Accuracy: {acc * 100:.2f}%")

    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    train()