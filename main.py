import torch
import torch.optim as optim
from model.dense_resunet import DenseResUNet
from data_preprocessing.data_augmentation import create_data_loaders
from torch import nn
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for images, labels in tqdm(train_loader, desc="Training", ncols=100):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels.float())

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        preds = (outputs > 0.5).float()
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_preds / total_preds
    return epoch_loss, epoch_accuracy

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", ncols=100):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.float())

            running_loss += loss.item()

            # Compute accuracy
            preds = (outputs > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = correct_preds / total_preds
    return epoch_loss, epoch_accuracy

def main():
    # Set up dataset directories
    train_dir = 'train'
    test_dir = 'test'

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_dir, test_dir, batch_size=32)

    # Initialize the model, loss function, and optimizer
    model = DenseResUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    for epoch in range(50):
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/50], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Evaluate on test data
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/50], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()
