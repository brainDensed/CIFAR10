import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from cifar10_pytorch.model import CNN
from cifar10_pytorch.config import MODEL_DIR, MODEL_PATH, DATA_DIR

torch.backends.cudnn.benchmark = True

def main():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    full_train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    full_val_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=val_test_transform
    )

    train_subset, val_subset = random_split(
        range(len(full_train_dataset)),
        [45000, 5000]
    )

    train_dataset = Subset(full_train_dataset, train_subset.indices)
    val_dataset = Subset(full_val_dataset, val_subset.indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        transform=val_test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CNN().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    patience = 3
    counter = 0
    best_loss = float("inf")

    for epoch in range(50):
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
    
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
    
            logits = model(images)
    
            loss = loss_fn(logits, labels)
            total_train_loss += loss.item()
    
            loss.backward()
    
            optimizer.step()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device=device, non_blocking=True)
                labels = labels.to(device=device, non_blocking=True)

                logits = model(images)

                loss = loss_fn(logits, labels)

                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        
        else:
            counter += 1
        
        if(counter >= patience):
            break

    
        print(f"Epoch: {epoch} | " 
        f"Train loss: {total_train_loss / len(train_loader):.4f} | "
        f"Val loss: {avg_val_loss:.4f}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    model.eval()
    
    total_test_loss = 0
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            logits = model(images)
        
            loss = loss_fn(logits, labels)
        
            total_test_loss += loss.item()
        
            predictions = torch.argmax(logits, dim=1)
        
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    print(f"accuracy: {correct / total}")
    print(f"avg loss: {total_test_loss / len(test_loader)}")

if __name__ == "__main__":
    main()
