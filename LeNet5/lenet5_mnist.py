import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class LeNet5(nn.Module):
    """Classic LeNet-5: expects 1x32x32 input."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),   # 32x32 -> 28x28
            nn.Tanh(),
            nn.AvgPool2d(2, 2),               # -> 14x14
            nn.Conv2d(6, 16, kernel_size=5),  # -> 10x10
            nn.Tanh(),
            nn.AvgPool2d(2, 2),               # -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    device = "cpu"

    # MNIST is 28x28; resize to 32x32 to match the original LeNet-5 input
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000)

    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Test
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()

        acc = 100 * correct / len(test_ds)
        print(f"Epoch {epoch}/{epochs} | loss {running_loss / len(train_loader):.4f} | test acc {acc:.2f}%")

    torch.save(model.state_dict(), "lenet5_mnist.pt")


if __name__ == "__main__":
    main()
