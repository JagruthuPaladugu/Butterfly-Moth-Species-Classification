
from src.preprocessing import load_data

train_loader, valid_loader, test_loader, classes = load_data()

images, labels = next(iter(train_loader))

print("Number of classes:", len(classes))
print("Class names:", classes[:5])

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
