from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Hiperparámetros
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones para las imágenes MNIST
transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),  # Redimensionamos las imágenes a 224x224 para ResNet18
        transforms.ToTensor(),  # Convertimos a tensor
        transforms.Normalize(
            (0.5,), (0.5,)
        ),  # Normalizamos con media y desviación típica
    ]
)

# Dataset MNIST
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Modificamos ResNet18 para trabajar con un solo canal (escala de grises)
model = resnet18(weights=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Identity()  # Quitamos la capa final para obtener las features
model = model.to(device)
model.eval()


# Función para obtener las features de ResNet18
def get_features(model, device, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            output = model(data)
            features.append(output.cpu().numpy())
            labels.append(target.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


# Obtener las features de ResNet18 sin entrenar
features, labels = get_features(model, device, test_loader)

# Usar t-SNE para reducir las dimensiones a 2D
# tsne = TSNE(n_components=2, random_state=42)
# features_2d = tsne.fit_transform(features)

pca = PCA(n_components=2, random_state=42)
features_2d = pca.fit_transform(features)

# Visualizar las features con Matplotlib
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab10", s=10
)
plt.colorbar(scatter)
plt.title("t-SNE visualization of ResNet18 features on MNIST")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
