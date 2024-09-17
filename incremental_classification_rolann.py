from sklearn.metrics import accuracy_score
import torch
from torch.nn.functional import one_hot
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.ROLANN_incremental import ROLANN_Incremental

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

num_classes_init = 5
num_samples_per_class = 100

train_loader = DataLoader(train_dataset, batch_size=num_samples_per_class * num_classes_init, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=num_samples_per_class * num_classes_init, shuffle=True)

x_init, y_init = next(iter(train_loader))
mask = y_init < num_classes_init
x_init, y_init = x_init[mask], y_init[mask]
x_init = x_init.view(x_init.size(0), -1)
y_init = one_hot(y_init, num_classes=num_classes_init) * 0.9 + 0.05

rolann = ROLANN_Incremental(num_classes_init)
rolann.aggregate_update(x_init, y_init)

x_test_init, y_test_init = next(iter(test_loader))
mask = y_test_init < num_classes_init
x_test_init, y_test_init = x_test_init[mask], y_test_init[mask]
x_test_init = x_test_init.view(x_test_init.size(0), -1)
y_test_init = one_hot(y_test_init, num_classes=num_classes_init) * 0.9 + 0.05

outputs_init = rolann(x_test_init)
print("Accuracy on initial classes:", accuracy_score(y_test_init.argmax(dim=1), outputs_init.argmax(dim=1)))

num_classes_new = 5
rolann.add_num_classes(num_classes_new)
num_classes_total = num_classes_init + num_classes_new

x_new, y_new = next(iter(train_loader))
mask = y_new == num_classes_init
x_new, y_new = x_new[mask], y_new[mask]
x_new = x_new.view(x_new.size(0), -1)
y_new = torch.ones(x_new.size(0), dtype=torch.int64) * num_classes_init
y_new = one_hot(y_new, num_classes_total) * 0.9 + 0.05

rolann.aggregate_update(x_new, y_new)

x_test_new, y_test_new = next(iter(test_loader))
mask = y_test_new == num_classes_init
x_test_new, y_test_new = x_test_new[mask], y_test_new[mask]
x_test_new = x_test_new.view(x_test_new.size(0), -1)
y_test_new = one_hot(y_test_new, num_classes_total) * 0.9 + 0.05

outputs_new = rolann(x_test_new)
print("Accuracy on new class:", accuracy_score(y_test_new.argmax(dim=1), outputs_new.argmax(dim=1)))
outputs_old = rolann(x_test_init)
print("Accuracy on old class:", accuracy_score(y_test_init.argmax(dim=1), outputs_old.argmax(dim=1)))

x_test_total, y_test_total = next(iter(test_loader))
x_test_total = x_test_total.view(x_test_total.size(0), -1)
y_test_total = one_hot(y_test_total, num_classes=num_classes_total) * 0.9 + 0.05

outputs_total = rolann(x_test_total)
print("Accuracy on all classes:", accuracy_score(y_test_total.argmax(dim=1), outputs_total.argmax(dim=1)))
