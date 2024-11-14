import torch
from torch.nn.functional import one_hot
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from models.ROLANN_incremental import ROLANN_Incremental
from models.samplers.MemoryExpansionBuffer import MemoryExpansionBuffer
import numpy as np
import matplotlib.pyplot as plt


def prepare_data(dataset, num_classes, samples_per_class):
    class_indices = [
        np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)
    ]
    selected_indices = np.concatenate(
        [
            np.random.choice(indices, samples_per_class, replace=False)
            for indices in class_indices
        ]
    )
    return Subset(dataset, selected_indices)


def evaluate(model, dataloader):
    all_preds, all_labels = [], []
    for x, y in dataloader:
        x = x.view(x.size(0), -1)
        outputs = model(x)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return accuracy_score(all_labels, all_preds)


def evaluate_task(model, dataloader, task_classes):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            mask = torch.isin(y, torch.tensor(task_classes))
            x, y = x[mask], y[mask]
            if x.size(0) == 0:
                continue
            x = x.view(x.size(0), -1)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total if total > 0 else 0


def plot_overall_accuracy(overall_accuracies, num_classes_per_task, num_classes_total):
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(num_classes_per_task, num_classes_total + 1, num_classes_per_task),
        overall_accuracies,
        marker="o",
    )
    plt.title("Overall Accuracy After Each Task")
    plt.xlabel("Number of Classes Learned")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig("overall_accuracy.png")
    plt.close()
    print("Overall accuracy plot saved as 'overall_accuracy.png'")


def plot_task_accuracies(task_accuracies, num_classes_per_task, num_classes_total):
    plt.figure(figsize=(12, 6))
    for task, accuracies in task_accuracies.items():
        plt.plot(
            range(
                task + num_classes_per_task, num_classes_total + 1, num_classes_per_task
            ),
            accuracies,
            marker="o",
            label=f"Task {task//num_classes_per_task + 1}",
        )
    plt.title("Task Accuracy Throughout Training")
    plt.xlabel("Number of Classes Learned")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("task_accuracies.png")
    plt.close()
    print("Task accuracies plot saved as 'task_accuracies.png'")


num_classes_total = 10
num_classes_per_task = 2
num_samples_per_class = 800
memory_size_per_class = 0
buffer_batch_size = 32
batch_size = 128

# Data preparation
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

buffer = MemoryExpansionBuffer(memory_size_per_class)
rolann = ROLANN_Incremental(0, activation="lin", lamb=1.0)

overall_accuracies = []
task_accuracies = {i: [] for i in range(0, num_classes_total, num_classes_per_task)}

for task in range(0, num_classes_total, num_classes_per_task):
    print(f"\nTraining on classes {task} to {task + num_classes_per_task - 1}")

    rolann.add_num_classes(num_classes_per_task)

    # Prepare data for current task
    train_subset = prepare_data(
        train_dataset, task + num_classes_per_task, num_samples_per_class
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # Train on current task
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.view(x_batch.size(0), -1)
        y_one_hot = (
            one_hot(y_batch, num_classes=rolann.num_classes).float() * 0.9 + 0.05
        )

        # Get memory samples and combine with current batch
        x_memory, y_memory = buffer.get_memory_samples(
            batch_size=buffer_batch_size, num_classes=rolann.num_classes
        )
        x_combined = (
            torch.cat([x_memory, x_batch], dim=0) if x_memory.size(0) > 0 else x_batch
        )
        y_combined = (
            torch.cat([y_memory, y_one_hot], dim=0)
            if y_memory.size(0) > 0
            else y_one_hot
        )

        # Update model
        rolann.aggregate_update(x_combined, y_combined)

        # Add samples to buffer
        buffer.add_samples(x_batch, y_one_hot)

    x_buffer, y_buffer = buffer.get_memory_samples(batch_size=batch_size)
    if x_buffer.size(0) > 0:
        rolann.aggregate_update(x_buffer, y_buffer)

    # Evaluate on all seen classes
    test_subset = prepare_data(
        test_dataset, task + num_classes_per_task, num_samples_per_class
    )
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    accuracy = evaluate(rolann, test_loader)
    overall_accuracies.append(accuracy)
    print(f"Accuracy on classes 0 to {task + num_classes_per_task - 1}: {accuracy:.4f}")

    for prev_task in range(0, task + num_classes_per_task, num_classes_per_task):
        task_acc = evaluate_task(
            rolann, test_loader, range(prev_task, prev_task + num_classes_per_task)
        )
        task_accuracies[prev_task].append(task_acc)

# Plotting
plot_overall_accuracy(overall_accuracies, num_classes_per_task, num_classes_total)
plot_task_accuracies(task_accuracies, num_classes_per_task, num_classes_total)

# Final evaluation on all classes
print("\nFinal Evaluation on All Classes")
test_subset = prepare_data(test_dataset, num_classes_total, num_samples_per_class)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
final_accuracy = evaluate(rolann, test_loader)
print(f"Final Accuracy on all classes: {final_accuracy:.4f}")

# Class-wise accuracy
class_correct = [0] * num_classes_total
class_total = [0] * num_classes_total

for x, y in test_loader:
    x = x.view(x.size(0), -1)
    outputs = rolann(x)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == y).squeeze()
    for i in range(y.size(0)):
        label = y[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

for i in range(num_classes_total):
    print(f"Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%")

print(f"\nMemory buffer distribution: {buffer.get_class_distribution()}")
