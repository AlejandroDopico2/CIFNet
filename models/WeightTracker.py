import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict

from tqdm import tqdm

class WeightTracker:
    def __init__(self):
        self.weight_history: Dict[int, List[torch.Tensor]] = {}

    def update(self, class_idx: int, weights: torch.Tensor):
        if class_idx not in self.weight_history:
            self.weight_history[class_idx] = []
        self.weight_history[class_idx].append(weights.detach().clone())

    def plot_weight_changes(self):
        num_classes = len(self.weight_history)
        fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
        if num_classes == 1:
            axes = [axes]

        for class_idx, weights_history in tqdm(self.weight_history.items(), desc="Classes", unit="class"):
            ax = axes[class_idx]
            weight_changes = torch.stack(weights_history)
            
            for i in tqdm(range(weight_changes.shape[1]), desc=f"Class {class_idx} Weights", unit="weight"):  # For each weight
                ax.plot(weight_changes[:, i].numpy(), label=f'Weight {i}')
            
            ax.set_title(f'Class {class_idx} Weight Changes')
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Weight Value')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_weight_distribution(self, step: int = -1):
        num_classes = len(self.weight_history)
        fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))
        if num_classes == 1:
            axes = [axes]

        for class_idx, weights_history in self.weight_history.items():
            ax = axes[class_idx]
            weights = weights_history[step]
            
            ax.hist(weights.numpy(), bins=30)
            ax.set_title(f'Class {class_idx} Weight Distribution')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def plot_weight_statistics(self):
        num_classes = len(self.weight_history.keys())

        fig = go.Figure()

        visibility_matrix = []
        class_options = []

        for class_idx in tqdm(range(num_classes), desc="Classes", unit="class"):
            weights_history = self.weight_history[class_idx]
            weight_tensor = torch.stack(weights_history)
            
            # Mean of weights over time
            mean_weights = weight_tensor.mean(dim=1)
            fig.add_trace(go.Scatter(x=list(range(len(mean_weights))), y=mean_weights,
                                     mode='lines', name=f'Class {class_idx} mean',
                                     visible=False))
            
            # Standard deviation of weights over time
            std_weights = weight_tensor.std(dim=1)
            fig.add_trace(go.Scatter(x=list(range(len(std_weights))), y=std_weights,
                                     mode='lines', name=f'Class {class_idx} Std Dev',
                                     visible=False))
            
            # Frobenius norm of weight changes
            weight_changes = torch.diff(weight_tensor, dim=0)
            frob_norm = torch.norm(weight_changes, p='fro', dim=1)
            fig.add_trace(go.Scatter(x=list(range(len(frob_norm))), y=frob_norm,
                                     mode='lines', name=f'Class {class_idx} Frobenius Norm',
                                     visible=False))
            
            # Append visibility matrix for each class
            visibility_matrix.append([False] * num_classes * 3)  # 3 traces per class
            visibility_matrix[-1][class_idx * 3:class_idx * 3 + 3] = [True] * 3

            # Add option for dropdown
            class_options.append({'label': f'Class {class_idx}', 'method': 'update', 
                                'args': [{'visible': visibility_matrix[-1]}, 
                                        {'title': f'Statistics for Class {class_idx}'}]})

        for i in range(3):  # 3 traces per class (mean, std, frobenius norm)
            fig.data[i].visible = True

        # Add dropdown menu
        fig.update_layout(
            updatemenus=[{
                'buttons': class_options,
                'direction': 'down',
                'showactive': True,
                'x': 0.17,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top'
            }],
            title='Weight Statistics',
            xaxis_title='Update Step',
            yaxis_title='Value',
            height=800
        )

        fig.show()