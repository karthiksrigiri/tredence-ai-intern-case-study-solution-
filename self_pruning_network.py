"""
Self-Pruning Neural Network on CIFAR-10
Tredence AI Engineering Intern – Case Study

Architecture: Feed-forward network with learnable gated (prunable) linear layers.
Mechanism:    Each weight has a sigmoid-gated scalar. L1 penalty on gate values
              encourages sparsity during training.
Author:       Submitted for Tredence AI Engineering Internship 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
#  Part 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that associates each weight with a
    learnable 'gate_score'. Sigmoid(gate_score) acts as a soft mask on the
    weight, allowing the optimizer to push gates toward 0 (pruned) or 1 (kept).

    Forward pass:
        gates        = sigmoid(gate_scores)          # shape: (out, in)
        pruned_w     = weight * gates                # element-wise mask
        output       = x @ pruned_w.T + bias        # standard affine step

    Gradients flow through both `weight` and `gate_scores` automatically
    because all operations are differentiable PyTorch ops.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard learnable weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores – same shape as weight
        # Initialized near 0 → sigmoid ≈ 0.5 (neither fully open nor closed)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming initialisation for the weight (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Convert gate scores to [0, 1] range
        gates = torch.sigmoid(self.gate_scores)          # (out, in)

        # Step 2: Soft-mask the weights
        pruned_weights = self.weight * gates             # (out, in)

        # Step 3: Standard linear transformation (implemented from scratch)
        return x @ pruned_weights.t() + self.bias       # (batch, out)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached for analysis)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_ratio(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate is below `threshold` (i.e. pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────
#  Part 2: Self-Pruning Feed-Forward Network
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    3-hidden-layer feed-forward network for CIFAR-10 classification.
    All linear layers are replaced with PrunableLinear.

    Input: 32×32×3 = 3072 raw pixels (flattened)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),   # 10 CIFAR-10 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)    # flatten (batch, 3072)
        return self.net(x)

    def prunable_layers(self):
        """Yield all PrunableLinear modules in the network."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all sigmoid-gate values across every PrunableLinear layer.

        Why L1 encourages sparsity:
          The L1 norm penalises each gate proportionally to its value.
          Its subgradient is a constant ±1 regardless of magnitude, so even
          tiny gate values receive a constant push toward zero. This is unlike
          L2, whose gradient shrinks as values approach zero, stalling pruning.
          Combined with the classification loss that needs some gates open for
          accuracy, the optimiser finds a sparse equilibrium.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            total = total + torch.sigmoid(layer.gate_scores).abs().sum()
        return total

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Global percentage of weights considered pruned."""
        pruned, total = 0, 0
        for layer in self.prunable_layers():
            gates = layer.get_gates()
            pruned += (gates < threshold).sum().item()
            total  += gates.numel()
        return 100.0 * pruned / total if total > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        """Collect all gate values into a single numpy array (for plotting)."""
        vals = []
        for layer in self.prunable_layers():
            vals.append(layer.get_gates().cpu().numpy().ravel())
        return np.concatenate(vals)


# ─────────────────────────────────────────────
#  Part 3: Data Loaders
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    """Download CIFAR-10 and return train / test DataLoaders."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────
#  Part 4: Training & Evaluation
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, lam: float):
    """Train for one epoch. Returns average total loss."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # Total Loss = CrossEntropy + λ × L1(gates)
        cls_loss     = F.cross_entropy(logits, labels)
        sparse_loss  = model.sparsity_loss()
        loss         = cls_loss + lam * sparse_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return test accuracy (%)."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def run_experiment(lam: float, epochs: int, train_loader, test_loader, device):
    """Train a self-pruning network for a given λ and return results."""
    print(f"\n{'='*55}")
    print(f"  λ = {lam}  |  Epochs = {epochs}")
    print(f"{'='*55}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, lam)
        scheduler.step()
        sparsity = model.global_sparsity()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  |  Loss: {avg_loss:.4f}  |  Sparsity: {sparsity:.1f}%")

    acc      = evaluate(model, test_loader, device)
    sparsity = model.global_sparsity()
    gates    = model.all_gate_values()

    print(f"\n  Final Test Accuracy : {acc:.2f}%")
    print(f"  Final Sparsity      : {sparsity:.2f}%")
    return acc, sparsity, gates


def plot_gate_distributions(results: dict, best_lam: float):
    """
    Plot gate value distribution for the best model.
    A successful self-pruning network shows a large spike near 0 (pruned weights)
    and a separate cluster of non-zero values (important weights).
    """
    gates = results[best_lam]["gates"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Gate Value Distribution  |  λ = {best_lam}  |  Best Model", fontsize=14, fontweight="bold")

    # Full distribution
    ax = axes[0]
    ax.hist(gates, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Gate Value (sigmoid output)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Full Range [0, 1]")
    ax.axvline(x=0.01, color="red", linestyle="--", label="Prune threshold (0.01)")
    ax.legend()

    # Zoom into [0, 0.2] to show the spike near 0 clearly
    ax = axes[1]
    ax.hist(gates[gates <= 0.2], bins=80, color="coral", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Gate Value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Zoomed: Near-zero gates (pruned region)")
    ax.axvline(x=0.01, color="red", linestyle="--", label="Prune threshold (0.01)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("gate_distribution.png", dpi=150, bbox_inches="tight")
    print("\n[Saved] gate_distribution.png")
    plt.show()


def print_results_table(results: dict):
    """Print a summary table of all λ experiments."""
    print("\n" + "="*55)
    print(f"{'Lambda':<12} {'Test Accuracy (%)':>20} {'Sparsity Level (%)':>20}")
    print("-"*55)
    for lam in sorted(results):
        acc      = results[lam]["accuracy"]
        sparsity = results[lam]["sparsity"]
        print(f"{lam:<12} {acc:>20.2f} {sparsity:>20.2f}")
    print("="*55)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EPOCHS     = 30          # Increase to 50+ for better accuracy in production
    BATCH_SIZE = 128
    LAMBDAS    = [1e-5, 1e-4, 1e-3]   # Low / Medium / High sparsity pressure

    train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

    results = {}
    for lam in LAMBDAS:
        acc, sparsity, gates = run_experiment(lam, EPOCHS, train_loader, test_loader, device)
        results[lam] = {"accuracy": acc, "sparsity": sparsity, "gates": gates}

    print_results_table(results)

    # Choose the λ that gives the best accuracy as "best model" for gate plot
    best_lam = max(results, key=lambda l: results[l]["accuracy"])
    print(f"\nBest λ by accuracy: {best_lam}")
    plot_gate_distributions(results, best_lam)


if __name__ == "__main__":
    main()
