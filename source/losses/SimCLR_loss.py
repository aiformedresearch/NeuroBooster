import torch
import torch.nn as nn
import torch.nn.functional as F

class simclr_loss(nn.Module):
    def __init__(self, args):
        super(simclr_loss, self).__init__()
        self.temperature = args.simclr_temperature
        self.batch_size = args.batch_size

    def forward(self, x, y):
        """
        Compute SimCLR NT-Xent loss between two batches x and y.

        Args:
            x: Tensor of shape [B, D] (augmented view 1)
            y: Tensor of shape [B, D] (augmented view 2)

        Returns:
            total_loss: scalar SimCLR loss
            pos_sim: average positive similarity (for logging)
            avg_sim: average similarity (for logging)
        """
        device = x.device
        batch_size = x.size(0)

        # Concatenate both views
        z = torch.cat([x, y], dim=0)  # [2B, D]
        z = F.normalize(z, dim=1)

        # Compute cosine similarity matrix [2B, 2B]
        sim_matrix = torch.matmul(z, z.T)  # cosine similarity

        # Remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)  # large negative number for stability

        # Similarity divided by temperature
        sim_matrix /= self.temperature

        # Positive pairs (i-th view1 with (i+B)-th view2, and vice versa)
        pos_indices = torch.arange(batch_size).to(device)
        positives = torch.cat([
            torch.sum(x * y, dim=-1),
            torch.sum(y * x, dim=-1)
        ])
        positives = positives / self.temperature

        # Cross-entropy loss: for each positive, use the full row as logits
        labels = torch.cat([
            pos_indices + batch_size,
            pos_indices
        ])

        logits = sim_matrix
        loss = F.cross_entropy(logits, labels)

        return loss, positives.mean(), sim_matrix.mean()
