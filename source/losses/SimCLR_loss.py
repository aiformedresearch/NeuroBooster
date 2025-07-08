import torch
import torch.nn as nn
import torch.nn.functional as F

class simclr_loss(nn.Module):
    def __init__(self, args):
        super(simclr_loss, self).__init__()
        self.temperature = args.simclr_temperature
        self.batch_size = args.batch_size

    def forward(self, x, y):
        device = x.device
        batch_size = x.size(0)

        # Normalize embeddings
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

        # Sanity check on cosine similarity
        cosine_vals = F.cosine_similarity(x, y, dim=1)
        max_sim = cosine_vals.max().item()
        min_sim = cosine_vals.min().item()
        mean_sim = cosine_vals.mean().item()
        
        # Concatenate embeddings
        z = torch.cat([x, y], dim=0)  # shape [2B, D]
        z = F.normalize(z, dim=1)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Mask diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)

        # Labels for cross-entropy
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Compute contrastive loss
        loss = F.cross_entropy(sim_matrix, labels)

        # Logging metrics
        with torch.no_grad():
            pos_sim = cosine_vals.mean()  # scalar in [-1, 1]
            raw_sim_matrix = torch.matmul(z, z.T)
            avg_sim = raw_sim_matrix.masked_fill(mask, 0).sum() / (2 * batch_size * (2 * batch_size - 1))

        return loss, pos_sim, avg_sim

