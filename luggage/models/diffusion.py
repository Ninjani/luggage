import math
import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
import typing as ty


def get_betas(num_timesteps, diffusion_schedule):
    if diffusion_schedule == 'linear':
        return torch.linspace(0.0001, 0.02, num_timesteps)
    elif diffusion_schedule == 'cosine':
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos((x / steps) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Invalid schedule: {diffusion_schedule}, must be 'linear' or 'cosine'")
    
def random_noise_function(batch, sigma):
    batch_noise = batch.clone()
    batch_noise.y = torch.normal(mean=0, std=sigma, size=batch_noise.y.shape)
    return batch_noise

def extract(values, indices, batch_idx):
    """
    Given one index value `i` for each of the 'batch_size' molecules,
    extract the necessary values from the input array `values` corresponding to the indices 'i' in `indices`
    and repeat them 'num_nodes' times for each molecule in the batch.

    Parameters:
    values (torch.Tensor): Array of shape (max(indices), 1) containing the values to extract from.
    indices (torch.Tensor): Array of shape (batch_size, 1) containing the indices to extract.
    batch_idx (torch.Tensor): Array of shape (batch_size * num_nodes, 1) containing the indices of the molecules.

    Returns:
    (torch.Tensor): Array of shape (batch_size * num_nodes, 1) containing the extracted values.
    """
    return torch.repeat_interleave(values[indices], torch.bincount(batch_idx)).reshape((len(batch_idx), 1))


class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionDenoiser(pl.LightningModule):
    def __init__(self, model: ty.Union[pl.LightningModule, torch.nn.Module], time_dimension: int = 32):
        super().__init__()
        self.timestep_embedder = SinusoidalPositionEmbeddings(time_dimension)
        self.model = model

    def forward(self, node_features, edge_index, batch, timesteps):
        timestep_encoding = torch.repeat_interleave(
            self.timestep_embedder(timesteps), torch.bincount(batch), dim=0)
        x = torch.cat([node_features, timestep_encoding], dim=1)
        x_out = self.model(x, edge_index)
        return x_out


class Diffusion(pl.LightningModule):
    def __init__(self, model: ty.Union[pl.LightningModule, torch.nn.Module],
                 modify_graph_function: ty.Callable[[Data, torch.Tensor], Data],
                 noise_function: ty.Callable[[Data, torch.Tensor], Data],
                 diffusion_type: str = 'discrete',
                 loss_function = torch.nn.MSELoss(), 
                 time_dimension = 32, 
                 diffusion_schedule="linear",
                 num_timesteps: int = 1000,
                 min_sigma: float=0.,
                 max_sigma: float=10.):
        super().__init__()
        self.save_hyperparameters()
        self.model = DiffusionDenoiser(model, time_dimension=time_dimension)
        self.modify_graph_function = modify_graph_function
        self.noise_function = noise_function
        self.loss_function = loss_function
        
        if diffusion_type == 'discrete':
            self.num_timesteps = num_timesteps
            self.diffusion_schedule = diffusion_schedule
            self.betas = get_betas(num_timesteps, diffusion_schedule)
            self.alphas = 1. - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, 0)
            self.alphas_cumprod_prev = torch.cat([torch.Tensor([1.]), self.alphas_cumprod[:-1]])
            self.sqrt_alphas = torch.sqrt(self.alphas)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
            self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        elif diffusion_type == 'continuous':
            self.min_sigma = min_sigma
            self.max_sigma = max_sigma
        self.diffusion_type = diffusion_type

        self.validation_step_outputs = []

    def diffuse_one_step(self, batch, timesteps) -> ty.Tuple[Data, torch.Tensor]:
        if self.diffusion_type == 'discrete':
            batch_noise = self.noise_function(batch, 1.)
            y_new = extract(self.sqrt_alphas_cumprod, timesteps, batch.batch) * batch.y + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, batch.batch) * batch_noise.y
            batch_new = self.modify_graph_function(batch.clone(), y_new)
            return batch_new, batch_noise.y
        else:
            timesteps = extract(timesteps, torch.arange(len(batch)), batch.batch)
            sigma = self.get_sigma(timesteps)
            batch_noise = self.noise_function(batch, sigma)
            y_noise = -batch_noise.y / sigma**2  
            y_new = batch.y + batch_noise.y
            batch_new = self.modify_graph_function(batch.clone(), y_new)
            return batch_new, y_noise
        
    def sample_timesteps(self, num_samples):
        """
        Returns a tensor of random timesteps.
        """
        if self.diffusion_type == 'discrete':
            return torch.randint(0, self.num_timesteps, size=(num_samples,)).to(self.device)
        else:
            return torch.rand(size=(num_samples,)).to(self.device)
    
    def get_sigma(self, time):
        sigma = self.min_sigma ** (1-time) * self.max_sigma ** time
        return sigma
        
    def forward(self, node_features, edge_index, batch, timesteps):
        # predicted noise
        return self.model(node_features, edge_index, batch, timesteps)
    
    def diffuse_and_get_loss(self, batch, batch_idx):
        timesteps = self.sample_timesteps(len(batch))
        batch_new, y_noise = self.diffuse_one_step(batch, timesteps)
        y_pred_noise = self(batch_new.x, batch_new.edge_index, batch_new.batch, timesteps)
        return self.loss_function(y_pred_noise, y_noise)
    
    @torch.no_grad()
    def reverse_diffuse_one_step(self, batch, timestep, dt):
        timesteps = torch.full((len(batch),), timestep, dtype=torch.long, device=self.device)
        if self.diffusion_type == 'discrete':
            sqrt_recip_alphas_t = extract((1. / self.sqrt_alphas), timesteps, batch.batch)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, batch.batch)
            betas_t = extract(self.betas, timesteps, batch.batch)
            w_noise = betas_t / sqrt_one_minus_alphas_cumprod_t
            # Use our model (noise predictor) to predict the mean noise
            y_pred_noise = self(batch.x, batch.edge_index, batch.batch, timestep)
            y_pred = sqrt_recip_alphas_t * (batch.y - w_noise * y_pred_noise)
            batch = self.modify_graph_function(batch, y_pred)
            if timestep[0] != 0:
                posterior_variance_t = extract(self.posterior_variance, timesteps, batch.batch)
                batch_noise = self.noise_function(batch, 1.)
                y_pred = batch.y + posterior_variance_t * batch_noise.y
                batch = self.modify_graph_function(batch, y_pred)
            return batch
        else:
            y_pred_noise = self(batch.x, batch.edge_index, batch.batch, timesteps)
            sigma = self.get_sigma(timestep)
            gradient = sigma * torch.sqrt(torch.tensor(2 * np.log(self.max_sigma / self.min_sigma)))
            batch_noise = self.noise_function(batch, 1.)
            y_pred = gradient ** 2 * dt * y_pred_noise + gradient * torch.sqrt(dt) * batch_noise.y
            batch = self.modify_graph_function(batch, y_pred)
        return batch

    @torch.no_grad()
    def reverse_diffuse(self, batch, noise=True, steps=1000):
        if noise:
            # start from pure noise (for each example in the batch)
            batch = self.noise_function(batch, self.max_sigma)
        y_preds = []
        if self.diffusion_type == 'discrete':
            timesteps = np.linspace(0, self.num_timesteps, steps)[::-1]
        else:
            timesteps = np.linspace(0, 1, steps)[::-1]
        for t in range(1, len(timesteps)):
            dt = timesteps[t] - timesteps[t-1]
            batch = self.reverse_diffuse_one_step(batch, timesteps[t], dt)
            y_preds.append(batch.y.detach())
        return y_preds
    
    def training_step(self, batch, batch_idx):
        loss = self.diffuse_and_get_loss(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=len(batch))
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.diffuse_and_get_loss(batch, batch_idx)
        y_pred = self.reverse_diffuse(batch.clone(), steps=1000)[-1]
        loss_inference = self.loss_function(y_pred, batch.y)
        self.log('val_loss', loss, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch.x.shape[0])
        self.log('val_loss_inference', loss_inference, on_step=True, sync_dist=True, on_epoch=True,
                 batch_size=batch.x.shape[0])
        self.validation_step_outputs.append(dict(loss=loss_inference, 
                                                 y_true=batch.y, 
                                                 y_pred=y_pred))
        return loss
