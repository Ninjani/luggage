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
                 noise_function = lambda x: torch.rand_like(x), 
                 num_timesteps: int = 1000,
                 loss_function = torch.nn.MSELoss(), 
                 time_dimension = 32, 
                 diffusion_schedule="linear"):
        super().__init__()
        self.save_hyperparameters()
        self.model = DiffusionDenoiser(model, time_dimension=time_dimension)
        self.make_diffusion_variables(num_timesteps, diffusion_schedule)
        self.modify_graph_function = modify_graph_function
        self.noise_function = noise_function
        self.num_timesteps = num_timesteps
        self.loss_function = loss_function
        
        self.validation_step_outputs = []

    def make_diffusion_variables(self, num_timesteps, diffusion_schedule):
        self.betas = get_betas(num_timesteps, diffusion_schedule)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([torch.Tensor([1.]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @staticmethod
    def extract(a, t, batch_idx):
        """
        Extract the necessary values from the input array 'a' corresponding to the time steps 't'
        for each molecule in 'batch_idx'.

        Returns:
        (torch.Tensor): Array of shape (batch_size * num_nodes, 1) containing the extracted values.
        """
        return torch.repeat_interleave(a[t], torch.bincount(batch_idx)).reshape((len(batch_idx), 1))

    def forward(self, node_features, edge_index, batch, timesteps):
        # predicted noise
        return self.model(node_features, edge_index, batch, timesteps)
    
    def get_mean_loss(self, batch, batch_idx):
        timesteps = self.sample_timesteps(len(batch))
        data_sample, noise = self.diffuse_one_step(batch, timesteps)
        predicted_noise = self(data_sample.x, data_sample.edge_index, data_sample.batch, timesteps)
        loss = self.loss_function(predicted_noise, noise)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.get_mean_loss(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=len(batch))
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.get_mean_loss(batch, batch_idx)
        y_pred = self.reverse_diffuse(batch.clone(), linspace=self.num_timesteps)[-1]
        loss_inference = self.loss_function(y_pred, batch.y)
        self.log('val_loss', loss, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch.x.shape[0])
        self.log('val_loss_inference', loss_inference, on_step=True, sync_dist=True, on_epoch=True,
                 batch_size=batch.x.shape[0])
        self.validation_step_outputs.append(dict(loss=loss_inference, 
                                                 y_true=batch.y, 
                                                 y_pred=y_pred))
        return loss

    def sample_timesteps(self, num_samples):
        """
        Returns a tensor of random timesteps.
        """
        return torch.randint(0, self.num_timesteps, size=(num_samples,)).to(self.device)
    
    
    def diffuse_one_step(self, data, timestep) -> ty.Tuple[Data, torch.Tensor]:
        """
        Sample with noise
        Returns:
            Tuple[Data, Tensor]: A tuple containing the noised graph and the noise added to it.
        """
        noise = self.noise_function(data.y)
        y_noise = self.extract(self.sqrt_alphas_cumprod, timestep, data.batch) * data.y + self.extract(self.sqrt_one_minus_alphas_cumprod, timestep, data.batch) * noise
        data_sample = self.modify_graph_function(data.clone(), y_noise)
        return data_sample, noise

    @torch.no_grad()
    def reverse_diffuse_one_step(self, data, timestep):
        sqrt_recip_alphas_t = self.extract((1. / self.sqrt_alphas), timestep, data.batch)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, timestep, data.batch)
        betas_t = self.extract(self.betas, timestep, data.batch)
        w_noise = betas_t / sqrt_one_minus_alphas_cumprod_t
        # Use our model (noise predictor) to predict the mean noise
        y_pred_noise = self.model(data.x, data.edge_index, data.batch, timestep)
        y_pred = sqrt_recip_alphas_t * (data.y - w_noise * y_pred_noise)
        data = self.modify_graph_function(data, y_pred)
        if timestep[0] != 0:
            posterior_variance_t = self.extract(self.posterior_variance, timestep, data.batch)
            noise = self.noise_function(data.y)
            y_pred = data.y + posterior_variance_t * noise
            data = self.modify_graph_function(data, y_pred)
        return data

    @torch.no_grad()
    def reverse_diffuse(self, batch, noise=True, linspace=1000, start_timestep=None):
        if start_timestep is None:
            start_timestep = self.num_timesteps - 1
        start_timestep = min(start_timestep, self.num_timesteps - 1)
        linspace = min(linspace, start_timestep)
        if noise:
            # start from pure noise (for each example in the batch)
            batch.y = self.noise_function(batch.y)
        y_preds = []
        for timestep in reversed(np.linspace(0, start_timestep, linspace).astype(int)):
            timesteps = torch.full((len(batch),), timestep).to(self.device)
            batch = self.reverse_diffuse_one_step(batch, timesteps)
            y_preds.append(batch.y.detach())
        return y_preds
