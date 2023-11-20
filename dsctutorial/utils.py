import pandas as pd
import numpy as np
import torch

def positional_encoding(index, freqs):
    encoding = []
    for freq in freqs:
        values = getattr(index, freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        encoding.append(pd.DataFrame({f'{freq}_cos': np.cos(steps), f'{freq}_sin': np.sin(steps)}, index=index))
    encoding = pd.concat(encoding, axis=1)
    return encoding

def load_from_checkpoint(self, path):
    from genrisk.generation.tcn_gan import _TCNGenerator, _TCNDiscriminator
    from genrisk.generation.gan import GANModule
    gen = _TCNGenerator(
        self.latent_dim,
        len(self.conditional_columns),
        self.kernel_size,
        self.hidden_dim,
        len(self.target_columns),
        self.num_layers
    )
    disc = _TCNDiscriminator(
        len(self.target_columns),
        len(self.conditional_columns),
        self.kernel_size,
        self.hidden_dim,
        self.num_layers,
    )
    self.model = GANModule.load_from_checkpoint(
        path,
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
        gen=gen,
        disc=disc,
        latent_dim=self.latent_dim,
        lr=self.lr,
        num_disc_steps=self.num_disc_steps,
    )
