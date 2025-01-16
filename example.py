import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# === Define Encoder for Conditioning (16-Channel LiDAR Input) ===
class LiDAREncoder(nn.Module):
    def __init__(self, input_dim=16, latent_dim=128):
        super(LiDAREncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

# === Define Decoder (Reverse Diffusion Process) ===
class ReverseDiffusionDecoder(nn.Module):
    def __init__(self, point_dim=3, latent_dim=128, time_dim=16):
        super(ReverseDiffusionDecoder, self).__init__()
        self.latent_fc = nn.Linear(latent_dim, 128)
        self.time_fc = nn.Linear(time_dim, 128)
        self.model = nn.Sequential(
            nn.Linear(point_dim + 128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, point_dim)
        )

    def forward(self, points, latent_z, time_embedding):
        # Broadcast latent_z and time_embedding to match points batch size
        latent_features = self.latent_fc(latent_z).unsqueeze(1).expand(-1, points.size(1), -1)
        time_features = self.time_fc(time_embedding).unsqueeze(1).expand(-1, points.size(1), -1)
        # Concatenate inputs
        x = torch.cat([points, latent_features, time_features], dim=-1)
        return self.model(x)

# === Define Diffusion Model ===
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, lidar_dim=16, point_dim=3, latent_dim=128, time_dim=16):
        super(ConditionalDiffusionModel, self).__init__()
        self.encoder = LiDAREncoder(input_dim=lidar_dim, latent_dim=latent_dim)
        self.decoder = ReverseDiffusionDecoder(point_dim=point_dim, latent_dim=latent_dim, time_dim=time_dim)

    def forward(self, points, lidar_features, time_embedding):
        # points: noised gt from my dataset
        # Encode the LiDAR features
        latent_z = self.encoder(lidar_features)
        # Decode the noisy points
        output_points = self.decoder(points, latent_z, time_embedding)
        return output_points

# === Helper Function for Time Embedding ===
def sinusoidal_time_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal positional embeddings for timesteps.
    """
    device = timesteps.device
    half_dim = embedding_dim // 2
    freqs = torch.exp(-torch.arange(half_dim, dtype=torch.float32, device=device) * 
                      torch.log(torch.tensor(10000.0)) / half_dim)
    args = timesteps[:, None] * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding

# === Training Loop ===
def train_model(model, dataloader, optimizer, criterion, num_epochs=10, beta_schedule=None):
    for epoch in range(num_epochs):
        for lidar_features, static_points in dataloader:
            # Forward Diffusion: Add noise to static_points
            batch_size, num_points, _ = static_points.size()
            timesteps = torch.randint(0, len(beta_schedule), (batch_size,), device=static_points.device)
            time_embedding = sinusoidal_time_embedding(timesteps, embedding_dim=16)

            # Generate noisy points
            noise = torch.randn_like(static_points)
            alpha_t = torch.sqrt(1.0 - beta_schedule[timesteps].view(-1, 1, 1))
            noisy_points = alpha_t * static_points + torch.sqrt(beta_schedule[timesteps].view(-1, 1, 1)) * noise

            # Reverse Diffusion: Predict denoised points
            predicted_points = model(noisy_points, lidar_features, time_embedding)

            # Compute loss
            loss = criterion(predicted_points, static_points)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# === Example Usage ===
# Define hyperparameters
lidar_dim = 16
point_dim = 3
latent_dim = 128
time_dim = 16
num_timesteps = 100
beta_schedule = torch.linspace(0.0001, 0.02, num_timesteps)  # Linear beta schedule

# Create model, optimizer, and criterion
model = ConditionalDiffusionModel(lidar_dim, point_dim, latent_dim, time_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Example Dataloader (Replace with actual dataset)
dummy_dataloader = DataLoader(
    [(torch.randn(32, lidar_dim), torch.randn(32, 1024, point_dim)) for _ in range(100)], batch_size=4
)

# Train the model
train_model(model, dummy_dataloader, optimizer, criterion, num_epochs=10, beta_schedule=beta_schedule)
