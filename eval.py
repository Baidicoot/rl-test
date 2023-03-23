import gymnasium as gym
import torch
from goofy import MultiLayerFeedForward

env = gym.make("LunarLander-v2", render_mode="human")

device = "cuda" if torch.cuda.is_available() else "cpu"

policy = MultiLayerFeedForward(2, 8, 4, 4).to(device)
policy.load_state_dict(torch.load("policy_100K.pt", map_location=device))

obs, info = env.reset()
state = torch.tensor(obs, device=device, dtype=torch.float32)
while True:
    action = policy(state)
    obs, reward, terminated, truncated, info = env.step(action.max(0)[1].item())
    done = terminated or truncated
    if done:
        obs, info = env.reset()
    state = torch.tensor(obs, device=device, dtype=torch.float32)