import gymnasium as gym
import torch
import torch.nn.functional as F
from goofy import MultiLayerFeedForward
from random import sample

BATCH_SIZE = 128
GAMMA = 0.9
TAU = 0.01

env = gym.make("LunarLander-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"

#policy = DecoderOnly(4, 2, 3, 32, 4)
policy = MultiLayerFeedForward(2, 8, 4, 4).to(device)
target = MultiLayerFeedForward(2, 8, 4, 4).to(device)

obs, info = env.reset()

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-3)

def optimize_policy(policy, replay):
    if len(replay) < BATCH_SIZE:
        return
    transitions = sample(replay, BATCH_SIZE)
    batch = {k: [d[k] for d in transitions] for k in transitions[0]}

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch["next_state"])), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch["next_state"]
                                                if s is not None]).to(device)
    state_batch = torch.cat(batch["state"]).to(device)
    action_batch = torch.cat(batch["action"]).to(device)
    reward_batch = torch.cat(batch["reward"]).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.full([BATCH_SIZE], 0, device=device, dtype=torch.float32)
    with torch.no_grad():
        next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimizer.step()

def get_action(state, replay):
    if len(replay) < BATCH_SIZE:
        return torch.tensor([[env.action_space.sample()]], device=device)
    return policy(state).max(1)[1].view(1, 1)

replay = []
obs, info = env.reset()
state = torch.tensor(obs, device=device).unsqueeze(0)
for i in range(100000):
    action = get_action(state, replay)
    obs_, reward, terminated, truncated, info = env.step(action.squeeze().cpu().numpy())
    reward = torch.tensor([reward], device=device)

    if terminated:
        state_ = None
    else:
        state_ = torch.tensor(obs_, device=device).unsqueeze(0)

    replay.append({"state": state, "action": action, "next_state": state_, "reward": reward})

    state = state_

    optimize_policy(policy, replay)

    target_net_state_dict = target.state_dict()
    policy_net_state_dict = policy.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target.load_state_dict(target_net_state_dict)

    if terminated or truncated:
        obs, info = env.reset()
        state = torch.tensor(obs, device=device).unsqueeze(0)

    if i % 5000 == 0:
        print("Episode: {}, Reward: {}".format(i, reward.item()))

torch.save(policy.state_dict(), "policy_100K.pt")

env.close()