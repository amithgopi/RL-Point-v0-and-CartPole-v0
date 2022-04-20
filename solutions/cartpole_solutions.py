import numpy
import torch

def compute_returns(rewards, masks, gamma):
    discountedReward = 0
    returns = []
    for step in reversed(range(len(rewards))):
        discountedReward = rewards[step] + gamma * discountedReward * masks[step]
        returns.append(discountedReward)
    returns.reverse()
    return torch.tensor(returns)

def estimate_critic_loss_version3(returns, log_policies, values, masks):
    traj_indexes = [-1] + [idx for idx in range(len(masks)) if masks[idx] == 0]
    trajectoryCount = len(traj_indexes) - 1

    returns = returns.unsqueeze(1)
    advantages = returns - values.detach()
    critic_loss = -(advantages*log_policies)
    critic_loss = critic_loss.sum() / trajectoryCount
    return critic_loss

def estimate_critic_loss_version2(returns, log_policies, values, masks):
    traj_indexes = [-1] + [idx for idx in range(len(masks)) if masks[idx] == 0]
    trajectoryCount = len(traj_indexes) - 1

    returns = returns.unsqueeze(1)
    critic_loss = -(returns*log_policies)
    critic_loss = critic_loss.sum() / trajectoryCount
    return critic_loss

def estimate_critic_loss_version1(returns, log_policies, values, masks):
    traj_indexes = [-1] + [idx for idx in range(len(masks)) if masks[idx] == 0]
    trajectoryCount = len(traj_indexes) - 1

    currentReturn = 0
    cumReturns = returns.clone()
    for i in range(len(masks)):
        if masks[i-1].item() == 0: 
            currentReturn = returns[i]
        cumReturns[i] = currentReturn

    cumReturns = cumReturns.unsqueeze(1)
    critic_loss = -(cumReturns*log_policies)
    critic_loss = critic_loss.sum() / trajectoryCount
    return critic_loss



def pg_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, masks, rewards, gamma, device):
    # traj_indexes = [-1] + [idx for idx in range(len(masks)) if masks[idx] == 0]
    # trajectoryCount = len(traj_indexes) - 1
    # print(returns.mean(),  returns.std())
    returns = compute_returns(rewards, masks, gamma).to(device)
    returns = (returns - returns.mean()) / returns.std()
    log_policies = policy_net.get_log_prob(states, actions)
    values = value_net(states)

    # returns = (returns - returns.mean()) / returns.std()
    # returns = returns.unsqueeze(1)
    
    # critic_loss = -(returns*log_policies)
    # critic_loss = critic_loss.sum() / trajectoryCount
    critic_loss = estimate_critic_loss_version3(returns, log_policies, values, masks)

    
    
    actor_loss = (returns - values).pow(2).mean()

    print(actor_loss, critic_loss)
    optimizer_policy.zero_grad()
    optimizer_value.zero_grad()
    critic_loss.backward()
    actor_loss.backward()
    optimizer_policy.step()
    optimizer_value.step()