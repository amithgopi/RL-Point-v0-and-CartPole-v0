
from audioop import reverse
from re import S
import numpy
import torch
from utils import to_device

def compute_returns(rewards, masks, gamma):
    discountedReward = 0
    returns = []
    for step in reversed(range(len(rewards))):
        discountedReward = rewards[step] + gamma * discountedReward * masks[step]
        returns.append(discountedReward)
    returns.reverse()
    return returns

def computer_policy_grad(actions, states, theta):
    policyGradients = []
    for step in range(len(states)):
        policyGrad = torch.outer(actions[step] - torch.matmul(theta, states[step]), states[step])
        policyGradients.append(policyGrad)
    return policyGradients

def new_grad(rewards: torch.Tensor, actions: torch.Tensor, states: torch.Tensor, masks, theta: torch.Tensor, gamma, type_, version = 1):
    rewards, masks, states, theta, actions = to_device(torch.device('cpu'), rewards, masks, states, theta, actions)
    states_bias: torch.Tensor = torch.cat([states, torch.ones_like(states[..., :1])], axis=-1)
    batchSize = len(rewards)
    
    gammaPowers, cumDiscountedReward, policyGradSum, steps = 1., 0, 0, 0
    returns = compute_returns(rewards, masks, gamma)
    returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / returns.std()
    policyGradients = computer_policy_grad(actions, states_bias, theta)
    policyGradients = torch.stack(policyGradients)

    cumDiscountedReward = []
    cumPolicyGradient = []

    G = torch.zeros(2, 3)
    trajectories = 0
    grad = None
    if version == 1:
        for i in range(batchSize):
            if masks[i-1].item() == 0:
                cumDiscountedReward.append(returns[i])
            # cumPolicyGradient += policyGradients[i]*masks[i]
            G += policyGradients[i]
            if masks[i].item() == 0:
                trajectories += 1 
                # G += policyGradients[i]
                cumPolicyGradient.append(G)
                G = torch.zeros(2, 3)
            # else: G += policyGradients[i]*masks[i]
        cumPolicyGradient = torch.stack(cumPolicyGradient)
        cumDiscountedReward = torch.tensor(cumDiscountedReward)
        # print(cumPolicyGradient, "\n", cumDiscountedReward)
        grad = torch.stack([R*G for R, G, in zip(cumDiscountedReward, cumPolicyGradient)]).type(type_).sum(0) /trajectories
    elif version == 2:
        product = torch.stack([R*G for R, G, in zip(returns, policyGradients)]).type(type_)
        grad = []
        for i in range(len(product)):
            G += product[i]
            if masks[i].item() == 0:
                trajectories += 1 
                grad.append(G)
                G = torch.zeros(2, 3)
        grad = torch.stack(grad).sum(0) / trajectories
    elif version == 3:
        baseline = 0
        cumDiscountedReward = 0
        for i in range(batchSize):
            if masks[i-1].item() == 0:
                cumDiscountedReward = returns[i] 
            if masks[i].item() == 0:
                trajectories += 1
                baseline += cumDiscountedReward
            returns[i] = returns[i] - ((baseline/trajectories)  if trajectories else 0)
        product = torch.stack([R*G for R, G, in zip(returns, policyGradients)]).type(type_)
        grad = []
        for i in range(len(product)):
            G += product[i]
            if masks[i].item() == 0:
                trajectories += 1 
                grad.append(G)
                G = torch.zeros(2, 3)
        grad = torch.stack(grad).sum(0) / trajectories


    return grad

# def estimate_trajectory_grad(rewards, actions, states, theta, gamma):
#     batchSize = len(rewards)
#     gammaPowers = gamma**((batchSize-1) - torch.FloatTensor(range(batchSize)))
#     returns = torch.cumsum(rewards*gammaPowers, dim=0) / gammaPowers
#     returns = returns.flip(0)
#     # returns = (returns - returns.mean()) / returns.std()
#     cumDiscountedReward = returns[0]

#     policy_log_grad = torch.bmm((actions - torch.matmul(theta, states.T).T).unsqueeze(2), states.unsqueeze(1))
#     # policy_log_grad = policy_log_grad*returns.view(-1, 1, 1)
#     # print(batchSize, cumDiscountedReward, gamma, "\n", returns, "\n",  policy_log_grad)
#     # policy_log_grad = returns*policy_log_grad
#     sum_policy_grad = policy_log_grad.sum(0)
#     grad = cumDiscountedReward*sum_policy_grad
#     # grad = sum_policy_grad
#     return grad

def estimate_net_grad(rewards, masks,states,actions,gamma,theta,device):
    # these computations would be performed on CPU
    rewards, masks, states, theta, actions = to_device(torch.device('cpu'), rewards, masks, states, theta, actions)
    tensor_type = type(rewards)
    
    #Comment start
    # states_bias = torch.cat([states, torch.ones_like(states[..., :1])], axis=-1)

    # traj_indexes = [-1] + [idx for idx in range(len(masks)) if masks[idx] == 0]
    # # length_ = [ length[i+1]-length[i] for i in range(len(length)-1) ]

    grad = new_grad(rewards, actions, states, masks, theta, gamma, tensor_type, 3)

    # returns = (returns - returns.mean()) / returns.std()

    """ ESTIMATE NET GRADIENT"""
    # Roughly normalize the gradient
    grad = grad / (torch.norm(grad) + 1e-8)
    grad_returns = to_device(device, grad)[0]
    return grad_returns

