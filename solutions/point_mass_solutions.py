
from audioop import reverse
from re import S
import numpy
import torch
from utils import to_device

def new_grad(rewards: torch.Tensor, actions: torch.Tensor, states: torch.Tensor, masks, theta: torch.Tensor, gamma, type_, version = 1):
    rewards, masks, states, theta, actions = to_device(torch.device('cpu'), rewards, masks, states, theta, actions)
    states_bias: torch.Tensor = torch.cat([states, torch.ones_like(states[..., :1])], axis=-1)
    batchSize = len(rewards)
    
    gammaPowers, cumDiscountedReward, policyGradSum, steps = 1., 0, 0, 0
    gammaPowersList = []
    returns = []
    policyGradList = []
    trajectories = 0
    grad = numpy.zeros((2, 3))
    for i in range(batchSize):
        steps += 1
        cumDiscountedReward += rewards[i].item()*gammaPowers
        gammaPowersList.append(gammaPowers)
        returns.append(cumDiscountedReward)
        policyGrad = actions[i].numpy() - numpy.matmul(theta.numpy(), states_bias[i].numpy()) 
        # print(policyGrad.size, states_bias[i].numpy().size)
        policyGrad = numpy.outer(policyGrad, states_bias[i].numpy())
        policyGradList.append(policyGrad)
        policyGradSum += policyGrad 
        gammaPowers = gammaPowers*gamma
        if masks[i].item() == 0:
            if version == 2:
                gammaPowersList = numpy.array(gammaPowersList)
                returns = numpy.array(returns)*gammaPowersList
                # reverseGammaPowers = numpy.array(gammaPowersList[::-1])
                returns = numpy.cumsum(returns[::-1]) / gammaPowersList[::-1]
                # returns = (returns - returns.mean()) / returns.std()
                grad = numpy.sum([r*p for r, p in zip(returns,numpy.array(policyGradList))], axis = 0)
                
            else:
                grad += policyGradSum*cumDiscountedReward

            trajectories += 1
            returns, gammaPowersList, policyGradList = [], [], []
            gammaPowers, cumDiscountedReward, policyGradSum, steps = 1., 0, 0, 0
            # reset
    # print(grad, grad.size)
    grad = grad / trajectories
    # print(grad, grad.size)
    grad = torch.tensor(grad).type(type_)
    return grad

        


def estimate_trajectory_grad(rewards, actions, states, theta, gamma):
    batchSize = len(rewards)
    gammaPowers = gamma**((batchSize-1) - torch.FloatTensor(range(batchSize)))
    returns = torch.cumsum(rewards*gammaPowers, dim=0) / gammaPowers
    returns = returns.flip(0)
    # returns = (returns - returns.mean()) / returns.std()
    cumDiscountedReward = returns[0]

    policy_log_grad = torch.bmm((actions - torch.matmul(theta, states.T).T).unsqueeze(2), states.unsqueeze(1))
    # policy_log_grad = policy_log_grad*returns.view(-1, 1, 1)
    # print(batchSize, cumDiscountedReward, gamma, "\n", returns, "\n",  policy_log_grad)
    # policy_log_grad = returns*policy_log_grad
    sum_policy_grad = policy_log_grad.sum(0)
    grad = cumDiscountedReward*sum_policy_grad
    # grad = sum_policy_grad
    return grad

def estimate_net_grad(rewards, masks,states,actions,gamma,theta,device):
    # these computations would be performed on CPU
    rewards, masks, states, theta, actions = to_device(torch.device('cpu'), rewards, masks, states, theta, actions)
    tensor_type = type(rewards)
    
    #Comment start
    # states_bias = torch.cat([states, torch.ones_like(states[..., :1])], axis=-1)

    # traj_indexes = [-1] + [idx for idx in range(len(masks)) if masks[idx] == 0]
    # # length_ = [ length[i+1]-length[i] for i in range(len(length)-1) ]
    # grad = torch.zeros(2, 3)
    # for i in range(len(traj_indexes)-1):
    #     # print(traj_indexes[i]+1, traj_indexes[i+1])
    #     grad_traj = estimate_trajectory_grad(rewards[traj_indexes[i]+1:traj_indexes[i+1]+1], 
    #                                         actions[traj_indexes[i]+1:traj_indexes[i+1]+1],
    #                                         states_bias[traj_indexes[i]+1:traj_indexes[i+1]+1],
    #                                         theta, gamma)
    #     # print(grad_traj)
    #     grad += grad_traj
    # grad = grad / (len(traj_indexes)-1)
    # comment end
    grad = new_grad(rewards, actions, states, masks, theta, gamma, tensor_type, 1)
    #
    # returns = (returns - returns.mean()) / returns.std()

    """ ESTIMATE NET GRADIENT"""
    #
    #
    #
    #
    # Roughly normalize the gradient
    # print(grad)
    grad = grad / (torch.norm(grad) + 1e-8)
    # print(grad)
    grad_returns = to_device(device, grad)[0]
    return grad_returns

