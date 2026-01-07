from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.policies import DiscretePolicy
from utils.predict import PredictNet

class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, a_i, num_in_pol, num_out_pol, pre_act_space, sa_size, hidden_dim=64,
                 pred_hidden_dim=64, pred_z_dim=20, pre_lr=0.01, lr=0.01, onehot_dim=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            pre_act_space (int): number of dimensions for predicted actions(others agents)
        """
        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     pre_act_space,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        self.target_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            pre_act_space,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

        self.predict = PredictNet(a_i, sa_size, hidden_dim=pred_hidden_dim, z_dim=pred_z_dim)
        self.predict_optimizer = Adam(self.predict.parameters(), lr=pre_lr)

    def pre_step(self, obs, previous_other_acs):
        return self.predict(obs, previous_other_acs)

    def step(self, obs, pred_action, explore=False, return_all_probs=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        return self.policy(obs, pred_action, sample=explore, return_all_probs=return_all_probs)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
