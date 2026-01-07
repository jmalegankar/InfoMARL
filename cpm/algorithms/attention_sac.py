import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import copy

from torch.optim.lr_scheduler import StepLR, MultiStepLR

from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients, onehot_from_logits
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic

MSELoss = torch.nn.MSELoss()


class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01, pre_lr=0.01, beta=0.1,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 pred_hidden_dim=128,
                 pred_z_dim=20,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            pre_lr (float): Learning rate for predict
            beta (float): Momentum coefficient
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)
        self.agents = [AttentionAgent(a_i=a_i, lr=pi_lr, pre_lr=pre_lr,
                                      sa_size=sa_size,
                                      hidden_dim=pol_hidden_dim,
                                      pred_hidden_dim=pred_hidden_dim,
                                      pred_z_dim=pred_z_dim,
                                      **params)
                       for a_i, params in zip(range(self.nagents), agent_init_params)]
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                             attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.pre_lr = pre_lr
        self.beta = beta
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.pre_dev = 'cpu'  # device for predicts
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    @property
    def predicts(self):
        return [a.predict for a in self.agents]

    def pre_step(self, obs, previous_acs):
        """
        Predict the actions of all others agents
        based on the state of each agent
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of predicted actions for all others agents
        """
        pred_acts = []
        for a_i, ob, a in zip(range(self.nagents), obs, self.agents):
            previous_other_acs = previous_acs.copy()
            del previous_other_acs[a_i]
            pred_act = a.pre_step(ob, previous_other_acs)
            pred_acts.append(pred_act)
        return pred_acts

    def step(self, observations, pred_actions, explore=False, return_all_probs=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, pred_action, explore=explore,
                       return_all_probs=return_all_probs) for a, obs, pred_action in
                zip(self.agents, observations, pred_actions)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, pre_actions, acs, probs, rews, next_obs, next_pre_actions, dones = sample
        # Q loss
        next_acs = []
        next_log_pis = []
        for pi, ob, next_pre in zip(self.target_policies, next_obs, next_pre_actions):
            curr_next_ac, curr_next_log_pi = pi(ob, next_pre,
                                                return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        obs, pre_actions, acs, probs, rews, next_obs, next_pre_actions, dones = sample
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        for a_i, pi, ob, pre_act in zip(range(self.nagents), self.policies, obs, pre_actions):
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, pre_act, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)
            logger.add_scalar('agent%i/policy_entropy' % a_i, ent,
                              self.niter)

            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                  pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                                  grad_norm, self.niter)

    def update_predict(self, sample, logger=None):
        '''
        update prediction network
        Min KL
        '''
        obs, pre_actions, acs, probs, rews, next_obs, next_pre_actions, dones = sample
        samp_predict_outs = []
        samp_predict_acs = []
        samp_acs = []
        samp_acs_probs = []

        for a_i, pi, pre, next_ob in zip(range(self.nagents), self.policies, self.predicts, next_obs):
            previous_probs = [prob for prob in probs]
            del previous_probs[a_i]
            predict_out, predict_ac = pre(next_ob, previous_probs, return_outs=True)
            samp_predict_ac = torch.stack(predict_ac, dim=1)
            curr_ac, curr_probs = pi(next_ob, samp_predict_ac, return_all_probs=True)
            samp_acs.append(curr_ac)
            samp_acs_probs.append(curr_probs.detach())
            samp_predict_outs.append(predict_out)
            samp_predict_acs.append(predict_ac)

        for a_i, pre_ac in zip(range(self.nagents), samp_predict_outs):
            pre_loss = 0
            curr_agent = self.agents[a_i]
            # Delete the action of the agent itself, in order to compare it with the predicted
            real_acts = samp_acs_probs.copy()
            del real_acts[a_i]
            for pre_a, real_a in zip(pre_ac, real_acts):
                real_a = [torch.argmax(act) for act in real_a]
                real_a = torch.stack(real_a)
                pre_loss += F.cross_entropy(pre_a, real_a.long())
            # pre_loss = F.kl_div(F.log_softmax(torch.stack(pre_ac), dim=-1), torch.stack(real_acts), reduction='sum')
            pre_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(curr_agent.predict.parameters(), 5)
            curr_agent.predict_optimizer.step()
            curr_agent.predict_optimizer.zero_grad()
            # self.scheduler.step()
            if logger is not None:
                logger.add_scalar('agent%i/losses/pre_loss' % a_i, pre_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pre' % a_i, grad_norm, self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
            a.predict.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device
        if not self.pre_dev == device:
            for a in self.agents:
                a.predict = fn(a.predict)
            self.pre_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
            a.predict.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.pre_dev == device:
            for a in self.agents:
                a.predict = fn(a.predict)
            self.pre_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01, pre_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        agent_init_params = []
        sa_size = []
        acsp_sum = 0
        for acsp in env.action_space:
            acsp_sum += acsp.n
        for acsp, obsp in zip(env.action_space,
                              env.observation_space):
            agent_init_params.append({'num_in_pol': obsp.shape[0],
                                      'num_out_pol': acsp.n,
                                      'pre_act_space': acsp_sum - acsp.n})
            sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr, 'pre_lr': pre_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance