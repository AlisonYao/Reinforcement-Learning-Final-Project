##############################
#     Alison Yao (yy2564)    #
##############################
import gym
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import time
from spinup.algos.sac_pytorch.core_auto import TanhGaussianPolicySACAdapt, Mlp, soft_update_model1_with_model2, ReplayBuffer
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
import os, sys

"""
SAC multistep variant
student version
"""

class MultistepReplayBuffer:
    """
    FIFO buffer for a multi-step agent
    """
    def __init__(self, obs_dim, act_dim, size):
        """
        :param obs_dim: size of observation
        :param act_dim: size of the action
        :param size: size of the buffer
        """
        ## init buffers as numpy arrays
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.tildone_buf = np.zeros(size, dtype=np.float32) # number of timestep until terminal state/end of episode (min is 1)
        self.ksteprews_buf = np.zeros(size, dtype=np.float32) # the sum of discounted rewards for the next k steps
        self.ptr, self.size, self.max_size = 0, 0, size
        self.ready_ptr = 0 # points to the next datapoint that will be ready for multistep est

        # temporary buffer to help the multistep part
        self.temp_max_size = 100000
        self.obs1_temp_buf = np.zeros([self.temp_max_size, obs_dim], dtype=np.float32)
        self.obs2_temp_buf = np.zeros([self.temp_max_size, obs_dim], dtype=np.float32)
        self.acts_temp_buf = np.zeros([self.temp_max_size, act_dim], dtype=np.float32)
        self.rews_temp_buf = np.zeros(self.temp_max_size, dtype=np.float32)
        self.done_temp_buf = np.zeros(self.temp_max_size, dtype=np.float32)
        self.temp_ptr = 0

        # when we looking for the state after k steps, if tildone value is >= k, then we pick the
        # observation that is k steps away, or maybe a better option is to pick the next_obs that is k-1 steps away
        # then we have what we need to compute the multi-step target
    def _store_ready_data(self, obs, act, rew, next_obs, done, til_done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.tildone_buf[self.ptr] = til_done
        self.ptr = (self.ptr + 1) % self.max_size
        ## keep track of the current buffer size
        self.size = min(self.size+1, self.max_size)
    def store(self, obs, act, rew, next_obs, done, current_ep_len, max_ep_len, multistep_k, gamma):
        """
        data will get stored in the pointer's location
        data should NOT be in tensor format.
        it's easier if you get data from environment
        then just store them with the geiven format
        """
        # first we store them in temporary buffers
        self.obs1_temp_buf[self.temp_ptr] = obs
        self.obs2_temp_buf[self.temp_ptr] = next_obs
        self.acts_temp_buf[self.temp_ptr] = act
        self.rews_temp_buf[self.temp_ptr] = rew
        self.done_temp_buf[self.temp_ptr] = done
        self.temp_ptr = (self.temp_ptr+1) % self.temp_max_size

        # then if the episode terminates, or if the current episode length is long enough,
        # we move some of the data in temp buffers to real buffers. (minibatch for update will only
        # come from real buffers)
        if current_ep_len == max_ep_len:
            # first case: when current episode terminates, we do not move any current temp data to real buffer, since we
            # don't have a multi-step estimate (essentially these data are discarded)
            pass
        elif done:
            # if the episode terminated (either with a long ep len, or an ep len shorter than k), then we will
            # store the temp data that are not yet stored
            n_left_to_store_data = min(multistep_k, current_ep_len)
            for i in range(n_left_to_store_data):
                n_steps = n_left_to_store_data - i
                start_temp_ptr = (self.temp_ptr - n_steps) % self.temp_max_size
                # compute sum of discounted reward
                reward_list = []
                reward_temp_ptr = start_temp_ptr
                for i in range(n_steps):
                    reward_list.append(self.rews_temp_buf[reward_temp_ptr])
                    reward_temp_ptr = (reward_temp_ptr + 1) % self.temp_max_size

                sum_discounted_reward = self.compute_sum_discounted_reward_from_reward_list(reward_list, gamma)
                self._store_ready_data(self.obs1_temp_buf[start_temp_ptr], self.acts_temp_buf[start_temp_ptr],
                                       sum_discounted_reward, self.obs2_temp_buf[self.temp_ptr - 1],
                                       self.done_temp_buf[self.temp_ptr - 1], n_steps)
        elif current_ep_len >= multistep_k:
            # if the episode does not terminate or reach episode max length
            # and we are ready to store one multistep data point
            start_temp_ptr = (self.temp_ptr - multistep_k) % self.temp_max_size
            # compute discounted reward
            n_steps = multistep_k

            reward_list = []
            reward_temp_ptr = start_temp_ptr
            for i in range(n_steps):
                reward_list.append(self.rews_temp_buf[reward_temp_ptr])
                reward_temp_ptr = (reward_temp_ptr + 1) % self.temp_max_size

            sum_discounted_reward = self.compute_sum_discounted_reward_from_reward_list(reward_list, gamma)
            self._store_ready_data(self.obs1_temp_buf[start_temp_ptr], self.acts_temp_buf[start_temp_ptr],
                                   sum_discounted_reward, self.obs2_temp_buf[self.temp_ptr-1],
                                   self.done_temp_buf[self.temp_ptr-1], n_steps)
        else:
            # if episode hasn't end, and current ep len is too small, then do nothing here
            pass

    def compute_sum_discounted_reward_from_reward_list(self, reward_list, gamma):
        sum_discounted_reward = 0
        # TODO given a list of rewards (the rewards you get in the next few steps)
        #  write code here to compute sum of discounted rewards (very easy)
        # sum of discounted reward for k steps = r1 + gamma * r2 + gamma**2 * r3 + ... + gamma**(k-1) * rk
        for i in range(len(reward_list)):
            sum_discounted_reward += reward_list[i] * (gamma ** i)
        return sum_discounted_reward

    def sample_batch(self, batch_size=32, idxs=None):
        ## sample with replacement from buffer
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    idxs=idxs)
    def get_all_batch(self):
        return dict(obs1=self.obs1_buf[:self.size],
                    obs2=self.obs2_buf[:self.size],
                    acts=self.acts_buf[:self.size],
                    rews=self.rews_buf[:self.size],
                    done=self.done_buf[:self.size])

def sac_multistep(env_fn, hidden_sizes=[256, 256], seed=0,
                  steps_per_epoch=1000, epochs=1000, replay_size=int(1e6), gamma=0.99,
                  polyak=0.995, lr=3e-4, alpha=0.2, batch_size=256, start_steps=10000,
                  max_ep_len=1000, save_freq=1, save_model=False,
                  auto_alpha=True, grad_clip=-1, logger_store_freq=100,
                  multistep_k=1, debug=False, use_single_variant=False,
                  logger_kwargs=dict(), ):
    """
    Largely following OpenAI documentation, but a bit different
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        hidden_sizes: number of entries is number of hidden layers
            each entry in this list indicate the size of that hidden layer.
            applies to all networks

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch. Note the epoch here is just logging epoch
            so every this many steps a logging to stdouot and also output file will happen
            note: not to be confused with training epoch which is a term used often in literature for all kinds of
            different things

        epochs (int): Number of epochs to run and train agent. Usage of this term can be different in different
            algorithms, use caution. Here every epoch you get new logs

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration. However during testing the action always come from policy

        max_ep_len (int): Maximum length of trajectory / episode / rollout. Environment will get reseted if
        timestep in an episode excedding this number

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_model (bool): set to True if want to save the trained agent

        auto_alpha: set to True to use the adaptive alpha scheme, target entropy will be set automatically

        grad_clip: whether to use gradient clipping. < 0 means no clipping

        logger_store_freq: how many steps to log debugging info, typically don't need to change

    """
    if debug:
        hidden_sizes = [2, 2]
        batch_size = 2
        start_steps = 1000
        multistep_k = 5
        use_single_variant = True
    print('[basic setups] multistep_k:', multistep_k, 'use_single_variant:', use_single_variant)
    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env, test_env = env_fn(), env_fn()

    ## seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## seed environment along with env action space so that everything about env is seeded
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    test_env.seed(seed + 10000)
    test_env.action_space.np_random.seed(seed + 10000)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # if environment has a smaller max episode length, then use the environment's max episode length
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()

    # Experience buffer
    replay_buffer = MultistepReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    """
    Auto tuning alpha
    """
    if auto_alpha:
        target_entropy =  -np.prod(env.action_space.shape).item() # H
        log_alpha = torch.zeros(1, requires_grad=True)
        alpha_optim = optim.Adam([log_alpha], lr=lr)
    else:
        target_entropy, log_alpha, alpha_optim = None, None, None

    def test_agent(n=1):
        """
        This will test the agent's performance by running n episodes
        During the runs, the agent only take deterministic action, so the
        actions are not drawn from a distribution, but just use the mean
        :param n: number of episodes to run the agent
        """
        ep_return_list = np.zeros(n)
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                a = policy_net.get_env_action(o, deterministic=True)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            ep_return_list[j] = ep_ret
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    """init all networks"""
    # see line 1
    policy_net = TanhGaussianPolicySACAdapt(obs_dim, act_dim, hidden_sizes,action_limit=act_limit)
    q1_net = Mlp(obs_dim+act_dim,1,hidden_sizes)
    q2_net = Mlp(obs_dim+act_dim,1,hidden_sizes)

    q1_target_net = Mlp(obs_dim+act_dim,1,hidden_sizes)
    q2_target_net = Mlp(obs_dim+act_dim,1,hidden_sizes)

    # see line 2: copy parameters from value_net to target_value_net
    q1_target_net.load_state_dict(q1_net.state_dict())
    q2_target_net.load_state_dict(q2_net.state_dict())

    # set up optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(),lr=lr)
    q1_optimizer = optim.Adam(q1_net.parameters(),lr=lr)
    q2_optimizer = optim.Adam(q2_net.parameters(),lr=lr)

    # mean squared error loss for v and q networks
    mse_criterion = nn.MSELoss()

    # Main loop: collect experience in env and update/log each epoch
    # NOTE: t here is the current number of total timesteps used
    # it is not the number of timesteps passed in the current episode
    current_update_index = 0
    for t in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy.
        """
        if t > start_steps:
            a = policy_net.get_env_action(o, deterministic=False)
        else:
            a = env.action_space.sample()
        # Step the env, get next observation, reward and done signal
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience (observation, action, reward, next observation, done) to replay buffer
        # the multi-step buffer (given to you) will store the data in a fashion that
        # they can be easily used for multi-step update
        replay_buffer.store(o, a, r, o2, d, ep_len, max_ep_len, multistep_k, gamma)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        """perform update"""
        if replay_buffer.size >= batch_size:
            # get data from replay buffer
            batch = replay_buffer.sample_batch(batch_size)
            obs_tensor = Tensor(batch['obs1'])
            # NOTE: given the multi-step buffer, obs_next_tensor now contains the observation that are
            # k-step away from current observation
            obs_next_tensor = Tensor(batch['obs2'])
            acts_tensor = Tensor(batch['acts'])
            # NOTE: given the multi-step buffer, rewards tensor now contain the sum of discounted rewards in the next
            # k steps (or up until termination, if terminated in less than k steps)
            rews_tensor = Tensor(batch['rews']).unsqueeze(1)
            # NOTE: given the multi-step buffer, done_tensor now shows whether the data's episode terminated in less
            # than k steps or not
            done_tensor = Tensor(batch['done']).unsqueeze(1)

            """
            now we do a SAC update, following the OpenAI spinup doc
            check the openai sac document psudocode part for reference
            line nubmers indicate lines in psudocode part
            we will first compute each of the losses
            and then update all the networks in the end
            """
            # see line 12: get a_tilda, which is newly sampled action (not action from replay buffer)

            """get q loss"""
            with torch.no_grad():
                a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = policy_net.forward(obs_next_tensor)
                q1_next = q1_target_net(torch.cat([obs_next_tensor, a_tilda_next], 1))
                q2_next = q2_target_net(torch.cat([obs_next_tensor, a_tilda_next], 1))

                # TODO: compute the k-step Q estiamte (in the form of reward + next Q), don't worry about the entropy terms
                if use_single_variant:
                    ### write code for computing the k-step estimate for the single Q estimate variant case
                    # target y = (gamma**0 * r1 + ... + gamma**(k-1) * rk) + gamma ** k * (1-d) * Q (not considering the entropy term)
                    # and rews_tensor already calculate the sum in the first pair of parenthesises
                    y_q = rews_tensor + gamma ** multistep_k * (1 - done_tensor) * q1_next
                else:
                    ### write code for computing the k-step estimate while using double clipped Q
                    # first get the compare Q1 and Q2 and get the min, then use that min to compute the target
                    min_next_q = torch.min(q1_next, q2_next)
                    y_q = rews_tensor + gamma ** multistep_k * (1 - done_tensor) * min_next_q

                # add the entropy, with a simplied heuristic way
                # NOTE: you don't need to modify the following 3 lines. They deal with entropy terms
                powers = np.arange(1, multistep_k+1) # k = 5 => posers = [1,2,3,4,5]
                entropy_discounted_sum = - sum(gamma ** powers) * (1 - done_tensor) * alpha * log_prob_a_tilda_next
                y_q += entropy_discounted_sum

            # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            q1_prediction = q1_net(torch.cat([obs_tensor, acts_tensor], 1))
            q1_loss = mse_criterion(q1_prediction, y_q)
            q2_prediction = q2_net(torch.cat([obs_tensor, acts_tensor], 1))
            q2_loss = mse_criterion(q2_prediction, y_q)

            """
            get policy loss
            """
            a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, _ = policy_net.forward(obs_tensor)

            # see line 12: second equation
            q1_a_tilda = q1_net(torch.cat([obs_tensor, a_tilda], 1))
            q2_a_tilda = q2_net(torch.cat([obs_tensor, a_tilda], 1))

            # TODO write code here to compute policy loss correctly, for both variants.
            if use_single_variant:
                # still pick Q1 network as the single network
                q_policy_part = q1_a_tilda
            else:
                # compare Q1 and Q2 to get the min
                q_policy_part = torch.min(q1_a_tilda, q2_a_tilda)

            # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            policy_loss = (alpha * log_prob_a_tilda - q_policy_part).mean()

            """
            alpha loss, update alpha
            """
            if auto_alpha:
                alpha_loss = -(log_alpha * (log_prob_a_tilda + target_entropy).detach()).mean()

                alpha_optim.zero_grad()
                alpha_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(log_alpha, grad_clip)
                alpha_optim.step()

                alpha = log_alpha.exp().item()
            else:
                alpha_loss = 0

            """update networks"""
            q1_optimizer.zero_grad()
            q1_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(q1_net.parameters(), grad_clip)
            q1_optimizer.step()

            q2_optimizer.zero_grad()
            q2_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(q2_net.parameters(), grad_clip)
            q2_optimizer.step()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
            policy_optimizer.step()

            # see line 16: update target value network with value network
            soft_update_model1_with_model2(q1_target_net, q1_net, polyak)
            soft_update_model1_with_model2(q2_target_net, q2_net, polyak)

            current_update_index += 1
            if current_update_index % logger_store_freq == 0:
                # store diagnostic info to logger
                logger.store(LossPi=policy_loss.item(), LossQ1=q1_loss.item(), LossQ2=q2_loss.item(),
                             LossAlpha=alpha_loss.item(),
                             Q1Vals=q1_prediction.detach().numpy(),
                             Q2Vals=q2_prediction.detach().numpy(),
                             Alpha=alpha,
                             LogPi=log_prob_a_tilda.detach().numpy())

        if d or (ep_len == max_ep_len):
            """when episode terminates, log info about this episode, then reset"""
            ## store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ## reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            """
            Save pytorch model, very different from tensorflow version
            We need to save the environment, the state_dict of each network
            and also the state_dict of each optimizer
            """
            if save_model:
                sac_state_dict = {'env':env,'policy_net':policy_net.state_dict(),
                                  'q1_net':q1_net.state_dict(), 'q2_net':q2_net.state_dict(),
                                  'q1_target_net': q1_target_net.state_dict(), 'q2_target_net': q2_target_net.state_dict(),
                                  'policy_opt':policy_optimizer,
                                  'q1_opt':q1_optimizer, 'q2_opt':q2_optimizer,
                                  'log_alpha':log_alpha, 'alpha_opt':alpha_optim, 'target_entropy':target_entropy}
                if (epoch % save_freq == 0) or (epoch == epochs-1):
                    logger.save_state(sac_state_dict, None)
            # use joblib.load(fname) to load

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # TODO write code here to estimate the bias of the Q networks
            #  recall that we can define the Q bias to be Q value - discounted MC return
            #  initialize another environment that is only used for provide such a bias estimate
            #  store that to logger

            # I eventually decided not to use buffer, because a buffer makes it a bit harder to discard the last 200 datapoints
            # but the discussion on buffer was enlightening Watcher! I learned a lot :)

            # this function is adapted from test_agent()
            def q_bias_analysis(N=1):
                # initialize another env
                # all b_XXX means bias
                b_env = env_fn()

                # loop for n episodes
                b_observations, b_actions, b_mc_returns = [], [], []
                for j in range(N):
                    b_o, b_r, b_d, b_ep_len = b_env.reset(), 0, False, 0
                    observations, actions, rewards, mc_returns = [], [], [], [0] # last return is 0
                    while not (b_d or (b_ep_len == max_ep_len)):
                        # add noise to action selection, so no deterministic action
                        b_a = policy_net.get_env_action(b_o, deterministic=False)
                        observations.append(b_o)
                        actions.append(b_a)
                        # b_env.render() # render the environment
                        # Step the env, get next observation, reward and done signal
                        b_o1, b_r, b_d, _ = b_env.step(b_a)
                        rewards.append(b_r)
                        b_ep_len += 1
                        b_o = b_o1
                    # check that everything is paired up
                    assert len(observations) == len(actions) == len(rewards)
                    # decide the cutoff point
                    if b_ep_len == 1000: # terminates because of reaching max limit, then discard the last 200
                        cut_idx = 800
                    else: # terminate before 1000 steps naturally then the MC return is accurate for later state-action pairs
                        cut_idx = len(observations)
                    b_observations += observations[:cut_idx]
                    b_actions += actions[:cut_idx]
                    G = 0
                    # loop the rewards list backward to calculate the returns
                    for r in reversed(rewards):
                        G = gamma * G + r
                        mc_returns.append(G)
                    mc_returns.reverse() # reverse [0, something, something, ...] into correct order
                    b_mc_returns += mc_returns[:cut_idx]

                # after rendering close env
                # b_env.close()

                # use b_obs, b_acts to calculate Q estimate
                b_obs_tensor = Tensor(b_observations)
                b_acts_tensor = Tensor(b_actions)
                if use_single_variant:
                    # as usual, choose q1 as single network
                    b_q_estimate = q1_net(torch.cat([b_obs_tensor, b_acts_tensor], 1))
                else:
                    # as usual, take the min
                    b_q1 = q1_net(torch.cat([b_obs_tensor, b_acts_tensor], 1))
                    b_q2 = q2_net(torch.cat([b_obs_tensor, b_acts_tensor], 1))
                    b_q_estimate = torch.min(b_q1, b_q2)

                # mc returns have been calculated but is still a list, so convert it into a tensor
                # need to unsqueeze it!!! need to unsqueeze it!!! need to unsqueeze it!!!
                # or a [1000, 1] tensor - a [1000] tensor would be a [1000, 1000] tensor
                # and you will get a 1e6 bias like I did ............
                b_mc_returns_tensor = Tensor(b_mc_returns).unsqueeze(1)

                # check all pairs are matched
                assert len(b_q_estimate) == len(b_mc_returns_tensor)

                # check how many steps each episode makes just for curiosity
                print('######### number of datapoints:', len(b_q_estimate))
                logger.log_tabular('NumDatapoints', len(b_q_estimate))

                # Q-bias = Q value - discounted MC return
                q_bias = b_q_estimate - b_mc_returns_tensor

                # put the result in buffer store (avg is taken care of, no need to calculate by hand)
                logger.store(QBias=q_bias.detach().numpy())

            # call q_bias_analysis function (running N episodes)
            q_bias_analysis(5)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)

            # TODO after you store bias info to logger, you should also write code here to log them
            #  so that you can later plot them
            logger.log_tabular('QBias', with_min_and_max=True)


            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
            sys.stdout.flush()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    # Note: these default arguments will be ignored when you import the function
    # from this file and run it in an experiment grid.

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    sac_multistep(lambda: gym.make(args.env), hidden_sizes=[args.hid] * args.l,
                  gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                  steps_per_epoch=args.steps_per_epoch, debug=args.debug,
                  logger_kwargs=logger_kwargs)
