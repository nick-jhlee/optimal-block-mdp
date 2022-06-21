import numpy as np
import gym
from gym.spaces import Discrete


class Synthetic(gym.Env):
    """
    Synthetic MDP in reward-free exploration, model-estimation setting
    H, n, S, A : change via env_config
    """

    def __init__(self, env_config={}):
        self.initialized = True
        params = env_config.keys()

        # Horizon length
        if 'H' in params:
            self.H = int(env_config['H'])
        else:
            self.H = 100

        # Context(Observation) space
        if 'n' in params:
            self.n = int(env_config['n'])
        else:
            self.n = 200
        self.observation_space = Discrete(self.n)

        # Latent state space
        if 'S' in params:
            self.S = int(env_config['S'])
        else:
            self.S = 2
        self.latent_space = Discrete(self.S)

        # For simplicity, suppose that the sizes of all clusters are all the same
        self.alphas = [1 / self.S for _ in range(self.S)]

        # Partitioned context space
        self.partitions = []
        self.cluster_sizes = []
        for i in range(self.S):
            size = int(self.n * self.alphas[i])
            self.cluster_sizes.append(size)
            self.partitions.append(Discrete(size, start=i * size))

        # Action space
        if 'A' in params:
            self.A = int(env_config['A'])
        else:
            self.A = 2
        self.action_space = Discrete(self.A)

        # Generate latent transition probability matrix for each action!
        # All of them satisfies the regularity condition
        self.ps = []
        # seeding_p = np.random.RandomState(seed=100)  # "local" seeding to fix the transition probability matrices
        for a in range(self.A):
            p = np.random.rand(self.S, self.S)
            # p = seeding_p.rand(self.S, self.S)
            p /= p.sum(axis=1)[:, None]
            self.ps.append(p)

        # Generate context emission probability for each cluster!
        # All of them satisfies the regularity condition
        self.qs = []
        # seeding_q = np.random.RandomState(seed=1000)
        for s in range(self.S):
            q = np.random.rand(self.cluster_sizes[s])
            # q = seeding_q.rand(self.cluster_sizes[s])
            q /= sum(q)
            self.qs.append(q)

    def compute_divergence(self):
        return None

    def step(self, action):
        if self.h == self.H:
            raise Exception("Exceeded horizon...")
        done = False
        if self.h == self.H - 1:
            done = True

        self.h += 1
        P = self.ps[action]
        self.state = np.random.choice(range(self.latent_space.n), p=P[self.state])
        q = self.qs[self.state]
        obs = self.make_obs(self.state, q)
        return obs, done

    # Uniformly random emission probability
    def make_obs(self, state, q):
        return np.random.choice(range(self.cluster_sizes[state]), p=q)

    def reset(self):
        if not self.initialized:
            raise Exception("Env not yet initialized!")
        self.h = 0
        self.state = self.latent_space.sample()  # uniformly random initial distribution over latent space
        q = self.qs[self.state]
        obs = self.make_obs(self.state, q)
        return obs


def generate_trajectories(T, env):
    # Gather offline(logged) data
    # env.init()
    trajectories = [[] for _ in range(T)]
    for t in range(T):
        o = env.reset()
        trajectories[t].append(o)
        done = False
        h = 0
        while not done:
            action = env.action_space.sample()
            o, done = env.step(action)
            trajectories[t] += [action, o]
            h += 1
    return trajectories


if __name__ == '__main__':
    T = 10
    env = Synthetic()
    trajectories = generate_trajectories(T, env)
    print(trajectories)
