import math
from sys import platform

if platform == "darwin":
    import numpy as cp
else:
    import numpy as cp
    # import cupy as cp

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
            self.H = 50

        # Context(Observation) space
        if 'n' in params:
            self.n = int(env_config['n'])
        else:
            self.n = 100
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
        if 'ps' in params:
            self.ps = env_config['ps']
        else:
            self.ps = []
            seeding_p = cp.random.RandomState(seed=10)  # "local" seeding to fix the transition probability matrices
            for a in range(self.A):
                p = seeding_p.rand(self.S, self.S)
                p /= p.sum(axis=1)[:, None]
                self.ps.append(p)

        # Generate context emission probability for each cluster!
        # All of them satisfies the regularity condition
        if 'qs' in params:
            self.qs = env_config['qs']
            if self.qs == 'uniform':
                self.qs = []
                for s in range(self.S):
                    q = cp.ones(self.cluster_sizes[s])
                    # q = cp.random.rand(self.cluster_sizes[s])
                    # q = seeding_q.rand(self.cluster_sizes[s])
                    q /= sum(q)
                    self.qs.append(q)
        else:
            self.qs = []
            seeding_q = cp.random.RandomState(seed=20)
            for s in range(self.S):
                # q = cp.random.rand(self.cluster_sizes[s])
                q = seeding_q.rand(self.cluster_sizes[s])
                q /= sum(q)
                self.qs.append(q)

        # latent state decoding function
        self.f = {}
        for s in range(self.S):
            cluster = self.partitions[s]
            for x in range(cluster.start, cluster.start + cluster.n):
                self.f[x] = s

    def compute_divergence(self):
        # to be done by Junghyun
        return None

    def step(self, action):
        if self.h == self.H:
            raise Exception("Exceeded horizon...")
        done = False
        if self.h == self.H - 1:
            done = True

        self.h += 1
        P = self.ps[action]
        tmp = cp.random.choice(range(self.S), size=1, p=P[self.state])
        self.state = int(tmp[0])
        obs = self.make_obs(self.state)
        return obs, done

    # Emission probability
    def make_obs(self, state):
        tmp = cp.random.choice(range(self.cluster_sizes[state]), size=1, p=self.qs[self.state])
        return self.partitions[state].start + int(tmp[0])

    def reset(self):
        if not self.initialized:
            raise Exception("Env not yet initialized!")
        self.h = 0
        self.state = self.latent_space.sample()  # uniformly random initial distribution over latent space
        obs = self.make_obs(self.state)
        return obs


def generate_trajectories(T, env):
    # Gather offline(logged) data, using uniformly random policy
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


def corrupt(env, trajectories, delta1=0.01, delta2=0.1, delta3=0.2):
    if delta1 == 0 or (delta2 == 0 and delta3 == 0):
        return trajectories
    T = len(trajectories)
    corrupt_idx = cp.random.choice(range(T), math.floor(delta1 * T))
    for i in corrupt_idx:
        trajectory = trajectories[i]
        if delta2 > 0:
            corrupt_X = cp.random.choice(range(env.H), math.floor(delta2 * env.H))
            for h in corrupt_X:
                states = list(range(env.S))
                x = trajectory[2 * h]
                states.remove(env.f[x])
                corrupt_s = cp.random.choice(states)
                trajectory[2 * h] = env.make_obs(corrupt_s)
        if delta3 > 0:
            corrupt_A = cp.random.choice(range(env.H), math.floor(delta3 * env.H))
            for h in corrupt_A:
                actions = list(range(env.A))
                a = trajectory[2 * h + 1]
                actions.remove(a)
                corrupt_a = cp.random.choice(actions)
                trajectory[2 * h + 1] = corrupt_a
        trajectories[i] = trajectory
    return trajectories


if __name__ == '__main__':
    T = 10
    env = Synthetic()
    trajectories = generate_trajectories(T, env)
    print(trajectories)
    print(corrupt(env, trajectories, delta1=0.1, delta2=0.2, delta3=0.3))
