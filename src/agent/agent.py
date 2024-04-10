
import numpy as np
import pyRDDLGym
import gym
import io
from pyRDDLGym.core.policy import RandomAgent
from aivle_gym.agent_env import AgentEnv
from aivle_gym.env_serializer import SampleSerializer
from gym.wrappers.monitoring.video_recorder import VideoRecorder


"""
Frozen lake enviroment for Value Iteration. Value iteration algorithm coded from scratch.
"""
class FrozenLakeAgentEnv(AgentEnv):
    def __init__(self, port: int):
        self.base_env = gym.make("FrozenLake-v1", render_mode="human")

        super().__init__(
            SampleSerializer(),
            self.base_env.action_space,
            self.base_env.observation_space,
            self.base_env.reward_range,
            uid=0,
            port=port,
            env=self.base_env,
        )  # uid can be any int for single-agent agent env

    def create_agent(self, **kwargs):
        agent = ValueIterationAgent(env=self.base_env, gamma=0.99, theta=0.00001, max_iterations=10000)
        agent.initialize()

        return agent

class ValueIterationAgent(object):
    def __init__(self, env=None,
                       gamma=0.99,
                       theta = 0.00001,
                       max_iterations=10000):

        self.env = env

        # Set of discrete actions for evaluator environment, shape - (|A|)
        self.disc_actions = env.action_space

        # Set of discrete states for evaluator environment, shape - (|S|)
        self.disc_states = env.observation_space

        # Set of probabilities for transition function for each action from every states, dicitonary of dist[s] = [s', prob, done, info]
        self.Prob = env.P

        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        # self.value_policy, self.policy_function = None, None

    def initialize(self):
        self.value_policy, self.policy_function = self.solve_value_iteration()


    def step(self, state):
        print(state)
        action = self.policy_function[int(state)]
        return action

    def calc_value_action(self, state, value_policy, nA):
            value_action_result = np.zeros(nA)
            for action in range(nA):
                for transition in self.Prob[state][action]:
                    transition_prob, next_state, reward, _ = transition
                    value_action_result[action] += transition_prob * (reward + self.gamma*value_policy[next_state])
            return value_action_result

    def solve_value_iteration(self):
        '''
        Value iteration algorithm coded from scratch
        return:
            value_policy (shape - (|S|)): utility value for each state
            policy_function (shape - (|S|), dtype = int64): action policy per state
        '''

        nS = self.env.observation_space.n
        nA = self.env.action_space.n
        value_policy = np.zeros(nS)
        policy_function = np.zeros(nS, dtype=int)
        error_threshold = self.theta*(1-self.gamma)/self.gamma

        for _ in range(self.max_iterations):
            max_error = 0
            for state in range(nS):
                value_action = self.calc_value_action(state, value_policy, nA)
                best_act_value = np.max(value_action)
                max_error = max(max_error, np.abs(best_act_value - value_policy[state]))
                value_policy[state] = best_act_value

                best_action = np.argmax(value_action)
                policy_function[state] = best_action

            if max_error <= error_threshold:
                break

        return value_policy, policy_function


def main():

    is_render = True
    render_path = 'temp_vis'
    env = gym.make("FrozenLake-v1", render_mode="rgb_array")

    agent_env = FrozenLakeAgentEnv(0)
    agent = agent_env.create_agent()
    state, _ = env.reset()

    total_reward = 0
    terminated = False
    video_path = "temp_vis/value_iteration.mp4"
    video = VideoRecorder(env, video_path)


    # for t in range(env.horizon):
    while not terminated:
        action = agent.step(state)
        next_state, reward, terminated, info, _ = env.step(action)  # self.env.step(action)

        total_reward += reward
        print()
        print(f'state      = {state}')
        print(f'action     = {action}')
        print(f'next state = {next_state}')
        print(f'reward     = {reward}')
        print(f'total_reward     = {total_reward}')
        state = next_state

        if is_render:
            video.capture_frame()

    env.close()



if __name__ == "__main__":
    main()
