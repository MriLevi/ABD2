import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import random
import time
import tcod
import pandas as pd
import sys


# in deze file staan de 3 classes voor batch simulations. Er is er voor gekozen om de aanpassingen hier te doen i.p.v. in
# de environment class zelf, omdat het hier altijd op dezelfde plek kan: de run loop.
class AuctionBatchSimulation():
    def __init__(self, env, runs, random_seeds=None):
        self.random_seeds = random_seeds
        self.runs = runs
        self.current_run = 0
        self.df = pd.DataFrame(columns=['points', 'steps_at_termination'])
        self.env = env

    def auction(self, tasks: list, value_matrix: dict):
        """"het auction algoritme. Het returned een dict met assignments (collect at coordinate OF explore)
        """
        env = self.env

        # make variables to keep track of agent task pairs and task prices
        agent_task_pairs = {}
        prices = {}
        for task in tasks:
            prices[task] = 0

        # auction all tasks until all agents have a task or all food tasks are paired with a agent
        while True:

            # pick an unpaired agent
            for agent_id in range(len(env.agents)):
                if agent_id in agent_task_pairs.values():
                    continue

                # pick the task with maximal value for agent at current prices
                task_values = []
                max_task = None
                max_task_value = 0
                for task in tasks:
                    current_task_value = value_matrix[agent_id][task] - prices[task]
                    task_values.append(current_task_value)
                    if current_task_value > max_task_value:
                        max_task_value = current_task_value
                        max_task = task

                # compute bid increment on maximal value task
                task_values.sort(reverse=True)

                bid_increment = max_task_value - task_values[1] + 1

                # add agent task pair and delete pairs with the same task
                agent_task_pairs[max_task] = agent_id

                # increase task price
                prices[max_task] += bid_increment

                if len(agent_task_pairs) == len(tasks) or len(agent_task_pairs) == len(env.agents):
                    return {v: k for k, v in agent_task_pairs.items()}

    def run(self):

        env = self.env

        for run in range(self.runs):
            print(f"{run}/{self.runs}")

            if self.random_seeds:
                env.random_seed = self.random_seeds[run]

            env.reset()

            observation = env.get_obs()
            while True:

                shared_observation = env.get_shared_observation()

                good_food_locs = shared_observation['good_food_locs']

                cost_dict = {
                    agent_ind: {str(k): 1000 - env.agents[agent_ind].get_cost_to_target(k, shared_observation) for k in
                                good_food_locs} for agent_ind in range(len(env.agents))}
                for i in range(len(env.agents)):
                    for j in range(len(env.agents)):
                        cost_dict[i][f"exploration{j}"] = 100 - j - 1

                target_dict = self.auction(
                    [str(gfl) for gfl in good_food_locs] + [f'exploration{j}' for j in range(len(env.agents))],
                    cost_dict)
                for i in range(len(env.agents)):
                    try:
                        target = eval(target_dict[i])
                    except NameError:
                        target = None

                    env.agents[i].shared_observation = shared_observation
                    action = [env.agents[i].run(observation, target)]
                    observation, reward, terminated, info = env.step(action, i)

                if terminated:
                    break

            self.update_stats()
            self.current_run += 1

    def update_stats(self):

        env = self.env
        new_row = pd.Series(
            {'points': sum([agent.points for agent in env.agents]), 'steps_at_termination': env.current_step})

        self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)


class NoAuctionSharedMemoryBatchSimulation():
    def __init__(self, env, runs, random_seeds=None):
        self.random_seeds = random_seeds
        self.runs = runs
        self.current_run = 0
        self.df = pd.DataFrame(columns=['points', 'steps_at_termination'])
        self.env = env

    def run(self):
        env = self.env

        for run in range(self.runs):
            if self.random_seeds:
                env.random_seed = self.random_seeds[run]
            print(f"{run}/{self.runs}")

            env.reset()

            observation = env.get_obs()
            while True:

                for i in range(len(env.agents)):
                    shared_observation = env.get_shared_observation()

                    good_food_options = shared_observation['good_food_locs']

                    if len(good_food_options) > 0:

                        lowest_cost_index = np.argmin(
                            [env.agents[i].get_cost_to_target(gf, shared_observation) for gf in good_food_options])
                        target = good_food_options[lowest_cost_index]
                    else:
                        target = None

                    env.agents[i].shared_observation = shared_observation
                    action = [env.agents[i].run(observation, target)]
                    observation, reward, terminated, info = env.step(action, i)

                if terminated:
                    break

            self.update_stats()
            self.current_run += 1

    def update_stats(self):
        env = self.env

        new_row = pd.Series(
            {'points': sum([agent.points for agent in env.agents]), 'steps_at_termination': env.current_step})

        self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)


class NoAuctionNoSharedMemoryBatchSimulation():
    def __init__(self, env, runs, random_seeds=None):
        self.random_seeds = random_seeds
        self.runs = runs
        self.current_run = 0
        self.df = pd.DataFrame(columns=['points', 'steps_at_termination'])
        self.env = env

    def run(self):
        env = self.env

        for run in range(self.runs):

            print(f"{run}/{self.runs}")

            if self.random_seeds:
                env.random_seed = self.random_seeds[run]
            env.reset()

            observation = env.get_obs()
            while True:

                for i in range(len(env.agents)):
                    shared_observation = env.get_shared_observation()

                    clipped_obs = env.agents[i].clip_vision(shared_observation)

                    good_food_options = clipped_obs['good_food_locs']

                    if len(good_food_options) > 0:

                        lowest_cost_index = np.argmin(
                            [env.agents[i].get_cost_to_target(gf, clipped_obs) for gf in good_food_options])
                        target = good_food_options[lowest_cost_index]
                    else:
                        target = None

                    # fake shared observation
                    env.agents[i].shared_observation = clipped_obs
                    action = [env.agents[i].run(observation, target)]
                    observation, reward, terminated, info = env.step(action, i)

                if terminated:
                    break

            self.update_stats()
            self.current_run += 1

    def update_stats(self):
        env = self.env

        new_row = pd.Series(
            {'points': sum([agent.points for agent in env.agents]), 'steps_at_termination': env.current_step})

        self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)
