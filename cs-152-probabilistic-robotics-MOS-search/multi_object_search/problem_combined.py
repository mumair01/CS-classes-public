# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-04-20 10:34:32
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-04-24 11:24:20


from email.policy import default
from tkinter.tix import IMAGE
from typing import Dict
from collections import defaultdict
import pomdp_py
from pomdp_problems.multi_object_search.env.env import *
from pomdp_problems.multi_object_search.env.visual import *
from pomdp_problems.multi_object_search.agent.agent import *
from pomdp_problems.multi_object_search.example_worlds import *
from pomdp_problems.multi_object_search.domain.observation import *
from pomdp_problems.multi_object_search.models.components.grid_map import *
import argparse
import time
import random
from matplotlib import pyplot as plt
import os
from datetime import datetime
import shutil
from tqdm import tqdm


class MosOOPOMDP(pomdp_py.OOPOMDP):
    """
    A MosOOPOMDP is instantiated given a string description
    of the search world, sensor descriptions for robots,
    and the necessary parameters for the agent's models.

    Note: This is of course a simulation, where you can
    generate a world and know where the target objects are
    and then construct the Environment object. But in the
    real robot scenario, you don't know where the objects
    are. In that case, as I have done it in the past, you
    could construct an Environment object and give None to
    the object poses.
    """

    def __init__(self, name, robot_id, env=None, grid_map=None,
                 sensors=None, sigma=0.01, epsilon=1,
                 belief_rep="histogram", prior={}, num_particles=100,
                 agent_has_map=False):
        """
        Args:
            robot_id (int or str): the id of the agent that will solve this MosOOPOMDP.
                If it is a `str`, it will be interpreted as an integer using `interpret_robot_id`
                in env/env.py.
            env (MosEnvironment): the environment.
            grid_map (str): Search space description. See env/env.py:interpret. An example:
                rx...
                .x.xT
                .....
                Ignored if env is not None
            sensors (dict): map from robot character to sensor string.
                For example: {'r': 'laser fov=90 min_range=1 max_range=5
                                    angle_increment=5'}
                Ignored if env is not None
            agent_has_map (bool): If True, we assume the agent is given the occupancy
                                  grid map of the world. Then, the agent can use this
                                  map to avoid planning invalid actions (bumping into things).
                                  But this map does not help the agent's prior belief directly.

            sigma, epsilon: observation model paramters
            belief_rep (str): belief representation. Either histogram or particles.
            prior (dict or str): either a dictionary as defined in agent/belief.py
                or a string, either "uniform" or "informed". For "uniform", a uniform
                prior will be given. For "informed", a perfect prior will be given.
            num_particles (int): setting for the particle belief representation
        """
        self.name = name
        if env is None:
            assert grid_map is not None and sensors is not None,\
                "Since env is not provided, you must provide string descriptions"\
                "of the world and sensors."
            worldstr = equip_sensors(grid_map, sensors)
            dim, robots, objects, obstacles, sensors = interpret(worldstr)
            init_state = MosOOState({**objects, **robots})
            env = MosEnvironment(dim,
                                 init_state, sensors,
                                 obstacles=obstacles)

        # construct prior
        # NOTE: uniform means no prior hints and informed means prior hints.
        if type(prior) == str:
            if prior == "uniform":
                prior = {}
            elif prior == "informed":
                prior = {}
                for objid in env.target_objects:
                    groundtruth_pose = env.state.pose(objid)
                    prior[objid] = {groundtruth_pose: 1.0}

        # Potential extension: a multi-agent POMDP. For now, the environment
        # can keep track of the states of multiple agents, but a POMDP is still
        # only defined over a single agent. Perhaps, MultiAgent is just a kind
        # of Agent, which will make the implementation of multi-agent POMDP cleaner.
        robot_id = robot_id if type(
            robot_id) == int else interpret_robot_id(robot_id)
        grid_map = GridMap(env.width, env.length,
                           {objid: env.state.pose(objid)
                            for objid in env.obstacles}) if agent_has_map else None
        agent = MosAgent(robot_id,
                         env.state.object_states[robot_id],
                         env.target_objects,
                         (env.width, env.length),
                         env.sensors[robot_id],
                         sigma=sigma,
                         epsilon=epsilon,
                         belief_rep=belief_rep,
                         prior=prior,
                         num_particles=num_particles,
                         grid_map=grid_map)
        super().__init__(agent, env,
                         name="MOS(%d,%d,%d)" % (env.width, env.length, len(env.target_objects)))


# def dynamic_hint(trust, problem):
#     objids = problem.env.target_objects
#     target_hint = random.sample(objids, 1)[0]
#     # random_object_belief = problem.agent.belief.object_beliefs[target_hint];
#     # groundtruth_pose = problem.env.state.pose(target_hint)
#     target_state = problem.env.state.object_states[target_hint]
#     prior_of_target_hint = {target_state: trust}  # prior = problem.agent.prior
#     new_belief = pomdp_py.Histogram(prior_of_target_hint)
#     problem.agent.cur_belief.set_object_belief(target_hint, new_belief)


def dynamic_hint(trust, problem):
    objids = problem.env.target_objects
    other_trust = (1 - trust) / (len(problem.env.target_objects) - 1)
    target_hint_id = random.sample(objids, 1)[0]
    objid = target_hint_id
    belief_obj = problem.agent.cur_belief.object_belief(objid)
    new_histogram = {}
    for next_state in belief_obj:
        if next_state.objid != target_hint_id:
            observation_prob = other_trust
        else:
            observation_prob = trust
        transition_prob = belief_obj[next_state]
        new_histogram[next_state] = observation_prob * transition_prob
    new_belief = pomdp_py.Histogram(new_histogram)
    problem.agent.cur_belief.set_object_belief(target_hint_id, new_belief)


### Belief Update ###


def belief_update(agent, real_action, real_observation, next_robot_state, planner):
    """Updates the agent's belief; The belief update may happen
    through planner update (e.g. when planner is POMCP)."""
    # Updates the planner; In case of POMCP, agent's belief is also updated.
    planner.update(agent, real_action, real_observation)

    # Update agent's belief, when planner is not POMCP
    if not isinstance(planner, pomdp_py.POMCP):
        # Update belief for every object
        for objid in agent.cur_belief.object_beliefs:
            belief_obj = agent.cur_belief.object_belief(objid)
            if isinstance(belief_obj, pomdp_py.Histogram):
                if objid == agent.robot_id:
                    # Assuming the agent can observe its own state:
                    new_belief = pomdp_py.Histogram({next_robot_state: 1.0})
                else:
                    # This is doing
                    #    B(si') = normalizer * O(oi|si',sr',a) * sum_s T(si'|s,a)*B(si)
                    #
                    # Notes: First, objects are static; Second,
                    # O(oi|s',a) ~= O(oi|si',sr',a) according to the definition
                    # of the observation model in models/observation.py.  Note
                    # that the exact belief update rule for this OOPOMDP needs to use
                    # a model like O(oi|si',sr',a) because it's intractable to
                    # consider s' (that means all combinations of all object
                    # states must be iterated).  Of course, there could be work
                    # around (out of scope) - Consider a volumetric observaiton,
                    # instead of the object-pose observation. That means oi is a
                    # set of pixels (2D) or voxels (3D). Note the real
                    # observation, oi, is most likely sampled from O(oi|s',a)
                    # because real world considers the occlusion between objects
                    # (due to full state s'). The problem is how to compute the
                    # probability of this oi given s' and a, where it's
                    # intractable to obtain s'. To this end, we can make a
                    # simplifying assumption that an object is contained within
                    # one pixel (or voxel); The pixel (or voxel) is labeled to
                    # indicate free space or object. The label of each pixel or
                    # voxel is certainly a result of considering the full state
                    # s. The occlusion can be handled nicely with the volumetric
                    # observation definition. Then that assumption can reduce the
                    # observation model from O(oi|s',a) to O(label_i|s',a) and
                    # it becomes easy to define O(label_i=i|s',a) and O(label_i=FREE|s',a).
                    # These ideas are used in my recent 3D object search work.
                    new_belief = pomdp_py.update_histogram_belief(belief_obj,
                                                                  real_action,
                                                                  real_observation.for_obj(
                                                                      objid),
                                                                  agent.observation_model[objid],
                                                                  agent.transition_model[objid],
                                                                  # The agent knows the objects are static.
                                                                  static_transition=objid != agent.robot_id,
                                                                  oargs={"next_robot_state": next_robot_state})
            else:
                raise ValueError("Unexpected program state."
                                 "Are you using the appropriate belief representation?")

            agent.cur_belief.set_object_belief(objid, new_belief)


### Solve the problem with POUCT/POMCP planner ###
### This is the main online POMDP solver logic ###
def solve(problem,
          max_depth=10,  # planning horizon
          discount_factor=0.99,
          planning_time=1.,       # amount of time (s) to plan each step
          exploration_const=1000,  # exploration constant
          visualize=True,
          max_time=120,  # maximum amount of time allowed to solve the problem
          # maximum number of planning steps the agent can take.
          max_steps=500,
          sim_mode=None):  # Mode of the simulation that's currently running.
    """
    This function terminates when:
    - maximum time (max_time) reached; This time includes planning and updates
    - agent has planned `max_steps` number of steps
    - agent has taken n FindAction(s) where n = number of target objects.

    Args:
        visualize (bool) if True, show the pygame visualization.
    """

    # This dictionary stores the results of the simulation over time.
    results = defaultdict(lambda: list())
    # print("results initially empty ", results)
    random_objid = random.sample(problem.env.target_objects, 1)[0]
    random_object_belief = problem.agent.belief.object_beliefs[random_objid]
    if isinstance(random_object_belief, pomdp_py.Histogram):
        # Use POUCT
        planner = pomdp_py.POUCT(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    elif isinstance(random_object_belief, pomdp_py.Particles):
        # Use POMCP
        planner = pomdp_py.POMCP(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    else:
        raise ValueError("Unsupported object belief type %s" %
                         str(type(random_object_belief)))

    robot_id = problem.agent.robot_id
    if visualize:
        # controllable=False means no keyboard control.
        viz = MosViz(problem.env, controllable=False)
        if viz.on_init() == False:
            raise Exception("Environment failed to initialize")
        viz.update(robot_id,
                   None,
                   None,
                   None,
                   problem.agent.cur_belief)
        viz.on_render()

    _time_used = 0
    _find_actions_count = 0
    _total_reward = 0  # total, undiscounted reward
    for i in range(max_steps):
        if sim_mode == "dynamic_hints":
            # TODO: Change this value of this trust parameter over time.
            dynamic_hint(0.999, problem)
        # Plan action
        _start = time.time()
        real_action = planner.plan(problem.agent)
        _time_used += time.time() - _start
        if _time_used > max_time:
            break  # no more time to update.

        # Execute action
        reward = problem.env.state_transition(real_action, execute=True,
                                              robot_id=robot_id)

        # Receive observation
        _start = time.time()
        real_observation = \
            problem.env.provide_observation(
                problem.agent.observation_model, real_action)

        # Updates
        problem.agent.clear_history()  # truncate history
        problem.agent.update_history(real_action, real_observation)
        belief_update(problem.agent, real_action, real_observation,
                      problem.env.state.object_states[robot_id],
                      planner)
        plan_time = time.time() - _start
        _time_used += plan_time

        # Info and render
        _total_reward += reward
        if isinstance(real_action, FindAction):
            _find_actions_count += 1
        # print("==== Step %d ====" % (i+1))
        # print("Action: %s" % str(real_action))
        # print("Observation: %s" % str(real_observation))
        # print("Reward: %s" % str(reward))
        # print("Reward (Cumulative): %s" % str(_total_reward))
        # print("Find Actions Count: %d" % _find_actions_count)
        # if isinstance(planner, pomdp_py.POUCT):
        #     print("__num_sims__: %d" % planner.last_num_sims)
        # for objid in problem.env.target_objects:
        #     if objid in set(problem.env.state.object_states[robot_id].objects_found):
        #         print("{} --> FOUND".format(objid))
        #     else:
        #         print("{} --> NOT FOUND")

        # if isinstance(planner, pomdp_py.POUCT):
        #     print("__num_sims__: %d" % planner.last_num_sims)

        # Store the data for later analysis
        results["step_time_used"].append(plan_time)
        results["observations_over_time"].append(real_observation)
        results["reward_over_time"].append(reward)
        results["actions_over_time"].append(real_action)
        results["total_reward"].append(_total_reward)
        hists = {}
        for objid, belief in problem.agent.cur_belief.object_beliefs.items():
            if objid != robot_id:
                # print("objid: {}".format(objid))
                # print("mpe: {}".format(belief.mpe()))
                # print("real object state: {}".format(
                #     problem.env.state.pose(objid)))
                hists[objid] = belief.get_histogram()
        results["beliefs"].append(hists)

        if visualize:
            # This is used to show the sensing range; Not sampled
            # according to observation model.
            robot_pose = problem.env.state.object_states[robot_id].pose
            viz_observation = MosOOObservation({})
            if isinstance(real_action, LookAction) or isinstance(real_action, FindAction):
                viz_observation = \
                    problem.env.sensors[robot_id].observe(robot_pose,
                                                          problem.env.state)
            viz.update(robot_id,
                       real_action,
                       real_observation,
                       viz_observation,
                       problem.agent.cur_belief)
            viz.on_loop()
            viz.on_render()

        # Termination check
        if set(problem.env.state.object_states[robot_id].objects_found)\
           == problem.env.target_objects:
            # print("Done!")
            time.sleep(1)
            break
        if _find_actions_count >= len(problem.env.target_objects):
            print("FindAction limit reached.")
            break
        if _time_used > max_time:
            print("Maximum time reached.")
            break
    return results


def problem_no_hints(grid_map, robot_char):
    # random world
    # NOTE: The laser sensor was not being used in the original simulation.
    # laserstr = make_laser_sensor(90, (1, 4), 0.5, False)
    proxstr = make_proximity_sensor(4, False)
    return MosOOPOMDP(
        "no_hints",
        robot_char,  # r is the robot character
        sigma=0.05,  # observation model parameter
        epsilon=0.95,  # observation model parameter
        grid_map=grid_map,
        sensors={robot_char: proxstr},
        prior="uniform",
        agent_has_map=True)


def problem_onset_hints(grid_map, robot_char):
    # random world
    # NOTE: The laser sensor was not being used in the original simulation.
    # laserstr = make_laser_sensor(90, (1, 4), 0.5, False)
    proxstr = make_proximity_sensor(4, False)
    return MosOOPOMDP(
        "onset_hints",
        robot_char,  # r is the robot character
        sigma=0.05,  # observation model parameter
        epsilon=0.95,  # observation model parameter
        grid_map=grid_map,
        sensors={robot_char: proxstr},
        prior="informed",
        agent_has_map=True)


def problem_dynamic_hints(grid_map, robot_char):
    # random world
    # NOTE: The laser sensor was not being used in the original simulation.
    # laserstr = make_laser_sensor(90, (1, 4), 0.5, False)
    proxstr = make_proximity_sensor(4, False)
    return MosOOPOMDP(
        "dynamic_hints",
        robot_char,  # r is the robot character
        sigma=0.05,  # observation model parameter
        epsilon=0.95,  # observation model parameter
        grid_map=grid_map,
        sensors={robot_char: proxstr},
        prior="uniform",
        agent_has_map=True)


def visualize_results(problem, results: Dict):
    print("No. of time steps: {}".format(len(results["reward_over_time"])))
    print(results["step_time_used"])
    plt.plot(range(len(results["step_time_used"])), results["step_time_used"])
    plt.title("Belief update time per step - {}".format(problem.name))
    plt.xlabel("Time Step")
    plt.ylabel("Planning Time Seconds")
    plt.show(block=True)
    print(results["reward_over_time"])
    plt.plot(range(len(results["reward_over_time"])),
             results["reward_over_time"])
    plt.title("Reward over time - {}".format(problem.name))
    plt.xlabel("Time Step")
    plt.ylabel("Reward at Current Time")
    plt.show(block=True)
    plt.plot(range(len(results["total_reward"])),
             results["total_reward"])
    plt.title("Cumulative Reward over time - {}".format(problem.name))
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Reward at Current Time")
    plt.show(block=True)
    # Getting all hists for the last time step.
    # for objid in problem.env.target_objects:
    #     # Get the belief over time
    #     for data in results["beliefs"]:
    #         obj_hist = data[objid]
    #         true_state = problem.env.state(objid)
    #         print(true_state)
    #         print(obj_hist[true_state])

    # last_hists = results["beliefs"][-1]
    # # PLot all the target hists one by one
    # for objid, hist in last_hists.items():
    #     x = hist.keys()
    #     print("hist keys")
    #     print(x)
    #     y = hist.values()
    #     print("hist values")
    #     print(y)
    #     plt.plot(y)
    #     plt.show(block=True)


def save_fig(images_dir, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(images_dir, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def convert_to_numpy_pad(a_list, pad=np.nan):
    """
    Pad a list of lists and convert to 2D array.
    """
    row_lengths = []
    for row in a_list:
        row_lengths.append(len(row))
    max_length = max(row_lengths)
    for row in a_list:
        while len(row) < max_length:
            row.append(pad)
    return np.array(a_list)


def average_cumulative_reward_over_time(simulation_results, visualize=True):
    # print("average_cumulative_reward_over_time")
    # Each key dim should be: num_exp x steps_taken.
    problem_results = defaultdict(lambda: list())
    for results in simulation_results:
        for problem, result in results.items():
            problem_results[problem.name].append(result["total_reward"])
    avg_cumulative_rewards = {}
    for k, v in problem_results.items():
        problem_results[k] = convert_to_numpy_pad(v)
        avg_cumulative_rewards[k] = np.nanmean(problem_results[k], axis=0)
        # print(avg_cumulative_rewards[k].shape)
    if visualize:
        for k, avg_rewards in avg_cumulative_rewards.items():
            plt.plot(avg_rewards, label="{}".format(k))
            plt.title("Average Cumulative Reward over time")
            plt.xlabel("Time Step")
            plt.ylabel("Average Cumulative Reward at Current Time")
        plt.legend()
        save_fig(IMAGES_DIR, "average_cumulative_reward_over_time")
        plt.show(block=True)
    return avg_cumulative_rewards


def average_planning_time_per_step(simulation_results, visualize=True):
    # print("average_planning_time_per_step")
    # Each key dim should be: num_exp x steps_taken.
    problem_results = defaultdict(lambda: list())
    for results in simulation_results:
        for problem, result in results.items():
            problem_results[problem.name].append(result["step_time_used"])
    avg_cumulative_rewards = {}
    for k, v in problem_results.items():
        problem_results[k] = convert_to_numpy_pad(v)
        avg_cumulative_rewards[k] = np.nanmean(problem_results[k], axis=0)
        # print(avg_cumulative_rewards[k].shape)
    if visualize:
        for k, avg_rewards in avg_cumulative_rewards.items():
            plt.plot(avg_rewards, label="{}".format(k))
            plt.title("Average planning time per step")
            plt.xlabel("Time Step")
            plt.ylabel("Average planning time per step")
        plt.legend()
        save_fig(IMAGES_DIR, "average_planning_time_per_step")
        plt.show(block=True)
    return avg_cumulative_rewards


def average_step_reward_over_time(simulation_results, visualize=True):
    # print("average_step_reward_over_time")
    # Each key dim should be: num_exp x steps_taken.
    problem_results = defaultdict(lambda: list())
    for results in simulation_results:
        for problem, result in results.items():
            problem_results[problem.name].append(result["reward_over_time"])
    avg_cumulative_rewards = {}
    for k, v in problem_results.items():
        problem_results[k] = convert_to_numpy_pad(v)
        avg_cumulative_rewards[k] = np.nanmean(problem_results[k], axis=0)
        # print(avg_cumulative_rewards[k].shape)
    if visualize:
        for k, avg_rewards in avg_cumulative_rewards.items():
            plt.plot(avg_rewards, label="{}".format(k))
            plt.title("Average reward per step over time")
            plt.xlabel("Time Step")
            plt.ylabel("Average reward at the current time step")
        plt.legend()
        save_fig(IMAGES_DIR, "average_step_reward_over_time")
        plt.show(block=True)
    return avg_cumulative_rewards


def average_number_of_steps(simulation_results):
    avg_num_steps = defaultdict(lambda: 0)
    for results in simulation_results:
        for problem, result in results.items():
            avg_num_steps[problem.name] += len(result["total_reward"])
    for problem in results.keys():
        avg_num_steps[problem.name] = avg_num_steps[problem.name] / \
            len(simulation_results)
    return avg_num_steps


def analyze_results(simulation_results):
    # These methods create visualizations
    average_cumulative_reward_over_time(simulation_results)
    average_planning_time_per_step(simulation_results)
    average_step_reward_over_time(simulation_results)
    # These methods create numeric results
    average_number_of_steps(simulation_results)


def initialize_and_solve_problems(sim_mode):
    # We want to use the same world across runs - seed the random
    random.seed(42)
    # We want to use the same random world for all the problems
    grid_map, robot_char = random_world(10, 10, 5, 10)
    # Initialize the problems
    no_hints = problem_no_hints(grid_map, robot_char)
    onset_hints = problem_onset_hints(grid_map, robot_char)
    dynamic_hints = problem_dynamic_hints(grid_map, robot_char)
    # Generate the problems
    problems = []
    if sim_mode == "no_hints":
        problems.append(no_hints)
    elif sim_mode == "onset_hints":
        problems.append(onset_hints)
    elif sim_mode == "dynamic_hints":
        problems.append(dynamic_hints)
    elif sim_mode == "all_sims":
        problems.extend([no_hints, onset_hints, dynamic_hints])
    else:
        raise Exception("sim_mode invalid")
    # Solve the problems
    results = {}
    for problem in problems:
        result = solve(
            problem=problem,
            max_depth=10,
            discount_factor=0.99,
            planning_time=1.,
            exploration_const=1000,
            visualize=True,
            max_time=120,
            max_steps=500,
            sim_mode=args.sim_mode)
        results[problem] = result
    return results


if __name__ == "__main__":
    sim_modes = ('no_hints', 'onset_hints', 'dynamic_hints', 'all_sims')
    parser = argparse.ArgumentParser()
    parser.add_argument('--simMode', '-sm', dest="sim_mode", type=str,
                        required=True,
                        help='Mode of the simulation - one of {}'.format(sim_modes))
    parser.add_argument('--numExp', '-ne', dest="num_exps", type=int,
                        required=True,
                        help='Number of experiments')
    args = parser.parse_args()
    assert args.sim_mode in sim_modes
    assert args.num_exps > 0
    # Create the dirs.
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    PROJECT_ROOT_DIR = "simulation_results"
    if PROJECT_ROOT_DIR != ".":
        # if os.path.isdir(PROJECT_ROOT_DIR):
        #     shutil.rmtree(PROJECT_ROOT_DIR)
        os.makedirs(PROJECT_ROOT_DIR, exist_ok=True)
    SIM_DIR = os.path.join(PROJECT_ROOT_DIR,
                           "{}_{}_{}".format(current_time, args.sim_mode, args.num_exps))
    os.makedirs(SIM_DIR, exist_ok=True)
    IMAGES_DIR = os.path.join(SIM_DIR, "images")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    # Run the simulation
    simulation_results = []
    for _ in tqdm(range(args.num_exps)):
        results = initialize_and_solve_problems(args.sim_mode)
        simulation_results.append(results)
    analyze_results(simulation_results)
