# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-04-13 13:07:43
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-04-13 13:10:03
# Defines the agent. There's nothing special
# about the MOS agent in fact, except that
# it uses models defined in ..models, and
# makes use of the belief initialization
# functions in belief.py
import pomdp_py
from .belief import *
from ..models.transition_model import *
from ..models.observation_model import *
from ..models.reward_model import *
from ..models.policy_model import *


class MosAgent(pomdp_py.Agent):
    """One agent is one robot."""

    def __init__(self,
                 robot_id,
                 # initial robot state (assuming robot state is observable perfectly)
                 init_robot_state,
                 object_ids,  # target object ids
                 # tuple (w,l) of the width (w) and length (l) of the gridworld search space.
                 dim,
                 sensor,      # Sensor equipped on the robot
                 sigma=0.01,     # parameter for observation model
                 epsilon=1,   # parameter for observation model
                 # belief representation, either "histogram" or "particles".
                 belief_rep="histogram",
                 prior={},       # prior belief, as defined in belief.py:initialize_belief
                 num_particles=100,  # used if the belief representation is particles
                 grid_map=None):  # GridMap used to avoid collision with obstacles (None if not provided)
        self.robot_id = robot_id
        self._object_ids = object_ids
        self.sensor = sensor

        # since the robot observes its own pose perfectly, it will have 100% prior
        # on this pose.
        prior[robot_id] = {init_robot_state.pose: 1.0}
        rth = init_robot_state.pose[2]

        # initialize belief
        init_belief = initialize_belief(dim,
                                        self.robot_id,
                                        self._object_ids,
                                        prior=prior,
                                        representation=belief_rep,
                                        robot_orientations={
                                            self.robot_id: rth},
                                        num_particles=num_particles)
        transition_model = MosTransitionModel(dim,
                                              {self.robot_id: self.sensor},
                                              self._object_ids)
        observation_model = MosObservationModel(dim,
                                                self.sensor,
                                                self._object_ids,
                                                sigma=sigma,
                                                epsilon=epsilon)
        reward_model = GoalRewardModel(
            self._object_ids, robot_id=self.robot_id)
        policy_model = PolicyModel(self.robot_id, grid_map=grid_map)
        super().__init__(init_belief, policy_model,
                         transition_model=transition_model,
                         observation_model=observation_model,
                         reward_model=reward_model)

    def clear_history(self):
        """Custum function; clear history"""
        self._history = None
