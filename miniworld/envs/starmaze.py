import math

import numpy as np
from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.math import gen_rot_matrix
from miniworld.miniworld import MiniWorldEnv


class StarMaze(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Two hallways connected in a Star-junction. The goal is to move the agent
    towards a red box within as little steps as possible. In
    `MiniWorld-YMazeLeft-v0`, the red box is located on the left wing of
    the Y-shaped junction. In `MiniWorld-YMazeRight-v0`,  the red box is
    located on the right wing of the Y-shaped junction.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +(1 - 0.2 * (step_count / max_episode_steps)) when box reached

    ## Arguments

    ```python
    env = gym.make("MiniWorld-YMazeLeft-v0")
    # or
    env = gym.make("MiniWorld-YMazeRight-v0")
    ```

    """

    def __init__(self, goal_pos=None, **kwargs):
        self.goal_pos = goal_pos

        MiniWorldEnv.__init__(self, max_episode_steps=280, **kwargs)
        utils.EzPickle.__init__(self, goal_pos, **kwargs)

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # Outline of the main (starting) arm
        main_outline = np.array(
            [
                [+1.18,0,+1.62],
                [-1.18,0,+1.62],
                [-1.18,0,+6.62],
                [+1.18,0,+6.62],
            ]
        )

        main_arm = self.add_room(outline=np.delete(main_outline, 1, 1))

        # Star-shaped hub room, outline of XZ points
        hub_room = self.add_room(
            outline=np.array(
                [
                    [0, -2],
                    [-1.9, -0.62],
                    [-1.18, +1.62],
                    [+1.18, +1.62],
                    [+1.9, -0.62]
                ]
            )
        )

        # Other arms of the maze
        m = gen_rot_matrix(np.array([0, 1, 0]), -72 * (math.pi / 180))
        arm2_outline = np.dot(main_outline, m)
        arm2 = self.add_room(outline=np.delete(arm2_outline, 1, 1))

        m = gen_rot_matrix(np.array([0, 1, 0]), -144 * (math.pi / 180))
        arm3_outline = np.dot(main_outline, m)
        arm3 = self.add_room(outline=np.delete(arm3_outline, 1, 1))

        m = gen_rot_matrix(np.array([0, 1, 0]), 144 * (math.pi / 180))
        arm4_outline = np.dot(main_outline, m)
        arm4 = self.add_room(outline=np.delete(arm4_outline, 1, 1))

        m = gen_rot_matrix(np.array([0, 1, 0]), 72 * (math.pi / 180))
        arm5_outline = np.dot(main_outline, m)
        arm5 = self.add_room(outline=np.delete(arm5_outline, 1, 1))

        # Connect the maze arms with the hub
        self.connect_rooms(main_arm, hub_room, min_x=-1.18, max_x=1.18)
        self.connect_rooms(arm2, hub_room, min_x=-1.9, max_x=-1.18)
        self.connect_rooms(arm3, hub_room, min_x=-1.9, max_x=0)
        self.connect_rooms(arm4, hub_room, min_x=0, max_x=1.9)
        self.connect_rooms(arm5, hub_room, min_x=1.18, max_x=1.9)

        # Add a box at a random end of the hallway
        self.box = Box(color="red")

        # Place the goal in the left or the right arm
        if self.goal_pos is not None:
            self.place_entity(
                self.box,
                min_x=self.goal_pos[0],
                max_x=self.goal_pos[0],
                min_z=self.goal_pos[2],
                max_z=self.goal_pos[2],
            )
        # else:
        #     if self.np_random.integers(0, 2) == 0:
        #         self.place_entity(self.box, room=left_arm, max_z=left_arm.min_z + 2.5)
        #     else:
        #         self.place_entity(self.box, room=right_arm, min_z=right_arm.max_z - 2.5)

        # Choose a random room and position to spawn at
        self.place_agent(
            dir=self.np_random.uniform(-math.pi / 4, math.pi / 4), room=hub_room
        )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = False

        info["goal_pos"] = self.box.pos

        return obs, reward, termination, truncation, info


class StarMazeLeft(StarMaze):
    def __init__(self, goal_pos=[0.5, 0, 0.5], **kwargs):
        super().__init__(goal_pos=goal_pos, **kwargs)


class StarMazeRight(StarMaze):
    def __init__(self, goal_pos=[3.9, 0, 7.0], **kwargs):
        super().__init__(goal_pos=goal_pos, **kwargs)
