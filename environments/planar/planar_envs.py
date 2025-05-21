import pymunk
import numpy as np
from environments.planar.base_planar_env import BasePlanarEnv

from typing import Union, Dict, Any, Tuple

class PlanarNavigationEnv(BasePlanarEnv):
    """Planar 2d environment where a position-controlled 
    effector must navigate to a goal, with or without obstacles
    """

    def __init__(self,
            step_freq: float = 20.0,
            render_window: bool = False,
            image_obs: bool = True,
            image_size: int = 128,
            is_goal: bool = True,
            variant_name: str = "empty",
        ):
        """Initialize environment.
        Args:
            step_freq: Expected frequency at which step is run.
            render_window: If True, human window interface with real time 
                scheduling is used. If False, off screen simulation is used.
            image_obs: If True, render and include pixel image in observation space.
            image_size: Size of square images for both human interface and observation.
            is_goal: if False, effector goal is not included in the environment.
            variant_name: Textual name for environment obstacle setup.
        """
        self._is_goal = is_goal
        self._variant_name = variant_name
        super().__init__(
            step_freq=step_freq,
            render_window=render_window,
            image_obs=image_obs,
            image_size=image_size)

    def _init_env(self):
        """Environment custom initialization"""
        #Goal
        if self._is_goal:
            self._goal_effector_pos = (0.0, 0.0)
        #Static bodies
        if self._variant_name == "empty":
            pass
        elif self._variant_name == "centered":
            self._static_shapes.append(pymunk.Circle(
                self._space.static_body, radius=0.5, offset=(0.0,0.0)))
        elif self._variant_name.startswith("maze_"):
            if self._variant_name == "maze_u":
                tmp_maze = np.array([
                    [0,0,0,0],
                    [1,1,1,0],
                    [0,0,0,0],
                    [0,0,0,0],
                ])
            elif self._variant_name == "maze_medium":
                tmp_maze = np.array([
                    [0,0,1,1,0,0],
                    [0,0,1,0,0,0],
                    [1,0,0,0,1,1],
                    [0,0,1,0,0,0],
                    [0,1,0,0,1,0],
                    [0,0,0,1,0,0],
                ])
            elif self._variant_name == "maze_large":
                tmp_maze = np.array([
                    [1,1,1,1,1,1,1,1,1,1],
                    [0,0,0,0,1,0,0,0,0,0],
                    [0,1,1,0,1,0,1,0,1,0],
                    [0,0,0,0,0,0,1,0,0,0],
                    [0,1,1,1,1,0,1,1,1,0],
                    [0,0,1,0,1,0,0,0,0,0],
                    [1,0,1,0,1,0,1,0,1,1],
                    [0,0,1,0,0,0,1,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1],
                ])
            else:
                raise IOError("Unknown variant:" + self._variant_name)
            tmp_size = tmp_maze.shape[0]
            for i in range(tmp_size):
                for j in range(tmp_size):
                    if tmp_maze[i,j] > 0.0:
                        self._static_shapes.append(self._create_box_shape(
                            self._space.static_body, 
                            sizes=(2.0/tmp_size,2.0/tmp_size), 
                            pos=(j*2.0/tmp_size-1.0+1.0/tmp_size,i*2.0/tmp_size-1.0+1.0/tmp_size), 
                            angle=0.0))
        else:
            raise IOError("Unknown variant:" + self._variant_name)

    def get_reward(self) -> float:
        """Compute sparse binary reward"""
        info = self.get_info()
        if self._is_goal:
            return float(info["dist_effector_pos"] > 0.04)
        else:
            return 0.0
    
class PlanarPushEnv(BasePlanarEnv):
    """Planar 2d environment where a position-controlled 
    effector must push an object toward a specified pose
    """
    def __init__(self,
            step_freq: float = 20.0,
            render_window: bool = False,
            image_obs: bool = True,
            image_size: int = 128,
            is_goal_effector: bool = True,
            is_goal_object: bool = True,
            fixed_goal: bool = True,
            variant_name: str = "circle",
        ):
        """Initialize environment.
        Args:
            step_freq: Expected frequency at which step is run.
            render_window: If True, human window interface with real time 
                scheduling is used. If False, off screen simulation is used.
            image_obs: If True, render and include pixel image in observation space.
            image_size: Size of square images for both human interface and observation.
            is_goal_effector: if False, effector goal is not included in the environment.
            is_goal_object: if False, object goal is not included in the environment.
            fixed_goal: If True, the goal is not randomized and set at the are center.
            variant_name: Textual name for environment object shape.
        """
        self._is_goal_effector = is_goal_effector
        self._is_goal_object = is_goal_object
        self._fixed_goal = fixed_goal
        self._variant_name = variant_name
        super().__init__(
            step_freq=step_freq,
            render_window=render_window,
            image_obs=image_obs,
            image_size=image_size)
    
    def _init_env(self):
        """Environment custom initialization"""
        #Goal
        if self._is_goal_effector:
            self._goal_effector_pos = (0.0, 0.5)
        if self._is_goal_object:
            self._goal_object_pos = (0.0, 0.0)
            self._goal_object_angle = 0.0
        #Dynamic body object
        tmp_maze = None
        if self._variant_name == "T":
            tmp_body = self._add_object_body()
            self._object_bodies.append(tmp_body)
            tmp_body.position = 0.0, 0.0
            self._object_shapes.append(self._create_box_shape(
                tmp_body, sizes=(0.45,0.15), pos=(0.0,0.0), angle=0.0))
            self._object_shapes.append(self._create_box_shape(
                tmp_body, sizes=(0.15,0.45), pos=(0.0,0.15), angle=0.0))
        elif self._variant_name == "circle":
            tmp_body = self._add_object_body()
            self._object_bodies.append(tmp_body)
            tmp_body.position = 0.0, 0.0
            self._object_shapes.append(pymunk.Circle(tmp_body, radius=0.2))
        elif self._variant_name == "circle_maze_u":
            tmp_body = self._add_object_body()
            self._object_bodies.append(tmp_body)
            tmp_body.position = 0.0, 0.0
            self._object_shapes.append(pymunk.Circle(tmp_body, radius=0.2))
            tmp_maze = np.array([
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
            ])
        elif self._variant_name == "circle_maze_medium":
            tmp_body = self._add_object_body()
            self._object_bodies.append(tmp_body)
            tmp_body.position = 0.0, 0.0
            self._object_shapes.append(pymunk.Circle(tmp_body, radius=0.2))
            tmp_maze = np.array([
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0],
                [1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0],
            ])
        else:
            raise IOError("Unknown variant:" + self._variant_name)
        if tmp_maze is not None:
            tmp_size = tmp_maze.shape[0]
            for i in range(tmp_size):
                for j in range(tmp_size):
                    if tmp_maze[i,j] > 0.0:
                        self._static_shapes.append(self._create_box_shape(
                            self._space.static_body, 
                            sizes=(2.0/tmp_size,2.0/tmp_size), 
                            pos=(j*2.0/tmp_size-1.0+1.0/tmp_size,i*2.0/tmp_size-1.0+1.0/tmp_size), 
                            angle=0.0))
    
    def reset(self, 
            seed: Union[None,int] = None, 
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Override base reset with fixed goal"""
        obs, info = super().reset(seed)
        if self._fixed_goal and self._is_goal_effector:
            self._goal_effector_pos = (0.0, 0.5)
        if self._fixed_goal and self._is_goal_object:
            self._goal_object_pos = 0.0, 0.0
            self._goal_object_angle = 0.0
        return obs, info
    
    def get_reward(self) -> float:
        """Compute sparse binary reward"""
        info = self.get_info()
        if self._variant_name.startswith("circle") and self._is_goal_object and not self._is_goal_effector:
            return float(info["dist_object_pos"] > 0.04)
        elif self._variant_name.startswith("circle") and self._is_goal_object and self._is_goal_effector:
            return float(info["dist_object_pos"] > 0.04 or info["dist_effector_pos"] > 0.04)
        else:
            return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Specialize info for circle"""
        info = super().get_info()
        if self._variant_name.startswith("circle") and self._is_goal_object:
            del info["dist_object_angle"]
        return info

class PlanarTestEnv(BasePlanarEnv):
    """Planar 2d environment for testing BasePlanarEnv"""
    
    def __init__(self,
            step_freq: float = 20.0,
            render_window: bool = False,
            image_obs: bool = True,
            image_size: int = 128,
        ):
        """Initialize environment.
        Args:
            step_freq: Expected frequency at which step is run.
            render_window: If True, human window interface with real time 
                scheduling is used. If False, off screen simulation is used.
            image_obs: If True, render and include pixel image in observation space.
            image_size: Size of square images for both human interface and observation.
        """
        super().__init__(
            step_freq=step_freq,
            render_window=render_window,
            image_obs=image_obs,
            image_size=image_size)
    
    def _init_env(self):
        """Environment custom initialization"""
        #Static obstacle
        self._static_shapes.append(pymunk.Circle(
            self._space.static_body, radius=0.6, offset=(-0.5,-0.5)))
        #Dynamic objects
        body1 = self._add_object_body()
        self._object_bodies.append(body1)
        body1.position = -0.4, -0.1
        self._object_shapes.append(self._create_box_shape(body1, sizes=(0.3,0.1), pos=(0.0,0.0), angle=0.0))
        self._object_shapes.append(self._create_box_shape(body1, sizes=(0.1,0.3), pos=(0.0,0.1), angle=0.0))
        body2 = self._add_object_body()
        self._object_bodies.append(body2)
        body2.position = -0.1, -0.5
        self._object_shapes.append(pymunk.Circle(body2, radius=0.15))
        #Goals 
        self._goal_effector_pos = (-0.5, 0.5)
        self._goal_object_pos = (0.5, -0.5)
        self._goal_object_angle = 0.8

