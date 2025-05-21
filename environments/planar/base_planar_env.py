import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "true"
import copy
import pygame
import pymunk
import pymunk.pygame_util
import gymnasium as gym
import numpy as np
from typing import Union, Dict, Any, Tuple

class BasePlanarEnv(gym.Env):
    """Base class for 2d planar environments using pygame and pymunk"""

    def __init__(self,
            step_freq: float,
            render_window: bool,
            image_obs: bool,
            image_size: int,
        ):
        """Initialize empty base environment
        Args:
            step_freq: Expected frequency at which step is run.
            render_window: If True, human window interface with real time 
                scheduling is used. If False, off screen simulation is used.
            image_obs: If True, render and include pixel image in observation space.
            image_size: Size of square images for both human interface and observation.
        """
        super().__init__()
        self._step_freq = step_freq
        #Initialize viewer
        pygame.display.init()
        #Set render mode and initialize viewer
        assert image_size > 0
        self._image_size = image_size
        if render_window:
            self._window_surface = pygame.display.set_mode((image_size, image_size))
        else:
            self._window_surface = None
        self._canvas_surface = pygame.Surface((image_size, image_size))
        self._image_obs = image_obs
        self._clock = pygame.time.Clock()
        #Physic initialization
        self._space = pymunk.Space(threaded=False)
        self._space.gravity = 0.0, 0.0
        self._handler = self._space.add_collision_handler(0,0)
        #Define main effector body
        self._effector_body = pymunk.Body(pymunk.Body.DYNAMIC)
        self._effector_body.position = 0.0, 0.0
        self._effector_body.velocity = 0.0, 0.0
        self._effector_body.angular_velocity = 0.0
        self._effector_shape = pymunk.Circle(self._effector_body, radius=0.075)
        self._effector_shape.density = 1.0
        self._effector_shape.elasticity = 0.1
        self._space.add(self._effector_body, self._effector_shape)
        self._command = self._effector_body.position
        #Define static bodies
        self._static_shapes = []
        #Add area border barriers
        self._static_shapes.append(pymunk.Segment(
            self._space.static_body, a=(1.0, 1.0), b=(1.0, -1.0), radius=0.01))
        self._static_shapes.append(pymunk.Segment(
            self._space.static_body, a=(1.0, -1.0), b=(-1.0, -1.0), radius=0.01))
        self._static_shapes.append(pymunk.Segment(
            self._space.static_body, a=(-1.0, -1.0), b=(-1.0, 1.0), radius=0.01))
        self._static_shapes.append(pymunk.Segment(
            self._space.static_body, a=(-1.0, 1.0), b=(1.0, 1.0), radius=0.01))
        #Define additional dynamic objects
        self._object_bodies = []
        self._object_shapes = []
        #Define goals
        self._goal_effector_pos = None
        self._goal_object_pos = None
        self._goal_object_angle = None
        #Environment custom initialization
        self._init_env()
        #Configure static shapes
        for shape in self._static_shapes:
            shape.elasticity = 0.1
            self._space.add(shape)
        #Configure dynamic shapes
        for shape in  self._object_shapes:
            shape.density = 0.8
            shape.elasticity = 0.1
            self._space.add(shape)
        #Define action spaces
        self.action_space = gym.spaces.Dict({
            "command_pos": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        })
        #Define observation spaces
        self.observation_space = gym.spaces.Dict({
            "effector_pos": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "effector_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "command_pos": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        })
        if self._image_obs:
            self.observation_space["obs_image"] = \
                gym.spaces.Box(low=0.0, high=1.0, shape=(image_size,image_size,3), dtype=np.float32)
            self.observation_space["goal_image"] = \
                gym.spaces.Box(low=0.0, high=1.0, shape=(image_size,image_size,3), dtype=np.float32)
        for i in range(len(self._object_bodies)):
            self.observation_space["object_"+str(i)+"_pos"] = \
                gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space["object_"+str(i)+"_angle"] = \
                gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
            self.observation_space["object_"+str(i)+"_vel"] = \
                gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
            self.observation_space["object_"+str(i)+"_anglevel"] = \
                gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        if self._goal_effector_pos is not None:
            self.observation_space["goal_effector_pos"] = \
                gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        if self._goal_object_pos is not None and self._goal_object_angle is not None:
            self.observation_space["goal_object_pos"] = \
                gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space["goal_object_angle"] = \
                gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)

    def _init_env(self):
        """Environment custom initialization 
        implemented by downstream classes"""
        pass

    def get_reward(self) -> float:
        """Compute and return the last step reward, 
        default implementation
        """
        return 0.0
        
    def get_info(self,
        ) -> Dict[str, Any]:
        """Compute and return default information dictionary with 
        linear and angular distances between effector/object and goals
        """
        info = {}
        if self._goal_effector_pos is not None:
            info["dist_effector_pos"] = np.linalg.norm(
                np.array(self._goal_effector_pos, dtype="f8") -
                np.array(self._effector_body.position, dtype="f8"))
        if self._goal_object_pos is not None and self._goal_object_angle is not None:
            info["dist_object_pos"] = np.linalg.norm(
                np.array(self._goal_object_pos, dtype="f8") -
                np.array(self._object_bodies[0].position, dtype="f8"))
            info["dist_object_angle"] = np.array(abs(
                (self._goal_object_angle - self._object_bodies[0].angle + np.pi) % (2 * np.pi) - np.pi))
        return info

    def get_obs(self,
        ) -> Dict[str, Any]:
        """Returns environment observations"""
        def normalize_angle(angle):
            return (angle + np.pi) % (2.0 * np.pi) - np.pi
        obs = {
            "effector_pos": np.array(self._effector_body.position, dtype="f4"),
            "effector_vel": np.array(self._effector_body.velocity, dtype="f4"),
            "command_pos": np.array(self._command, dtype="f4"),
        }
        for i in range(len(self._object_bodies)):
            obs["object_"+str(i)+"_pos"] = np.array(self._object_bodies[i].position, dtype="f4")
            obs["object_"+str(i)+"_angle"] = normalize_angle(np.array([self._object_bodies[i].angle], dtype="f4"))
            obs["object_"+str(i)+"_vel"] = np.array(self._object_bodies[i].velocity, dtype="f4")
            obs["object_"+str(i)+"_anglevel"] = \
                np.array([self._object_bodies[i].angular_velocity], dtype="f4")
        if self._goal_effector_pos is not None:
            obs["goal_effector_pos"] = np.array(self._goal_effector_pos, dtype="f4")
        if self._goal_object_pos is not None and self._goal_object_angle is not None:
            obs["goal_object_pos"] = np.array(self._goal_object_pos, dtype="f4")
            obs["goal_object_angle"] = normalize_angle(np.array([self._goal_object_angle], dtype="f4"))
        if self._image_obs:
            self._render_frame(self._canvas_surface, False)
            obs["obs_image"] = np.array(pygame.surfarray.array3d(self._canvas_surface), dtype="f4")/255.0
            self._render_frame_goal(self._canvas_surface)
            obs["goal_image"] = np.array(pygame.surfarray.array3d(self._canvas_surface), dtype="f4")/255.0
        return obs

    def set_obs(self,
            obs: Dict[str, Any],
        ):
        """Assign internal simulation state from given observations"""
        self._effector_body.position = obs["effector_pos"].tolist()
        self._effector_body.velocity = obs["effector_vel"].tolist()
        self._command = pymunk.Vec2d(obs["command_pos"][0], obs["command_pos"][1])
        for i in range(len(self._object_bodies)):
            self._object_bodies[i].position = obs["object_"+str(i)+"_pos"].tolist()
            self._object_bodies[i].angle = obs["object_"+str(i)+"_angle"][0]
            self._object_bodies[i].velocity = obs["object_"+str(i)+"_vel"].tolist()
            self._object_bodies[i].angular_velocity = obs["object_"+str(i)+"_anglevel"][0]
        if self._goal_effector_pos is not None:
            self._goal_effector_pos = obs["goal_effector_pos"].tolist()
        if self._goal_object_pos is not None and self._goal_object_angle is not None:
            self._goal_object_pos = obs["goal_object_pos"].tolist()
            self._goal_object_angle = obs["goal_object_angle"][0]

    def _render_frame(self,
            surface: pygame.Surface,
            verbose: bool,
        ):
        """Draw on given surface current simulation state.
        Args:
            surface: Pygame surface to draw on.
            verbose: If True, also draw command information.
        """
        colors = [
            (0,158,115), #Goal effector
            (0,158,115), #Goal object
            (100,100,100), #Static shape
            (216,27,96), #Dynamic shape
            (30,136,229), #Effector
            (210,157,1), #Command
        ]
        #Clear surface
        surface.fill((255,255,255))
        #Draw goal effector
        if verbose and self._goal_effector_pos is not None:
            self._draw_shape(
                surface,
                self._effector_shape, 
                pos=self._goal_effector_pos,
                color=colors[0], 
                width=5)
        #Draw goal object
        if verbose and self._goal_object_pos is not None and self._goal_object_angle is not None:
            for shape in self._object_bodies[0].shapes:
                self._draw_shape(
                    surface,
                    shape, 
                    pos=self._goal_object_pos,
                    angle=self._goal_object_angle,
                    color=colors[1], 
                    width=5)
        #Draw static shapes
        for shape in self._static_shapes:
            self._draw_shape(
                surface,
                shape, 
                color=colors[2],
                width=0)
        #Draw dynamic object shapes
        for shape in self._object_shapes:
            self._draw_shape(
                surface,
                shape, 
                pos=shape.body.position,
                angle=shape.body.angle,
                color=colors[3], 
                width=0)
        #Draw effector shape
        self._draw_shape(
            surface,
            self._effector_shape, 
            pos=self._effector_body.position,
            color=colors[4], 
            width=0)
        #Draw command arrow
        if verbose:
            self._draw_arrow(
                surface,
                self._effector_body.position,
                self._command,
                radius=0.01,
                color=colors[5])

    def _render_frame_goal(self,
            surface: pygame.Surface,
        ):
        """Draw on given surface goal simulation state.
        Args:
            surface: Pygame surface to draw on.
        """
        colors = [
            (0,158,115), #Goal effector
            (0,158,115), #Goal object
            (100,100,100), #Static shape
            (216,27,96), #Dynamic shape
            (30,136,229), #Effector
            (210,157,1), #Command
        ]
        #Clear surface
        surface.fill((255,255,255))
        #Draw static shapes
        for shape in self._static_shapes:
            self._draw_shape(
                surface,
                shape, 
                color=colors[2],
                width=0)
        #Draw goal effector
        if self._goal_effector_pos is not None:
            self._draw_shape(
                surface,
                self._effector_shape, 
                pos=self._goal_effector_pos,
                color=colors[4], 
                width=0)
        #Draw goal object
        if self._goal_object_pos is not None and self._goal_object_angle is not None:
            for shape in self._object_shapes:
                self._draw_shape(
                    surface,
                    shape, 
                    pos=self._goal_object_pos,
                    angle=self._goal_object_angle,
                    color=colors[3], 
                    width=0)
    
    def render(self,
        ) -> np.ndarray:
        """Compute and return the color frame of current environment state 
        with verbose information (image_size,image_size,3) uint8.
        """
        self._render_frame(self._canvas_surface, True)
        array_rgb = pygame.surfarray.array3d(self._canvas_surface)
        return array_rgb

    def _is_state_valid(self,
        ) -> bool:
        """Returns true is the current internal simulation state 
        is valid, i.e. without collision
        """
        is_collision = False
        def on_collision(arbiter, space, data):
            nonlocal is_collision
            is_collision = True
            return True
        self._handler.begin = on_collision
        self._handler.pre_solve = on_collision
        self._handler.post_solve = on_collision
        self._space.step(0.01)
        self._space.step(0.01)
        return not is_collision
        
    def reset(self, 
            seed: Union[None,int] = None, 
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Default random reset position of all objects 
        and effector and goals within area.
        Args:
            seed: Seed for numpy internal random engine to make reset deterministic.
        Returns:
            Observation dictionary after reset.
            Information dictionary.
        """
        if seed is not None:
            np.random.seed(seed)

        #Sample feasible state for goal
        while True:
            self._effector_body.position = np.random.uniform(low=-0.95, high=0.95, size=(2,)).tolist()
            self._effector_body.velocity = 0.0, 0.0
            self._effector_body.angle = 0.0
            self._effector_body.angular_velocity = 0.0
            for i in range(len(self._object_bodies)):
                self._object_bodies[i].position = np.random.uniform(low=-0.95, high=0.95, size=(2,)).tolist()
                self._object_bodies[i].velocity = 0.0, 0.0
                self._object_bodies[i].angle = np.random.uniform(low=-np.pi, high=np.pi)
                self._object_bodies[i].angular_velocity = 0.0
            if self._is_state_valid():
                break
        #Assign state as goal
        if self._goal_effector_pos is not None:
            self._goal_effector_pos = self._effector_body.position
        if self._goal_object_pos is not None and self._goal_object_angle is not None:
            self._goal_object_pos = self._object_bodies[0].position
            self._goal_object_angle = self._object_bodies[0].angle
        #Re-sample feasible initial state
        while True:
            self._effector_body.position = np.random.uniform(low=-0.95, high=0.95, size=(2,)).tolist()
            self._effector_body.velocity = 0.0, 0.0
            self._effector_body.angle = 0.0
            self._effector_body.angular_velocity = 0.0
            for i in range(len(self._object_bodies)):
                self._object_bodies[i].position = np.random.uniform(low=-0.95, high=0.95, size=(2,)).tolist()
                self._object_bodies[i].velocity = 0.0, 0.0
                self._object_bodies[i].angle = np.random.uniform(low=-np.pi, high=np.pi)
                self._object_bodies[i].angular_velocity = 0.0
            if self._is_state_valid():
                break
        self._command = self._effector_body.position
        return self.get_obs(), self.get_info()

    def step(self,
            action: Dict[str, np.ndarray],
        ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Update the environment with given action 
        and return new observations.
        Args:
            action: Dictionary of actions.
        Returns:
            obs: Dictionary of new observations.
            reward: Reward associated to action state transition.
            terminated: If True, the agent has reach a terminal state.
            truncated: If True, the episode is prematurely 
                stopped because of time limit.
            info: Dictionary of additional information.
        """
        #Retrieve effector command as desired position
        assert "command_pos" in action and action["command_pos"].shape == (2,)
        self._command = pymunk.Vec2d(action["command_pos"][0], action["command_pos"][1])
        #Run physics
        assert (60/self._step_freq)%1.0 < 1e-3, "step_freq not supported: " + str(self._step_freq)
        for k in range(60//int(self._step_freq)):
            #Compute force action from saturated PD command position and velocity
            delta_pos = self._command - self._effector_body.position
            delta_force = 50.0*delta_pos
            if delta_force.length > 5.0:
                delta_force = 5.0*delta_force.normalized()
            delta_force -= 1.5*self._effector_body.velocity
            #Apply effort to effector
            self._effector_body.apply_force_at_local_point(delta_force)
            #Step physic engine
            self._space.step(1.0/200.0)
        #Rendering on window
        if self._window_surface:
            for event in pygame.event.get():
                pass
            self._render_frame(self._window_surface, True)
            pygame.display.flip()
            self._clock.tick(self._step_freq)
        #Generate and return outputs
        obs = self.get_obs()
        reward = self.get_reward()
        terminated = False
        truncated = False
        info = self.get_info()
        return obs, reward, terminated, truncated, info

    def human_actor(self,
        ) -> Dict[str, np.ndarray]:
        """Use Pygame and mouse interface to return 
        human teleoperated action. 
        Use mouse left click to provide commanded position.
        Returns:
            Dict of valid actions, i.e. effector 
                desired position in world frame.
        """
        assert self._window_surface is not None, \
            "Window rendering is not enabled"
        if pygame.mouse.get_pressed()[0]:
            pos_in_screen = np.array(pygame.mouse.get_pos())
            pos_in_world = (pos_in_screen-self._image_size/2)/self._image_size*2
            return {"command_pos": pos_in_world}
        else:
            return {"command_pos": np.array(self._effector_body.position)}

    def close(self,
        ):
        """Terminate environment and 
        deallocate underlying Pygame library
        """
        pygame.quit()

    def _add_object_body(self,
        ) -> pymunk.Body:
        """Create and configure a new dynamic body.
        Returns:
            Newly created Pymunk body.
        """
        #Create body
        body = pymunk.Body(pymunk.Body.DYNAMIC)
        body.position = 0.0, 0.0
        body.angle = 0.0
        body.velocity = 0.0, 0.0
        body.angular_velocity = 0.0
        self._space.add(body)
        #Define top down linear friction
        pivot = pymunk.PivotJoint(self._space.static_body, body, (0,0), (0,0))
        pivot.max_bias = 0
        pivot.max_force = 2.0
        self._space.add(pivot)
        #Define top down angular friction
        gear = pymunk.GearJoint(self._space.static_body, body, 0.0, 1.0)
        gear.max_bias = 0
        gear.max_force = 0.3
        self._space.add(gear)
        return body

    def _create_box_shape(self,
            body: pymunk.Body,
            sizes: Tuple[float,float],
            pos: Tuple[float,float] = (0.0,0.0),
            angle: float = 0.0,
        ) -> pymunk.Shape:
        """Create a Pymunk box shape as a Polygon shape 
        associated to given Pymunk body.
        Args:
            body:  Pymunk body to attach the shape to.
            sizes: Full X and Y sizes of the box.
            pos: Shape position offset in body frame.
            ang: Shape angle offset in body frame.
        Returns:
            Pymunk shape.
        """
        vertices = [
            (-sizes[0]/2, -sizes[1]/2),
            (sizes[0]/2, -sizes[1]/2),
            (sizes[0]/2, sizes[1]/2),
            (-sizes[0]/2, sizes[1]/2)]
        transform = pymunk.Transform.translation(pos[0], pos[1]).rotated(angle)
        transformed_vertices = [transform@v for v in vertices]
        return pymunk.Poly(body, transformed_vertices)

    def _draw_shape(self, 
            surface: pygame.Surface,
            shape: pymunk.Shape,
            pos: Tuple[float,float] = (0.0,0.0),
            angle: float = 0.0,
            color: Tuple[int,int,int] = (0,255,0),
            width: int = 0):
        """Draw given Pymunk shape with Pygame.
        Args:
            surface: Pygame surface to draw on.
            shape: Pymunk surface to Draw. 
                Circle, Segment, and Poly are implemented.
            pos: Shape position in world frame.
            angle: Shape angle in world frame.
            color: Shape RGB color.
            width: Shape border drawing width.
                Shape is filled if 0.
        """
        cam_scale = self._image_size/2
        cam_offset = self._image_size/2, self._image_size/2
        if isinstance(shape, pymunk.shapes.Circle):
            pos = cam_scale*(pos + shape.offset)+cam_offset
            pygame.draw.circle(
                surface,
                color=color, 
                center=(round(pos[0]), round(pos[1])), 
                radius=round(cam_scale*shape.radius),
                width=width)
        elif isinstance(shape, pymunk.shapes.Segment):
            p1 = cam_scale*(pos + shape.a)+cam_offset
            p2 = cam_scale*(pos + shape.b)+cam_offset
            radius = cam_scale*shape.radius
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
               return
            scale = radius/(orthog[0]*orthog[0]+orthog[1]*orthog[1])**0.5
            orthog[0] = round(orthog[0]*scale)
            orthog[1] = round(orthog[1]*scale)
            points = [
                (p1[0]-orthog[0], p1[1]-orthog[1]),
                (p1[0]+orthog[0], p1[1]+orthog[1]),
                (p2[0]+orthog[0], p2[1]+orthog[1]),
                (p2[0]-orthog[0], p2[1]-orthog[1]),
            ]
            pygame.draw.polygon(
                surface,
                color=color, 
                points=points,
                width=width)
            pygame.draw.circle(
                surface,
                color=color,
                center=(round(p1[0]), round(p1[1])),
                radius=round(radius),
                width=width)
            pygame.draw.circle(
                surface,
                color=color,
                center=(round(p2[0]), round(p2[1])),
                radius=round(radius),
                width=width)
        elif isinstance(shape, pymunk.shapes.Poly):
            points = shape.get_vertices()
            points = [v.rotated(angle)+pos for v in points]
            points = [cam_scale*v+cam_offset for v in points]
            points = [(round(v[0]), round(v[1])) for v in points]
            pygame.draw.polygon(
                surface,
                color=color, 
                points=points,
                width=width)
        else:
            raise NotImplementedError("Shape drawing not implemented")

    def _draw_arrow(self, 
            surface: pygame.Surface,
            start_pos: Tuple[float,float],
            end_pos: Tuple[float,float],
            radius: float,
            color: Tuple[int,int,int] = (0,255,0),
        ):
        """Draw a oriented line.
        Args:
            surface: Pygame surface to draw on.
            start_pos: Arrow starting position in world frame.
            end_pos: Arrow end position in world frame.
            radius: Arrow thickness in  world frame.
            color: Arrow RGB color.
        """
        cam_scale = self._image_size/2
        cam_offset = self._image_size/2, self._image_size/2
        p1 = start_pos*cam_scale + cam_offset
        p2 = end_pos*cam_scale + cam_offset
        pygame.draw.line(
            surface,
            start_pos=(round(p1[0]), round(p1[1])), 
            end_pos=(round(p2[0]), round(p2[1])),
            color=color, 
            width=round(radius*cam_scale))
        pygame.draw.circle(
            surface,
            color=color, 
            center=(round(p2[0]), round(p2[1])), 
            radius=round(2.0*radius*cam_scale),
            width=0)

