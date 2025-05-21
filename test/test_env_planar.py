import time
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "true"
import pygame
import numpy as np
import matplotlib.pyplot as plt

from environments.planar.planar_envs import PlanarTestEnv, PlanarNavigationEnv, PlanarPushEnv

envs = [
    PlanarTestEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=512),
    PlanarNavigationEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=512,
        is_goal=True,
        variant_name="empty"),
    PlanarNavigationEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=512,
        is_goal=True,
        variant_name="centered"),
    PlanarNavigationEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=512,
        is_goal=True,
        variant_name="maze_u"),
    PlanarNavigationEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=512,
        is_goal=True,
        variant_name="maze_medium"),
    PlanarNavigationEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=512,
        is_goal=True,
        variant_name="maze_large"),
    PlanarPushEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=512,
        is_goal_effector=True,
        is_goal_object=True,
        fixed_goal=False,
        variant_name="circle"),
    PlanarPushEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=512,
        is_goal_effector=True,
        is_goal_object=True,
        fixed_goal=False,
        variant_name="circle_maze_u"),
    PlanarPushEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=512,
        is_goal_effector=True,
        is_goal_object=True,
        fixed_goal=False,
        variant_name="circle_maze_medium"),
    PlanarPushEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=512,
        is_goal_effector=True,
        is_goal_object=True,
        fixed_goal=True,
        variant_name="T"),
]

for env in envs:
    print("============", type(env))
    obs, info = env.reset(seed=42)
    print("Observation:")
    for key,val in obs.items():
        if key == "obs_image" or key == "goal_image":
            print("--", key, type(val), val.shape, val.dtype)
        else:
            print("--", key, type(val), val.shape, val.dtype, val)
    print("Information:")
    for key,val in info.items():
        print("--", key, type(val), val.shape, val)
    while True:
        action = env.human_actor()
        obs, reward, terminated, truncated, info = env.step(action)
        print("Reward:", reward, "Info:", info)
        #array_rgb = env.render()
        #env.set_obs(observation)
        if terminated or truncated:
            observation, info = env.reset()
        #User inputs
        if pygame.key.get_pressed()[pygame.K_p]:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(obs["obs_image"].transpose(1,0,2))
            axs[1].imshow(obs["goal_image"].transpose(1,0,2))
            if "object_0_pos" in obs:
                tmp_pos = 0.5*(obs["object_0_pos"]+1.0)*512
                axs[0].scatter(tmp_pos[0], tmp_pos[1], color="black")
            if "effector_pos" in obs:
                tmp_pos = 0.5*(obs["effector_pos"]+1.0)*512
                axs[0].scatter(tmp_pos[0], tmp_pos[1], color="yellow")
            fig.tight_layout()
            plt.show()
        if pygame.key.get_pressed()[pygame.K_r]:
            time.sleep(0.1)
            env.reset()
        if pygame.key.get_pressed()[pygame.K_q] or pygame.key.get_pressed()[pygame.K_ESCAPE]:
            time.sleep(0.1)
            break
env.close()

