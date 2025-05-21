import time
import numpy as np
import h5py
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "true"
import pygame
import argparse
from environments.planar.planar_envs import PlanarNavigationEnv, PlanarPushEnv

#User inputs and configuration
parser = argparse.ArgumentParser(description="Record demonstrations from planar environments and write as a HDF5 dataset")
parser.add_argument("path_hdf5", help="Path to output HDF5 dataset")
parser.add_argument("env_name", help="Planar environment name", choices=["planar_navigation", "planar_push"])
parser.add_argument("variant_name", help="Environment variant", choices=[
    "empty", "maze_u", "maze_medium", "maze_large", "circle", "circle_maze_u", "circle_maze_medium", "T"])
parser.add_argument("--image_size", default=512, help="Observation and render image size")
params = parser.parse_args()

#Create the environment
if params.env_name == "planar_navigation":
    env = PlanarNavigationEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=int(params.image_size),
        is_goal=False,
        variant_name=params.variant_name)
if params.env_name == "planar_push":
    env = PlanarPushEnv(
        step_freq=20.0,
        render_window=True,
        image_obs=True,
        image_size=int(params.image_size),
        is_goal_effector=False,
        is_goal_object=False,
        variant_name=params.variant_name)

#Logging variables
path_dataset = params.path_hdf5
is_logging = False
index_demo = 0
all_proprio = {}
all_frames = {}

#Check if file about to be written already exists and ask confirmation
if os.path.exists(path_dataset):
    print("Path already exists:", path_dataset)
    key = input("Overwrite file (y/N) ? ")
    if key != "y":
        print("Aborting.")
        exit()
#Print keyboard usage
print("Keyboard: [q,ESC] quit, [r] reset, [l] begin/end log, [c] cancel log")
#Write hdf5 dataset
with h5py.File(path_dataset, "w") as f:
    #Create main group
    group_data = f.create_group("data", track_order=True)
    #Main loop
    obs, info = env.reset()
    while True:
        #Retrieve human action
        action = env.human_actor()
        #Clear data if not logging
        if not is_logging:
            all_proprio = {}
            all_frames = {}
            for key in obs.keys():
                if key.endswith("_image"):
                    all_frames[key] = []
                    all_proprio[key.replace("_image", "_index")] = []
                    all_proprio[key.replace("_image", "_delay")] = []
                else:
                    all_proprio[key] = []
        #Save data into list
        for key,value in obs.items():
            if key.endswith("_image"):
                all_frames[key].append(value)
                all_proprio[key.replace("_image", "_index")].append(
                    np.array([len(all_frames[key])-1], dtype="u8"))
                all_proprio[key.replace("_image", "_delay")].append(
                    np.array([0.0]))
            else:
                all_proprio[key].append(value)
        #Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        #User inputs
        is_saving = False
        if pygame.key.get_pressed()[pygame.K_l]:
            if is_logging:
                is_logging = False
                is_saving = True
                print("Log end")
            else:
                is_logging = True
                print("Log begin")
            time.sleep(0.2)
        if is_logging and pygame.key.get_pressed()[pygame.K_c]:
                is_logging = False
                is_saving = False
                print("Log cancelled")
        if not is_logging and pygame.key.get_pressed()[pygame.K_r]:
            env.reset()
            time.sleep(0.1)
        if pygame.key.get_pressed()[pygame.K_q] or pygame.key.get_pressed()[pygame.K_ESCAPE]:
            break
        #Write data log into dataset
        if is_saving:
            print("Saving log", index_demo, "...")
            name_demo = "demo_{0:05d}".format(index_demo)
            group_demo = group_data.create_group(name_demo)
            group_proprio = group_demo.create_group("proprio")
            group_frames = group_demo.create_group("frames")
            for key,value in all_proprio.items():
                assert len(value[0].shape) == 1
                tmp_type = "f4"
                if key.endswith("_index"):
                    tmp_type = "u8"
                group_proprio.create_dataset(
                    key, 
                    data=np.array(value), 
                    dtype=tmp_type)
            for key,value in all_frames.items():
                assert len(value[0].shape) == 3
                group_frames.create_dataset(
                    key, 
                    data=np.array(value), 
                    dtype="f4", 
                    compression="gzip",
                    chunks=(1,value[0].shape[0],value[0].shape[1],value[0].shape[2]))
            index_demo += 1
            print("Done")
    env.close()

