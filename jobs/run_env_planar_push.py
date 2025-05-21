import os
import time
import copy
import random
import numpy as np
import torch
import h5py
import platform
from collections import OrderedDict
import matplotlib
import matplotlib.animation as anm
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchaug.transforms import (
    RandomColorJitter,
    RandomGaussianBlur,
    RandomErasing,
    RandomAffine,
    CenterCrop,
    Resize,
    RandomVerticalFlip,
    RandomHorizontalFlip,
    SequentialTransform,
)

from utils import job
from gcrl.agent import GCRLAgentBase, GCRLAgentBC, GCRLAgent0, GCRLAgent1, GCRLAgent2, GCRLAgent5, GCRLAgent6
from gcrl.dataset import GCTrajsDataset
from environments.planar.planar_envs import PlanarPushEnv

from typing import List, Tuple, Dict, Union

#Job parameters
params = {
    #Job mode
    "mode": "",
    #Dataset processing
    #"path_hdf5_demo": "data_static/20250415_demo_planar_push_circle_maze_medium_reach_goal.hdf5",
    #"path_hdf5_demo": "data_static/20250415_demo_planar_push_circle_maze_medium_play_full.hdf5",
    #"path_hdf5_demo": "data_static/20250415_demo_planar_push_circle_maze_medium_play_stitch.hdf5",
    #"path_hdf5_eval": "data_static/20250415_eval_planar_push_circle_maze_medium.hdf5",
    #"path_npz_demo": "data_static/dataset_demo_reach_goal.npz",
    #"path_npz_demo": "data_static/dataset_demo_play_full.npz",
    "path_npz_demo": "data_static/dataset_demo_play_stitch.npz",
    "path_npz_eval": "data_static/dataset_eval.npz",
    "use_images": False,
    #Loading previous agents
    "is_load": False,
    #Trajectories
    "trajs_obs_len": 3,
    "trajs_obs_stride": 10,
    "trajs_act_len": 8,
    "trajs_act_stride": 1,
    "max_goal_dist": 600,
    #Device
    "use_cuda": False,
    #Agents
    "randomize_config": False,
    "encoder_proprio_name": "identity",
    "encoder_proprio_size": 2,
    #Training
    "epoch": 200000,
    "ema_cutoff": 1000,
    "size_batch": 128,
    "save_model_every": 10000000,
    "plot_model_every": 10000000,
    "eval_model_every": 40000,
    #Inference
    "inference_steps": 600,
    "inference_horizon": 7,
    "num_eval_episode": 6,
    "seed": 0,
}
params, logger = job.init(params, "Run planar push environment on Extremum Flow Matching Agents")
print("Parameters:")
for key,value in params.items():
    print("    {0:40} {1}".format(key, value))
print("Logger output path:", logger.get_output_path())
print("Hostname:", platform.node())

#Default plot parameters
plt.rcParams["figure.figsize"] = [25.0, 14.0]

#Device configuration
if params["use_cuda"]:
    if torch.cuda.is_available():
        print("Available GPUS:", torch.cuda.device_count())
        print(
            "Using CUDA device: id=", torch.cuda.current_device(), 
            "name=", torch.cuda.get_device_name(torch.cuda.current_device()))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise IOError("CUDA not available")
else:
    device = torch.device("cpu")
print("Using device:", device)

#Initialize agents configuration
config = GCRLAgentBase.get_config()
config["epoch"] = params["epoch"]
config["ema_cutoff"] = params["ema_cutoff"]
config["trajs_obs_len"] = params["trajs_obs_len"]
config["trajs_act_len"] = params["trajs_act_len"]
config["trajs_obs_stride"] = params["trajs_obs_stride"]
config["trajs_act_stride"] = params["trajs_act_stride"]
config["max_goal_dist"] = params["max_goal_dist"]
params.update(config)
#Randompize configuration
if params["randomize_config"]:
    params["encoder_proprio_name"] = random.choice(["identity", "mlp"])
    params["encoder_proprio_size"] = random.choice([2, 16, 32, 64])
    params["model_encoder_mlp_hidden"] = random.choice([[32], [64], [128]])
    params["model_flow_unet_hidden"] = random.choice([
        [32*1, 64*1, 128*1],
        [32*2, 64*2, 128*2],
        [32*3, 64*3, 128*3],
        [32*4, 64*4, 128*4],
        ])
    params["model_flow_mlp_hidden"] = random.choice([
        [512, 512, 512],
        [1024, 1024, 1024],
        [2048, 2048, 2048],
        ])
    params["weight_decay"] = random.choice([1e-3, 1e-6, 1e-8])
    params["learning_rate"] = random.choice([5e-4, 2e-4, 1e-4, 5e-5])
    params["critic_ratio_training"] = random.choice([0.1, 0.2, 0.5, 0.7])
    params["trajs_obs_len"] = random.choice([2, 4, 8])
    params["trajs_act_len"] = random.choice([8, 16])
    params["trajs_obs_stride"] = random.choice([1, 2, 4, 8, 16])
    params["trajs_act_stride"] = 1
    params["max_goal_dist"] = random.choice([50, 75, 100, 150, 200, 500])
    params["ema_cutoff"] = random.choice([500, 1000, 2000, 5000])
    if params["trajs_obs_len"] < 8:
        params["model_planner_name"] = "mlp"
    if params["encoder_proprio_name"] == "identity":
        params["encoder_proprio_size"] = 2
#Verbose configuration
print("Agents config:")
for key,value in params.items():
    print("    {0:40} {1}".format(key, value))

@torch.no_grad()
def encode_frame(
        frame_raw: Union[np.ndarray, torch.Tensor],
        device: torch.device = None,
    ) -> torch.Tensor:
    if isinstance(frame_raw, np.ndarray):
        tensor_frame = torch.tensor(frame_raw, dtype=torch.uint8, device=device)
    else:
        tensor_frame = frame_raw.to(device)
    torch._assert(torch.is_tensor(tensor_frame), "")
    #Tracable center crop
    tmp_size1 = tensor_frame.size(0)//2
    tmp_size2 = tensor_frame.size(1)//2
    tmp_size3 = min(tmp_size1, tmp_size2)
    tensor_frame = tensor_frame[tmp_size1-tmp_size3:tmp_size1+tmp_size3,tmp_size2-tmp_size3:tmp_size2+tmp_size3,:]
    #Other transformations
    tensor_frame = tensor_frame.permute(2,0,1)
    transforms = SequentialTransform([
        Resize(128),
        RandomVerticalFlip(p=1.0),
        RandomHorizontalFlip(p=1.0),
    ])
    tensor_frame = transforms(tensor_frame)
    tensor_frame = tensor_frame.permute(1,2,0)
    return tensor_frame

def do_process_dataset(
    path_hdf5: str, 
    path_dataset: str,
    ):
    #Extract demonstrations from dataset
    data_demos = OrderedDict()
    data_demos["eff_pos"] = []
    data_demos["obj_pos"] = []
    data_demos["cmd_pos"] = []
    data_demos["images"] = []
    data_demos["terminal"] = []
    with h5py.File(path_hdf5, "r") as file_dataset:
        for name_demo in file_dataset["data"]:
            array_effector = file_dataset["data"][name_demo]["proprio/effector_pos"][()]
            array_object = file_dataset["data"][name_demo]["proprio/object_0_pos"][()]
            array_cmd = file_dataset["data"][name_demo]["proprio/command_pos"][()]
            array_terminal = np.zeros(array_cmd.shape[0], dtype=np.uint8)
            array_terminal[-1] = 1
            print(name_demo, array_effector.shape, array_object.shape, array_cmd.shape)
            list_images = []
            for i in range(array_cmd.shape[0]):
                frame_raw = 255.0*file_dataset["data"][name_demo]["frames/obs_image"][i]
                frame_processed = encode_frame(frame_raw).numpy()
                list_images.append(frame_processed)
            data_demos["eff_pos"].append(array_effector)
            data_demos["obj_pos"].append(array_object)
            data_demos["cmd_pos"].append(array_cmd)
            data_demos["images"].append(np.array(list_images))
            data_demos["terminal"].append(array_terminal)
    for label in data_demos.keys():
        data_demos[label] = np.concatenate(data_demos[label], axis=0)
    print("Dataset Train:")
    for n,v in data_demos.items():
        print("    ", n, v.shape, v.dtype)
    #Conversion to npz
    print("Writing dataset:", path_dataset)
    np.savez(path_dataset, **data_demos)

#Extract demonstrations from hdf5 dataset and generate processed npz dataset
if params["mode"] == "process_dataset":
    do_process_dataset(params["path_hdf5_demo"], params["path_npz_demo"])
    do_process_dataset(params["path_hdf5_eval"], params["path_npz_eval"])
    exit()

#Load processed dataset 
print("Load dataset train:", params["path_npz_demo"])
data_demos = np.load(params["path_npz_demo"])
print("Dataset Train:")
for n,v in data_demos.items():
    print("    ", n, v.shape, v.dtype)
print("Load dataset eval:", params["path_npz_eval"])
data_evals = np.load(params["path_npz_eval"])
print("Dataset Eval:")
for n,v in data_evals.items():
    print("    ", n, v.shape, v.dtype)

#Create Goal Conditioned Dataset for trajectories
dataset_train = GCTrajsDataset(
    dict_obs={
        #"images": data_demos["images"],
        "cmd_pos": data_demos["cmd_pos"],
        "eff_pos": data_demos["eff_pos"],
        "obj_pos": data_demos["obj_pos"],
    },
    array_act=data_demos["cmd_pos"],
    array_terminal=data_demos["terminal"],
    trajs_obs_len=params["trajs_obs_len"],
    trajs_act_len=params["trajs_act_len"],
    trajs_obs_stride=params["trajs_obs_stride"],
    trajs_act_stride=params["trajs_act_stride"],
    max_dist_goal=params["max_goal_dist"],
    is_act_vel=False,
    device=device,
)
dataset_eval = GCTrajsDataset(
    dict_obs={
        #"images": data_evals["images"],
        "cmd_pos": data_evals["cmd_pos"],
        "eff_pos": data_evals["eff_pos"],
        "obj_pos": data_evals["obj_pos"],
    },
    array_act=data_evals["cmd_pos"],
    array_terminal=data_evals["terminal"],
    trajs_obs_len=params["trajs_obs_len"],
    trajs_act_len=params["trajs_act_len"],
    trajs_obs_stride=params["trajs_obs_stride"],
    trajs_act_stride=params["trajs_act_stride"],
    max_dist_goal=params["max_goal_dist"],
    is_act_vel=False,
    device=device,
)

#Verbose
tmp_dist_train = []
for i in range(dataset_train.count_episodes()):
    tmp_obs, tmp_act = dataset_train.get_episode(i)
    tmp_dist_train.append(tmp_act.size(0))
print("Dataset GCTrajs Train:")
print("    Count episodes:", dataset_train.count_episodes())
print("    Length episodes:", 
    "mean", np.array(tmp_dist_train).mean(), 
    "min", np.array(tmp_dist_train).min(), 
    "max", np.array(tmp_dist_train).max())
print("Dataset GCTrajs Eval:")
print("    Count episodes:", dataset_eval.count_episodes())
size_act = tmp_act.size(1)
print("size_act", size_act)

#Plot dataset episodes
if params["mode"] == "plots":
    #Plot state
    fig, axs = plt.subplots(1, 2)
    for i in range(dataset_train.count_episodes()):
        tmp_episode_obs, tmp_episode_act = dataset_train.get_episode(i)
        axs[0].plot(tmp_episode_obs["eff_pos"][:,0], tmp_episode_obs["eff_pos"][:,1])
        axs[1].plot(tmp_episode_obs["obj_pos"][:,0], tmp_episode_obs["obj_pos"][:,1])
    axs[0].axis("equal")
    axs[1].axis("equal")
    axs[0].grid()
    axs[1].grid()
    fig.tight_layout()
    plt.show()
    #Plot action episodes
    fig, axs = plt.subplots(size_act, 1)
    for i in range(dataset_train.count_episodes()):
        tmp_episode_obs, tmp_episode_act = dataset_train.get_episode(i)
        for j in range(size_act):
            axs[j].plot(tmp_episode_act[:,j].cpu(), marker=".")
            if i == 0:
                axs[j].set_title("act_"+str(j))
                axs[j].grid()
    fig.tight_layout()
    plt.show()
    #Plot sample observations
    batch_obs, batch_goal, batch_dist, batch_traj_obs, batch_traj_act = dataset_train.sample(1000)
    tmp_size_proprio = 0
    for label in batch_obs.keys():
        if label != "images":
            tmp_size_proprio += batch_traj_obs[label].size(2)
    fig, axs = plt.subplots(2, tmp_size_proprio)
    tmp_index_proprio = 0
    for label in batch_obs.keys():
        if label == "images":
            continue
        for i in range(batch_traj_obs[label].size(2)):
            for j in range(batch_traj_obs[label].size(0)):
                axs[0,tmp_index_proprio].plot(batch_traj_obs[label][j,:,i].cpu(), marker="o", alpha=0.2)
            axs[0,tmp_index_proprio].set_title(label+"_"+str(i))
            axs[0,tmp_index_proprio].grid()
            axs[1,tmp_index_proprio].violinplot(batch_traj_obs[label][:,:,i].flatten().cpu(), showmeans=True, showmedians=True)
            axs[1,tmp_index_proprio].grid()
            tmp_index_proprio += 1
    fig.tight_layout()
    plt.show()
    #Plot sample actions
    batch_obs, batch_goal, batch_dist, batch_traj_obs, batch_traj_act = dataset_train.sample(1000)
    fig, axs = plt.subplots(2, size_act)
    for i in range(size_act):
        for j in range(batch_traj_act.size(0)):
            axs[0,i].plot(batch_traj_act[j,:,i].cpu(), marker="o", alpha=0.2)
        axs[0,i].set_title("act_"+str(i))
        axs[0,i].grid()
        axs[1,i].violinplot(batch_traj_act[:,:,i].flatten().cpu(), showmeans=True, showmedians=True)
        axs[1,i].grid()
    fig.tight_layout()
    plt.show()
    #Plot samples
    fig, axs = plt.subplots(5, 3+size_act)
    batch_obs, batch_goal, batch_dist, batch_traj_obs, batch_traj_act = dataset_train.sample(5)
    list_ims = []
    for k in range(5):
        if "images" in batch_obs:
            axs[k,0].imshow(batch_obs["images"][k].cpu(), interpolation="nearest")
            axs[k,1].imshow(batch_goal["images"][k].cpu(), interpolation="nearest")
            axs[k,1].set_title("dist={:.2f}".format(batch_dist[k].item()))
            im = axs[k,2].imshow(batch_traj_obs["images"][k,0].cpu(), interpolation="nearest")
        list_ims.append(im)
        for i in range(size_act):
            axs[k,3+i].plot(batch_traj_act[k,:,i].cpu(), marker="o")
            axs[k,3+i].grid()
            axs[k,3+i].set_title("act_"+str(i))
    fig.tight_layout()
    def animate(j):
        for k in range(5):
            list_ims[k].set_array(batch_traj_obs["images"][k,j])
        return tuple(list_ims)
    anim = anm.FuncAnimation(fig, animate, 
        frames=batch_traj_obs["images"].size(1), interval=50, blit=False)
    plt.show()
    #Plot episodes
    fig, axs = plt.subplots(4, 1+size_act)
    list_ims = []
    list_scats = []
    list_data_obs = []
    list_data_act = []
    for k in range(4):
        index_episode = random.randint(0, dataset_train.count_episodes()-1)
        tmp_episode_obs, tmp_episode_act = dataset_train.get_episode(index_episode)
        index_begin = random.randint(0, tmp_episode_obs["images"].size(0)-params["max_goal_dist"])
        index_end = index_begin + params["max_goal_dist"]
        im = axs[k,0].imshow(tmp_episode_obs["images"][index_begin].cpu(), interpolation="nearest")
        list_ims.append(im)
        list_data_obs.append(tmp_episode_obs["images"][index_begin:index_end].clone())
        list_data_act.append(tmp_episode_act[index_begin:index_end].clone())
        for i in range(size_act):
            axs[k,1+i].plot(tmp_episode_act[index_begin:index_end,i], marker="o")
            scat = axs[k,1+i].scatter(0, tmp_episode_act[index_begin,i], 
                marker=".", s=800, color="red", alpha=0.5)
            list_scats.append(scat)
            axs[k,1+i].grid()
    fig.tight_layout()
    def animate(j):
        for k in range(4):
            list_ims[k].set_array(list_data_obs[k][j])
        for k in range(4):
            for i in range(size_act):
                list_scats[k*size_act+i].set_offsets([j, list_data_act[k][j,i]])
        return tuple(list_ims + list_scats)
    anim = anm.FuncAnimation(fig, animate, frames=params["max_goal_dist"], interval=10, blit=True)
    plt.show()
    exit()

#Plot dataset episodes
if params["mode"] == "plot_demo":
    #Retrieve environment background
    env = PlanarPushEnv(
        step_freq=20.0,
        render_window=False,
        image_obs=True,
        image_size=512,
        is_goal_effector=True,
        is_goal_object=True,
        fixed_goal=False,
        variant_name="circle_maze_medium")
    obs, info = env.reset()
    env.set_obs({
        "command_pos": np.array([2.0, 2.0]),
        "effector_pos": np.array([2.0, 2.0]),
        "object_0_pos": np.array([2.0, 2.0]),
        "goal_effector_pos": np.array([2.0, 2.0]),
        "goal_object_pos": np.array([2.0, 2.0]),
        "object_0_angle": np.array([0.0]),
        "object_0_anglevel": np.array([0.0]),
        "goal_object_angle": np.array([0.0]),
        "effector_vel": np.array([0.0, 0.0]),
        "object_0_vel": np.array([0.0, 0.0]),
    })
    obs = env.get_obs()
    #Plot demonstrations
    fig, axs = plt.subplots(1, 1)
    cmap = matplotlib.colormaps["viridis"]
    colors = cmap(np.linspace(0.0, 1.0, dataset_train.count_episodes()))
    for i in np.linspace(0, dataset_train.count_episodes()-1, 5, dtype=np.int32):
        tmp_episode_obs, tmp_episode_act = dataset_train.get_episode(i)
        axs.imshow(np.transpose(env.render(), (1, 0, 2)))
        axs.plot(
            512//2*tmp_episode_obs["obj_pos"][:,0]+512//2, 
            512//2*tmp_episode_obs["obj_pos"][:,1]+512//2,
            alpha=0.6,
            linewidth=4,
        )
        axs.scatter(
            512//2*tmp_episode_obs["obj_pos"][0,0]+512//2, 
            512//2*tmp_episode_obs["obj_pos"][0,1]+512//2,
            alpha=0.6,
            s=300,
            marker="s",
            color=colors[i],
        )
        axs.scatter(
            512//2*tmp_episode_obs["obj_pos"][-1,0]+512//2, 
            512//2*tmp_episode_obs["obj_pos"][-1,1]+512//2,
            alpha=0.6,
            s=300,
            marker="*",
            color=colors[i],
        )
    axs.axis("equal")
    fig.tight_layout()
    plt.show()
    #Plot evaluations
    fig, axs = plt.subplots(1, 1)
    for i in range(dataset_eval.count_episodes()):
        tmp_episode_obs, tmp_episode_act = dataset_eval.get_episode(i)
        axs.imshow(np.transpose(env.render(), (1, 0, 2)))
        axs.plot(
            512//2*tmp_episode_obs["obj_pos"][:,0]+512//2, 
            512//2*tmp_episode_obs["obj_pos"][:,1]+512//2,
            alpha=0.6,
            linewidth=4,
        )
        axs.scatter(
            512//2*tmp_episode_obs["obj_pos"][0,0]+512//2, 
            512//2*tmp_episode_obs["obj_pos"][0,1]+512//2,
            alpha=0.6,
            s=300,
            marker="s",
        )
        axs.scatter(
            512//2*tmp_episode_obs["obj_pos"][-1,0]+512//2, 
            512//2*tmp_episode_obs["obj_pos"][-1,1]+512//2,
            alpha=0.6,
            s=300,
            marker="*",
        )
    axs.axis("equal")
    fig.tight_layout()
    plt.show()
    exit()

#Initialize agents
if params["use_images"]:
    encoder_config = [
        ("images", "impala", 8192, 128),
        ("cmd_pos", params["encoder_proprio_name"], 2, params["encoder_proprio_size"]),
    ]
else:
    encoder_config = [
        #("cmd_pos", params["encoder_proprio_name"], 2, params["encoder_proprio_size"]),
        ("eff_pos", params["encoder_proprio_name"], 2, params["encoder_proprio_size"]),
        ("obj_pos", params["encoder_proprio_name"], 2, params["encoder_proprio_size"]),
    ]
agents = {
    #"agentBC": GCRLAgentBC(
    #    config=params, 
    #    encoder_config=encoder_config,
    #    size_act=size_act,
    #    device=device,
    #),
    "agent0": GCRLAgent0(
        config=params, 
        encoder_config=encoder_config,
        size_act=size_act,
        device=device,
    ),
    "agent1_norl": GCRLAgent1(
        config=params, 
        encoder_config=encoder_config,
        size_act=size_act,
        use_rl=False,
        use_merge_traj=False,
        device=device,
    ),
    "agent1_withrl": GCRLAgent1(
        config=params, 
        encoder_config=encoder_config,
        size_act=size_act,
        use_rl=True,
        use_merge_traj=False,
        device=device,
    ),
    "agent2_norl": GCRLAgent2(
        config=params, 
        encoder_config=encoder_config,
        size_act=size_act,
        use_rl=False,
        device=device,
    ),
    "agent2_withrl": GCRLAgent2(
        config=params, 
        encoder_config=encoder_config,
        size_act=size_act,
        use_rl=True,
        device=device,
    ),
    "agent5_norl": GCRLAgent5(
        config=params, 
        encoder_config=encoder_config,
        size_act=size_act,
        use_rl=False,
        device=device,
    ),
    "agent5_withrl": GCRLAgent5(
        config=params, 
        encoder_config=encoder_config,
        size_act=size_act,
        use_rl=True,
        device=device,
    ),
    "agent6_norl": GCRLAgent6(
        config=params, 
        encoder_config=encoder_config,
        size_act=size_act,
        use_rl=False,
        device=device,
    ),
    "agent6_withrl": GCRLAgent6(
        config=params, 
        encoder_config=encoder_config,
        size_act=size_act,
        use_rl=True,
        device=device,
    ),
}
#Load agents
if params["is_load"]:
    print("Loading agents...")
    for label in agents.keys():
        agents[label].load("/tmp/"+label+"_")
#Verbose
for label in agents.keys():
    print("====", label)
    agents[label].print_parameters()

@torch.no_grad()
def do_plot_model(agent, device):
    batch_obs, batch_goal, batch_dist, batch_traj_obs, batch_traj_act = dataset_train.sample(5)
    list_pred = []
    for j in range(20):
        list_pred.append(agent.inference(batch_obs, batch_goal))
    fig, axs = plt.subplots(5, 2+size_act)
    for k in range(5):
        if "images" in batch_obs:
            axs[k,0].imshow(batch_obs["images"][k].cpu(), interpolation="nearest")
            axs[k,1].imshow(batch_obs["images"][k].cpu(), interpolation="nearest")
        axs[k,0].set_title("obs")
        axs[k,1].set_title("goal length={:.2f}".format(batch_dist[k].item()))
        for i in range(size_act):
            axs[k,2+i].plot(batch_traj_act[k,:,i].cpu(), marker="o", color="blue")
            for j in range(20):
                axs[k,2+i].plot(list_pred[j][k,:,i].cpu(), marker="o", color="red", alpha=0.2)
            axs[k,2+i].set_title("act_"+str(i))
            axs[k,2+i].grid()
    fig.tight_layout()
    return fig, axs

@torch.no_grad()
def do_plot_eval(agent, device):
    size_episode = dataset_eval.count_episodes()
    fig, axs = plt.subplots(size_episode, 5, width_ratios=[1, 1, 3, 3, 3], squeeze=False)
    for i in range(size_episode):
        tmp_obs, tmp_act = dataset_eval.get_episode(i)
        axs[i,0].imshow(tmp_obs["images"][0].cpu())
        axs[i,0].plot(
            -128//2*tmp_obs["obj_pos"][:,1].cpu()+128//2, 
            -128//2*tmp_obs["obj_pos"][:,0].cpu()+128//2)
        axs[i,0].plot(
            -128//2*tmp_obs["eff_pos"][:,1].cpu()+128//2, 
            -128//2*tmp_obs["eff_pos"][:,0].cpu()+128//2)
        axs[i,1].imshow(tmp_obs["images"][-1].cpu())
        axs[i,0].set_title("Init")
        axs[i,1].set_title("Goal dist={}".format(tmp_act.shape[0]/params["max_goal_dist"]))
        for j in range(size_act):
            axs[i,2+j].plot(tmp_act[:,j].cpu())
            axs[i,2+j].set_title("act_"+str(j))
            axs[i,2+j].grid()
        for k in range(0, tmp_act.shape[0], params["inference_horizon"]):
            tmp_pred = agent.inference(
                {l:v[k:k+1] for l,v in tmp_obs.items()},
                {l:v[-2:-1] for l,v in tmp_obs.items()},
            )
            for j in range(size_act):
                axs[i,2+j].plot(
                    k+np.array(range(params["inference_horizon"])), 
                    tmp_pred[0,0:params["inference_horizon"],j].cpu(),
                    marker=".")
            if "critic_1" in agent._info:
                axs[i,2+size_act].scatter(k, agent._info["critic_1"].mean().cpu(), color="red")
                axs[i,2+size_act].scatter(k, agent._info["critic_2"].mean().cpu(), color="blue")
                axs[i,2+size_act].scatter(k, agent._info["critic_max"].mean().cpu(), color="black")
        axs[i,2+size_act].set_title("Critic")
        axs[i,2+size_act].grid()
    fig.tight_layout()
    return fig, axs

#Plot models
if params["mode"] == "plot_model":
    for label in agents.keys():
        print("==== Plot Model", label)
        fig, axs = do_plot_model(agents[label], device)
        plt.show()
    exit()

#Plot eval
if params["mode"] == "plot_eval":
    for label in agents.keys():
        print("==== Plot Eval", label)
        fig, axs = do_plot_eval(agents[label], device)
        plt.show()
    exit()

@torch.no_grad()
def do_metrics_model(agent, device):
    list_delta = []
    for k in range(20):
        size_batch = params["size_batch"]
        batch_obs, batch_goal, batch_dist, batch_traj_obs, batch_traj_act = dataset_train.sample(size_batch)
        list_delta.append(agent.inference(batch_obs, batch_goal)[:,0,:] - batch_obs["cmd_pos"])
    metrics = dict()
    metrics["error_cmd_origin"] = torch.cat(list_delta, dim=0).abs().sum(dim=1).mean().cpu().item()
    return metrics

#Plot models
if params["mode"] == "metrics_model":
    for label in agents.keys():
        print("==== Metrics", label)
        metrics = do_metrics_model(agents[label], device)
        for label in metrics.keys():
            print(label, metrics[label])
    exit()

@torch.no_grad()
def do_eval_single(
        agent,
        seed: int,
    ):
    """Simulate a single evaluation run"""
    env = PlanarPushEnv(
        step_freq=20.0,
        render_window=False,
        image_obs=True,
        image_size=512,
        is_goal_effector=True,
        is_goal_object=True,
        fixed_goal=False,
        variant_name="circle_maze_medium")
    obs, info = env.reset(seed=seed)
    tmp_obs, tmp_act = dataset_eval.get_episode(seed)
    env.set_obs({
        "command_pos": tmp_obs["cmd_pos"][0].cpu().numpy(),
        "effector_pos": tmp_obs["eff_pos"][0].cpu().numpy(),
        "object_0_pos": tmp_obs["obj_pos"][0].cpu().numpy(),
        "goal_effector_pos": tmp_obs["eff_pos"][-1].cpu().numpy(),
        "goal_object_pos": tmp_obs["obj_pos"][-1].cpu().numpy(),
        "object_0_angle": np.array([0.0]),
        "object_0_anglevel": np.array([0.0]),
        "goal_object_angle": np.array([0.0]),
        "effector_vel": np.array([0.0, 0.0]),
        "object_0_vel": np.array([0.0, 0.0]),
    })
    obs = env.get_obs()
    frame_goal = obs["goal_image"].copy()
    list_trajs_act = []
    list_cmd_pos = []
    list_info_env = []
    list_info_agent = []
    list_frames = []
    is_success = False
    for k in range(0, params["inference_steps"], params["inference_horizon"]-1):
        #Evaluate the policy
        tmp_trajs_act = agent.inference(
            {
                "images": encode_frame(255.0*obs["obs_image"], device=device).unsqueeze(0),
                "cmd_pos": torch.tensor(obs["command_pos"], device=device, dtype=torch.float32).unsqueeze(0),
                "eff_pos": torch.tensor(obs["effector_pos"], device=device, dtype=torch.float32).unsqueeze(0),
                "obj_pos": torch.tensor(obs["object_0_pos"], device=device, dtype=torch.float32).unsqueeze(0),
            }, 
            {
                "images": encode_frame(255.0*obs["goal_image"], device=device).unsqueeze(0),
                "cmd_pos": torch.tensor(obs["goal_effector_pos"], device=device, dtype=torch.float32).unsqueeze(0),
                "eff_pos": torch.tensor(obs["goal_effector_pos"], device=device, dtype=torch.float32).unsqueeze(0),
                "obj_pos": torch.tensor(obs["goal_object_pos"], device=device, dtype=torch.float32).unsqueeze(0),
            },
        )
        list_cmd_pos.append(copy.deepcopy(obs["command_pos"]))
        list_trajs_act.append(tmp_trajs_act.cpu().clone())
        list_info_agent.append(copy.deepcopy(agent._info))
        #Simulate environment
        for l in range(1, params["inference_horizon"], 1):
            tmp_act = tmp_trajs_act[0,l,:].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step({"command_pos": tmp_act})
            list_frames.append(env.render())
            list_info_env.append(info)
            if reward < 1.0:
                is_success = True
    return is_success, list_trajs_act, list_cmd_pos, list_frames, frame_goal, list_info_env, list_info_agent

@torch.no_grad()
def do_plot_eval_single(
        agent,
        seed: int,
    ):
    is_success, list_trajs_act, list_cmd_pos, list_frames, frame_goal, list_info_env, list_info_agent = do_eval_single(agent, seed)
    tmp_dist = np.array([info["dist_object_pos"] for info in list_info_env]).min()
    tmp_origin = np.array([(act[:,0,:]-cmd[:]).abs().sum() for act,cmd in zip(list_trajs_act, list_cmd_pos)]).mean()
    print("Episode length:", len(list_frames))
    print("dist_object_pos:", tmp_dist)
    print("error_origin_act:", tmp_origin)
    print("is_success:", is_success)
    #Animate episode
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(24, 14)
    axs[1,0].imshow(frame_goal)
    axs[1,0].set_title("Goal is_success="+str(is_success))
    axs[1,1].plot([info["dist_effector_pos"] for info in list_info_env], label="dist_effector_pos")
    axs[1,1].plot([info["dist_object_pos"] for info in list_info_env], label="dist_object_pos")
    axs[1,1].set_title("Info")
    axs[1,1].set_ylim(0.0, 2.0)
    axs[1,1].grid()
    axs[1,1].legend()
    im = axs[0,0].imshow(list_frames[0])
    axs[0,0].set_title("Episode")
    lines = []
    scats = []
    for i in range(size_act):
        line, = axs[0,1].plot(list_trajs_act[0][0,:,i], marker=".")
        scat = axs[0,1].scatter(0.0, list_cmd_pos[0][i], s=500)
        lines.append(line)
        scats.append(scat)
    axs[0,1].set_ylim(-1.0, 1.0)
    axs[0,1].grid()
    title = axs[0,0].text(0.0,0.9, "", 
        fontsize=12, va="top", bbox=dict(facecolor="white", edgecolor="black", alpha=0.6))
    fig.tight_layout()
    def animate(k):
        title.set_text("Frame: {}/{}".format(str(k), len(list_frames)))
        im.set_array(list_frames[k])
        tmp_idx = k//params["inference_horizon"]
        for i in range(size_act):
            lines[i].set_ydata(list_trajs_act[tmp_idx][0,:,i])
            scats[i].set_offsets([0.0, list_cmd_pos[tmp_idx][i]])
        lll = [im, title]
        for i in range(size_act):
            lll.append(lines[i])
            lll.append(scats[i])
        return tuple(lll)
    anim = anm.FuncAnimation(fig, animate, frames=len(list_frames), interval=10, blit=True)
    #anim.save("/tmp/anim.mp4", writer="ffmpeg")
    plt.show()

#Run single evaluation simulation
if params["mode"] == "eval_single":
    for label in agents.keys():
        print("==== Eval", label)
        do_plot_eval_single(agents[label], params["seed"])
    exit()

@torch.no_grad()
def do_eval_stats(
        agent,
    ):
    list_is_success = []
    list_std_success = []
    list_dist = []
    list_origin = []
    for k in range(dataset_eval.count_episodes()):
        list_run_is_success = []
        for i in range(params["num_eval_episode"]):
            is_success, list_trajs_act, list_cmd_pos, list_frames, frame_goal, list_info_env, list_info_agent = do_eval_single(agent, seed=k)
            tmp_dist = np.array([info["dist_object_pos"] for info in list_info_env]).min()
            tmp_origin = np.array([(act[:,0,:]-cmd[:]).abs().sum() for act,cmd in zip(list_trajs_act, list_cmd_pos)]).mean()
            list_is_success.append(float(is_success))
            list_run_is_success.append(float(is_success))
            list_dist.append(tmp_dist)
            list_origin.append(tmp_origin)
            print(type(agent).__name__, "seed", k, "try", i, "dist_obj", tmp_dist, "error_origin", tmp_origin, "is_success", is_success)
        tmp_std_success = np.array(list_run_is_success).std()
        list_std_success.append(tmp_std_success)
        print("std:", tmp_std_success)
    overall_success = np.array(list_is_success).mean()
    overall_std = np.array(list_std_success).mean()
    overall_dist = np.array(list_dist).mean()
    overall_origin = np.array(list_origin).mean()
    print("overall_success:", overall_success, "overall_std", overall_std, "overall_dist:", overall_dist, "overall_origin:", overall_origin)
    return overall_success, overall_std, overall_dist, overall_origin

#Run statistics evaluation
if params["mode"] == "eval_stats":
    for label in agents.keys():
        print("==== Eval", label)
        do_eval_stats(agents[label])
    exit()

#Agent training
if params["mode"] == "train":
    #Training loop
    time_start = time.time()
    with tqdm(range(params["epoch"]+1), desc="Epoch") as range_epoch:
        for epoch in range_epoch:
            batch_obs, batch_goal, batch_dist, batch_traj_obs, batch_traj_act = dataset_train.sample(params["size_batch"])
            batch_other_obs = dataset_train.sample_obs(params["size_batch"])
            for label in agents.keys():
                loss = agents[label].train(
                    batch_obs, 
                    batch_goal, 
                    batch_dist, 
                    batch_traj_obs, 
                    batch_traj_act, 
                    batch_other_obs,
                )
                logger.add_scalar("loss/"+label, loss, epoch)
            logger.add_scalar("system/time", time.time()-time_start, epoch)
            #Save agents
            if epoch > 0 and epoch%params["save_model_every"] == 0:
                for label in agents.keys():
                    agents[label].save("/tmp/"+label+"_")
                    tmp_epoch = None
                    for name in agents[label]._model_encoders.keys():
                        logger.add_model(label+"_model_encoder_"+name, agents[label]._ema_encoders[name].getModel(), tmp_epoch)
                    if agents[label]._model_critic_1 is not None:
                        logger.add_model(label+"_model_critic_1", agents[label]._ema_critic_1.getModel(), tmp_epoch)
                    if agents[label]._model_critic_2 is not None:
                        logger.add_model(label+"_model_critic_2", agents[label]._ema_critic_2.getModel(), tmp_epoch)
                    if agents[label]._model_planner is not None:
                        logger.add_model(label+"_model_planner", agents[label]._ema_planner.getModel(), tmp_epoch)
                    if agents[label]._model_actor is not None:
                        logger.add_model(label+"_model_actor", agents[label]._ema_actor.getModel(), tmp_epoch)
                    if agents[label]._model_world is not None:
                        logger.add_model(label+"_model_world", agents[label]._ema_world.getModel(), tmp_epoch)
            #Plot agents
            if epoch > 0 and epoch%params["plot_model_every"] == 0:
                for label in agents.keys():
                    #Plot model
                    fig, axs = do_plot_model(agents[label], device)
                    logger.add_frame_as_fig("model_"+label, fig, epoch)
                    fig, axs = do_plot_eval(agents[label], device)
                    logger.add_frame_as_fig("eval_"+label, fig, epoch)
                    #Metrics
                    metrics = do_metrics_model(agents[label], device)
                    for name in metrics.keys():
                        logger.add_scalar("metrics_"+name+"/"+label, metrics[name], epoch)
            #Eval agents
            if epoch >= 0 and epoch%params["eval_model_every"] == 0:
                for label in agents.keys():
                    overall_success, overall_std, overall_dist, overall_origin = do_eval_stats(agents[label])
                    logger.add_scalar("eval_success/"+label, overall_success, epoch)
                    logger.add_scalar("eval_std/"+label, overall_std, epoch)
                    logger.add_scalar("eval_dist/"+label, overall_dist, epoch)
                    logger.add_scalar("eval_origin/"+label, overall_origin, epoch)
    #Finalize logger
    logger.set_parameters(params)
    logger.close()
    exit()

