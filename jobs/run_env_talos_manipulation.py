import time
import math
import numpy as np
import h5py
import random
import cv2
import torch
import copy
import platform
import threading
from collections import OrderedDict
import matplotlib
import matplotlib.animation as anm
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict, Union

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
from gcrl.agent import GCRLAgentBase, GCRLAgent0, GCRLAgent1, GCRLAgent2, GCRLAgent5, GCRLAgent6
from gcrl.dataset import GCTrajsDataset
from gcrl.deployment import PolicyInference
from utils.rotation_roma import rotation_unitquat_to_6dvec, rotation_6dvec_to_unitquat

#Job parameters
params = {
    #Job mode
    "mode": "",
    #Dataset processing
    "path_hdf5_demo": "/tmp/docker_share/demo_all.h5",
    "path_npz_demo": "/tmp/docker_share/dataset_all.npz",
    #Device
    "use_cuda": False,
    #Loading previous model
    "is_load": False,
    #Trajectories
    "trajs_obs_len": 3,
    "trajs_act_len": 16,
    "trajs_obs_stride": 15,
    "trajs_act_stride": 3,
    "max_goal_dist": 1000,
    #Training
    "epoch": 500000,
    "ema_cutoff": 1000,
    "size_batch": 128,
    "apply_augmentation": True,
    "save_model_every": 20000,
}
params, logger = job.init(params, "Train Talos vision-based manipulation with Extremum Flow Matching Agents")
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
        #RandomVerticalFlip(p=1.0),
        #RandomHorizontalFlip(p=1.0),
    ])
    tensor_frame = transforms(tensor_frame)
    tensor_frame = tensor_frame.permute(1,2,0)
    return tensor_frame

@torch.no_grad()
def encode_proprio(
        dict_in: Dict[str, torch.Tensor],
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
    dict_out = OrderedDict()
    for label_in in dict_in.keys():
        tensor_in = dict_in[label_in]
        if isinstance(tensor_in, np.ndarray):
            tensor_in = torch.tensor(tensor_in, dtype=torch.float32, device=device)
        else:
            tensor_in = tensor_in.to(device)
        torch._assert(torch.is_tensor(tensor_in), "")
        if label_in.endswith("_quat"):
            label_out = label_in.replace("_quat", "_rot6d")
            tensor_out = rotation_unitquat_to_6dvec(tensor_in)
        else:
            label_out = label_in
            tensor_out = tensor_in
        dict_out[label_out] = tensor_out
    return dict_out

#Extract demonstrations from hdf5 dataset and generate processed npz dataset
if params["mode"] == "process_dataset":
    list_demos_images = []
    list_demos_proprio = OrderedDict()
    list_demos_terminals = []
    with h5py.File(params["path_hdf5_demo"], "r") as file_dataset:
        for name_demo in file_dataset:
            #Extract and process images
            key_images = name_demo + "/images/cam_left_color"
            tmp_dtype = file_dataset[key_images].dtype
            tmp_shape = file_dataset[key_images].shape
            list_images = []
            print("Processing demonstration frames:", key_images)
            for i in range(0, tmp_shape[0], 1):
                frame_raw = cv2.imdecode(file_dataset[key_images][i], cv2.IMREAD_UNCHANGED)
                frame_processed = encode_frame(frame_raw).numpy()
                list_images.append(frame_processed)
                if False:
                    key_ts = name_demo + "/timestamps"
                    print(
                        i, 
                        "frame_raw", type(frame_raw), frame_raw.shape, frame_raw.dtype, 
                        "frame_processed", type(frame_processed), frame_processed.shape, frame_processed.dtype, 
                        "--- ts", file_dataset[key_ts][i,0], 
                        "delta", (file_dataset[key_ts][i,0]-file_dataset[key_ts][i-1,0] if i > 0 else 0.0))
                    cv2.imshow("frame", 
                        cv2.cvtColor(
                            cv2.resize(frame_processed, dsize=(1000, 1000), interpolation=cv2.INTER_NEAREST), 
                            cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            cv2.destroyAllWindows()
            array_images = np.array(list_images)
            list_demos_images.append(array_images)
            print("array_images", array_images.shape, array_images.dtype)
            #Append terminals
            array_terminals = np.zeros(array_images.shape[0], dtype=np.uint8)
            array_terminals[-1] = 1
            list_demos_terminals.append(array_terminals)
            print("array_terminals", array_terminals.shape, array_terminals.dtype)
            #Extract proprioceptive data
            for label in file_dataset[name_demo + "/proprio"]:
                key_proprio = name_demo + "/proprio/" + label
                array_proprio = file_dataset[key_proprio][()]
                if not label in list_demos_proprio:
                    list_demos_proprio[label] = []
                list_demos_proprio[label].append(array_proprio)
                print("array_proprio", label, array_proprio.shape, array_proprio.dtype)
    for label in list_demos_proprio:
        list_demos_proprio[label] = np.concatenate(list_demos_proprio[label], axis=0)
    list_demos_proprio = encode_proprio(list_demos_proprio)
    #Conversion to npz
    print("Writing dataset:", params["path_npz_demo"])
    np.savez(
        params["path_npz_demo"],
        images=np.concatenate(list_demos_images, axis=0),
        terminals=np.concatenate(list_demos_terminals, axis=0),
        **list_demos_proprio, 
    )
    exit()

#Load processed dataset 
print("Load dataset:", params["path_npz_demo"])
data_train_dataset = np.load(params["path_npz_demo"])
print("Dataset Train:")
for n,v in data_train_dataset.items():
    print("    ", n, v.shape, v.dtype)

#Create Goal Conditioned Dataset for trajectories
dataset_train = GCTrajsDataset(
    dict_obs={
        "images": data_train_dataset["images"],
        "cmd_pos": data_train_dataset["cmd_pos"],
        "cmd_rot6d": data_train_dataset["cmd_rot6d"],
        "read_pos": data_train_dataset["read_pos"],
        "read_rot6d": data_train_dataset["read_rot6d"],
    },
    array_act=np.concatenate([
        data_train_dataset["cmd_pos"],
        data_train_dataset["cmd_rot6d"],
        data_train_dataset["cmd_gripper"],
        ], axis=1),
    array_terminal=data_train_dataset["terminals"],
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
size_act = tmp_act.size(1)
print("size_act", size_act)

#Plot dataset episodes
if params["mode"] == "plots":
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

@torch.no_grad()
def apply_image_augmentation(tensor_obs: torch.Tensor) -> torch.Tensor:
    tensor_obs = tensor_obs.permute(0,3,1,2)
    transforms = SequentialTransform([
        RandomColorJitter(
            num_chunks=1, 
            batch_transform=False, 
            p=0.4, 
            brightness=0.1, 
            contrast=0.1, 
            saturation=0.1, 
            hue=0.08),
        #RandomGaussianBlur(
        #    batch_transform=False, 
        #    kernel_size=3,
        #    sigma=0.5,
        #    p=0.3),
        #RandomErasing(
        #    num_chunks=1,
        #    batch_transform=False, 
        #    scale=(0.02, 0.33), ratio=(0.3, 3.3),
        #    p=0.1),
        #RandomAffine(
        #    num_chunks=1,
        #    batch_transform=False, 
        #    degrees=5.0,
        #    translate=(0.05, 0.05),
        #    scale=(0.95,1.05)),
    ])
    tensor_obs = transforms(tensor_obs)
    tensor_obs = tensor_obs.permute(0,2,3,1)
    return tensor_obs

#Plot image augmentation
if params["mode"] == "plot_augmentation":
    batch_obs = dataset_train.sample_obs(8, device)
    print(batch_obs["images"].size(), batch_obs["images"].dtype)
    tmp_obs = apply_image_augmentation(batch_obs["images"].repeat(5,1,1,1))
    fig, axs = plt.subplots(5,8)
    for k in range(5*8):
        axs.flatten()[k].imshow(tmp_obs[k].cpu(), interpolation="nearest")
    fig.tight_layout()
    plt.show()
    exit()

#Initialize agents
config = GCRLAgentBase.get_config()
config["epoch"] = params["epoch"]
config["ema_cutoff"] = params["ema_cutoff"]
config["trajs_obs_len"] = params["trajs_obs_len"]
config["trajs_act_len"] = params["trajs_act_len"]
config["trajs_obs_stride"] = params["trajs_obs_stride"]
config["trajs_act_stride"] = params["trajs_act_stride"]
config["max_goal_dist"] = params["max_goal_dist"]
print("Agents config:")
for key,value in config.items():
    print("    {0:40} {1}".format(key, value))
agents = {
    #"agent0": GCRLAgent0(
    #    config=config, 
    #    encoder_config=[
    #        ("images", "impala", 8192, 64),
    #        ("cmd_pos", "identity", 3, 3),
    #        ("cmd_rot6d", "identity", 6, 6),
    #        ("read_pos", "identity", 3, 3),
    #        ("read_rot6d", "identity", 6, 6),
    #    ],
    #    size_act=size_act,
    #    device=device,
    #),
    "agent1_norl_2": GCRLAgent1(
        config=config, 
        encoder_config=[
            ("images", "impala", 8192, 64),
            ("cmd_pos", "identity", 3, 3),
            ("cmd_rot6d", "identity", 6, 6),
            ("read_pos", "identity", 3, 3),
            ("read_rot6d", "identity", 6, 6),
        ],
        size_act=size_act,
        use_rl=False,
        use_merge_traj=False,
        device=device,
    ),
    #"agent1_withrl": GCRLAgent1(
    #    config=config, 
    #    encoder_config=[
    #        ("images", "impala", 8192, 64),
    #        ("cmd_pos", "identity", 3, 3),
    #        ("cmd_rot6d", "identity", 6, 6),
    #        ("read_pos", "identity", 3, 3),
    #        ("read_rot6d", "identity", 6, 6),
    #    ],
    #    size_act=size_act,
    #    use_rl=True,
    #    use_merge_traj=False,
    #    device=device,
    #),
    #"agent2_norl": GCRLAgent2(
    #    config=config, 
    #    encoder_config=[
    #        ("images", "impala", 8192, 64),
    #        ("cmd_pos", "identity", 3, 3),
    #        ("cmd_rot6d", "identity", 6, 6),
    #        ("read_pos", "identity", 3, 3),
    #        ("read_rot6d", "identity", 6, 6),
    #    ],
    #    size_act=size_act,
    #    use_rl=False,
    #    device=device,
    #),
    #"agent2_withrl": GCRLAgent2(
    #    config=config, 
    #    encoder_config=[
    #        ("images", "impala", 8192, 64),
    #        ("cmd_pos", "identity", 3, 3),
    #        ("cmd_rot6d", "identity", 6, 6),
    #        ("read_pos", "identity", 3, 3),
    #        ("read_rot6d", "identity", 6, 6),
    #    ],
    #    size_act=size_act,
    #    use_rl=True,
    #    device=device,
    #),
    #"agent5_norl": GCRLAgent5(
    #    config=config, 
    #    encoder_config=[
    #        ("images", "impala", 8192, 64),
    #        ("cmd_pos", "identity", 3, 3),
    #        ("cmd_rot6d", "identity", 6, 6),
    #        ("read_pos", "identity", 3, 3),
    #        ("read_rot6d", "identity", 6, 6),
    #    ],
    #    size_act=size_act,
    #    use_rl=False,
    #    device=device,
    #),
    #"agent5_withrl": GCRLAgent5(
    #    config=config, 
    #    encoder_config=[
    #        ("images", "impala", 8192, 64),
    #        ("cmd_pos", "identity", 3, 3),
    #        ("cmd_rot6d", "identity", 6, 6),
    #        ("read_pos", "identity", 3, 3),
    #        ("read_rot6d", "identity", 6, 6),
    #    ],
    #    size_act=size_act,
    #    use_rl=True,
    #    device=device,
    #),
    #"agent6_norl": GCRLAgent6(
    #    config=config, 
    #    encoder_config=[
    #        ("images", "impala", 8192, 64),
    #        ("cmd_pos", "identity", 3, 3),
    #        ("cmd_rot6d", "identity", 6, 6),
    #        ("read_pos", "identity", 3, 3),
    #        ("read_rot6d", "identity", 6, 6),
    #    ],
    #    size_act=size_act,
    #    use_rl=False,
    #    device=device,
    #),
    #"agent6_withrl": GCRLAgent6(
    #    config=config, 
    #    encoder_config=[
    #        ("images", "impala", 8192, 64),
    #        ("cmd_pos", "identity", 3, 3),
    #        ("cmd_rot6d", "identity", 6, 6),
    #        ("read_pos", "identity", 3, 3),
    #        ("read_rot6d", "identity", 6, 6),
    #    ],
    #    size_act=size_act,
    #    use_rl=True,
    #    device=device,
    #),
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

#Plot models
if params["mode"] == "plot_model":
    for label in agents.keys():
        print("==== Plot", label)
        fig, axs = do_plot_model(agents[label], device)
        plt.show()
    exit()

#Agent training
if params["mode"] == "train":
    #Training loop
    time_start = time.time()
    with tqdm(range(params["epoch"]+1), desc="Epoch") as range_epoch:
        for epoch in range_epoch:
            #Sample batch
            batch_obs, batch_goal, batch_dist, batch_traj_obs, batch_traj_act = dataset_train.sample(params["size_batch"])
            batch_other_obs = dataset_train.sample_obs(params["size_batch"])
            #Apply image augmentation
            if params["apply_augmentation"]:
                batch_obs["images"] = apply_image_augmentation(batch_obs["images"])
                batch_goal["images"] = apply_image_augmentation(batch_goal["images"])
                batch_other_obs["images"] = apply_image_augmentation(batch_other_obs["images"])
            #Train agents
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
                    for name in agents[label]._model_encoders.keys():
                        logger.add_model(label+"_model_encoder_"+name, agents[label]._ema_encoders[name].getModel(), epoch)
                    if agents[label]._model_critic_1 is not None:
                        logger.add_model(label+"_model_critic_1", agents[label]._ema_critic_1.getModel(), epoch)
                    if agents[label]._model_critic_2 is not None:
                        logger.add_model(label+"_model_critic_2", agents[label]._ema_critic_2.getModel(), epoch)
                    if agents[label]._model_planner is not None:
                        logger.add_model(label+"_model_planner", agents[label]._ema_planner.getModel(), epoch)
                    if agents[label]._model_actor is not None:
                        logger.add_model(label+"_model_actor", agents[label]._ema_actor.getModel(), epoch)
                    if agents[label]._model_world is not None:
                        logger.add_model(label+"_model_world", agents[label]._ema_world.getModel(), epoch)
                    #Plot model
                    fig, axs = do_plot_model(agents[label], device)
                    logger.add_frame_as_fig(label, fig, epoch)
    #Finalize logger
    logger.set_parameters(params)
    logger.close()
    exit()

class AgentDeployment(torch.nn.Module):
    """Deployment model for given agent"""
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    @torch.no_grad()
    def forward(self, 
            dict_obs_raw: Dict[str, torch.Tensor], 
            dict_goal_raw: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
        """Agent policy inference.
        Args:
            dict_obs_raw: Dict of tensor of non processed current observations.
            dict_goal_raw: Dict of tensor of non process desired goal.
        Returns the dict:
            cmd_pos: Predicted command position.
            cmd_quat: Predicted command quaternion.
        """
        dict_obs_proc = OrderedDict()
        dict_goal_proc = OrderedDict()
        for label in dict_obs_raw.keys():
            if label == "images":
                dict_obs_proc[label] = encode_frame(dict_obs_raw[label]).unsqueeze(0)
                dict_goal_proc[label] = encode_frame(dict_goal_raw[label]).unsqueeze(0)
            else:
                dict_obs_proc[label] = dict_obs_raw[label].unsqueeze(0)
                dict_goal_proc[label] = dict_goal_raw[label].unsqueeze(0)
        dict_obs_proc = encode_proprio(dict_obs_proc)
        dict_goal_proc = encode_proprio(dict_goal_proc)
        pred_act = self._agent.inference(dict_obs_proc, dict_goal_proc)[0]
        pred_cmd_pos = pred_act[:,0:3]
        pred_cmd_rot6d = pred_act[:,3:9]
        pred_cmd_gripper = pred_act[:,9:10]
        pred_cmd_quat = rotation_6dvec_to_unitquat(pred_cmd_rot6d)
        tmp_out = OrderedDict()
        tmp_out["cmd_pos"] = pred_cmd_pos
        tmp_out["cmd_quat"] = pred_cmd_quat
        tmp_out["cmd_gripper"] = pred_cmd_gripper
        return tmp_out

#Export agents for deployment
if params["mode"] == "export_agents":
    with h5py.File(params["path_hdf5_demo"], "r") as file_dataset:
        name_demo = list(file_dataset)[0]
        key_images = name_demo + "/images/cam_left_color"
        for label_agent in agents.keys():
            print("==== Export", label_agent)
            #Load dummy data
            dict_data = OrderedDict()
            dict_data["images"] = torch.tensor(
                cv2.imdecode(file_dataset[key_images][0], cv2.IMREAD_UNCHANGED),
                dtype=torch.uint8, 
                device=device)
            for label in file_dataset[name_demo + "/proprio"]:
                key_proprio = name_demo + "/proprio/" + label
                dict_data[label] = torch.tensor(
                    file_dataset[key_proprio][0], 
                    dtype=torch.float32, 
                    device=device)
            #Export agent to torch script
            deployment = AgentDeployment(agents[label_agent])
            traced_agent = torch.jit.trace(
                deployment,
                (dict_data, dict_data),
                check_trace=False, strict=False)
            optimized_agent = torch.jit.optimize_for_inference(traced_agent)
            if str(device) == "cpu":
                tmp_name = "/tmp/"+label_agent+".cpu.pt"
            else:
                tmp_name = "/tmp/"+label_agent+".gpu.pt"
            optimized_agent.save(tmp_name)
            print("Exporting", tmp_name, "Done.")
    exit()

#Test exported agents
if params["mode"] == "test_agents":
    for label in agents.keys():
        print("==== Test", label)
        with torch.no_grad():
            if str(device) == "cpu":
                tmp_name = "/tmp/"+label+".cpu.pt"
            else:
                tmp_name = "/tmp/"+label+".gpu.pt"
            print("Loading", tmp_name)
            agent = torch.jit.load(tmp_name).to(device)
            agent.eval()
            with h5py.File(params["path_hdf5_demo"], "r") as file_dataset:
                name_demo = random.choice(list(file_dataset.keys()))
                key_images = name_demo + "/images/cam_left_color"
                tmp_shape = file_dataset[key_images].shape
                tmp_eval_length = params["max_goal_dist"]
                idx_begin = random.randint(0, tmp_shape[0]-tmp_eval_length)
                idx_end = idx_begin + tmp_eval_length
                dict_goal = OrderedDict()
                dict_goal["images"] = torch.tensor(
                    cv2.imdecode(file_dataset[key_images][idx_end], cv2.IMREAD_UNCHANGED),
                    dtype=torch.uint8,
                    device=device)
                for label in file_dataset[name_demo + "/proprio"]:
                    key_proprio = name_demo + "/proprio/" + label
                    dict_goal[label] = torch.tensor(
                        file_dataset[key_proprio][idx_end], 
                        dtype=torch.float32,
                        device=device)
                list_obs = []
                list_pred = []
                for i in range(idx_begin, idx_end, 1):
                    time1 = time.time()
                    dict_obs = OrderedDict()
                    dict_obs["images"] = torch.tensor(
                        cv2.imdecode(file_dataset[key_images][i], cv2.IMREAD_UNCHANGED),
                        dtype=torch.uint8,
                        device=device)
                    for label in file_dataset[name_demo + "/proprio"]:
                        key_proprio = name_demo + "/proprio/" + label
                        dict_obs[label] = torch.tensor(
                            file_dataset[key_proprio][i], 
                            dtype=torch.float32,
                            device=device)
                    time2 = time.time()
                    dict_pred = agent(dict_obs, dict_goal)
                    time3 = time.time()
                    print(name_demo, i, dict_pred.keys(), time2-time1, time3-time2)
                    list_obs.append(dict_obs)
                    list_pred.append(dict_pred)
            fig, axs = plt.subplots(2, 8)
            im = axs[0,0].imshow(encode_frame(list_obs[0]["images"]), interpolation="nearest")
            title = axs[0,0].text(0.0,0.9, "", 
                fontsize=12, va="top", bbox=dict(facecolor="white", edgecolor="black", alpha=0.6))
            axs[0,1].imshow(encode_frame(dict_goal["images"]), interpolation="nearest")
            for i in range(len(list_obs)):
                axs[1,0].scatter(i, list_obs[i]["cmd_pos"][0], color="blue")
                axs[1,1].scatter(i, list_obs[i]["cmd_pos"][1], color="blue")
                axs[1,2].scatter(i, list_obs[i]["cmd_pos"][2], color="blue")
                axs[1,3].scatter(i, list_obs[i]["cmd_quat"][0], color="blue")
                axs[1,4].scatter(i, list_obs[i]["cmd_quat"][1], color="blue")
                axs[1,5].scatter(i, list_obs[i]["cmd_quat"][2], color="blue")
                axs[1,6].scatter(i, list_obs[i]["cmd_quat"][3], color="blue")
                axs[1,7].scatter(i, list_obs[i]["cmd_gripper"][0], color="blue")
            for i in range(0, len(list_obs), params["trajs_act_len"]*params["trajs_act_stride"]):
                axs[1,0].plot(
                    np.arange(i,i+params["trajs_act_len"]*params["trajs_act_stride"], params["trajs_act_stride"]), 
                    list_pred[i]["cmd_pos"][:,0], marker=".", alpha=1.0)
                axs[1,1].plot(
                    np.arange(i,i+params["trajs_act_len"]*params["trajs_act_stride"], params["trajs_act_stride"]), 
                    list_pred[i]["cmd_pos"][:,1], marker=".", alpha=1.0)
                axs[1,2].plot(
                    np.arange(i,i+params["trajs_act_len"]*params["trajs_act_stride"], params["trajs_act_stride"]), 
                    list_pred[i]["cmd_pos"][:,2], marker=".", alpha=1.0)
                axs[1,3].plot(
                    np.arange(i,i+params["trajs_act_len"]*params["trajs_act_stride"], params["trajs_act_stride"]), 
                    list_pred[i]["cmd_quat"][:,0], marker=".", alpha=1.0)
                axs[1,4].plot(
                    np.arange(i,i+params["trajs_act_len"]*params["trajs_act_stride"], params["trajs_act_stride"]), 
                    list_pred[i]["cmd_quat"][:,1], marker=".", alpha=1.0)
                axs[1,5].plot(
                    np.arange(i,i+params["trajs_act_len"]*params["trajs_act_stride"], params["trajs_act_stride"]), 
                    list_pred[i]["cmd_quat"][:,2], marker=".", alpha=1.0)
                axs[1,6].plot(
                    np.arange(i,i+params["trajs_act_len"]*params["trajs_act_stride"], params["trajs_act_stride"]), 
                    list_pred[i]["cmd_quat"][:,3], marker=".", alpha=1.0)
                axs[1,7].plot(
                    np.arange(i,i+params["trajs_act_len"]*params["trajs_act_stride"], params["trajs_act_stride"]), 
                    list_pred[i]["cmd_gripper"][:,0], marker=".", alpha=1.0)
            fig.tight_layout()
            def animate(k):
                title.set_text("Frame: {}/{}".format(str(k), len(list_obs)))
                im.set_array(encode_frame(list_obs[k]["images"]))
                return im, title
            anim = anm.FuncAnimation(fig, animate, frames=len(list_obs), interval=10, blit=True)
            plt.show()
    exit()

#Test asynchronous inference and interpolation
if params["mode"] == "test_inference":
    if str(device) == "cpu":
        tmp_name = "/tmp/agent0.cpu.pt"
    else:
        tmp_name = "/tmp/agent0.gpu.pt"
    print("Loading", tmp_name)
    policy = PolicyInference(
        model_policy=tmp_name,
        timestep_prediction=0.03*params["trajs_act_stride"],
        cutoff_period=0.5,
        device=device)
    policy.reset()
    with torch.no_grad():
        with h5py.File(params["path_hdf5_demo"], "r") as file_dataset:
            name_demo = random.choice(list(file_dataset.keys()))
            key_images = name_demo + "/images/cam_left_color"
            tmp_shape = file_dataset[key_images].shape
            tmp_eval_length = params["max_goal_dist"]
            idx_begin = random.randint(0, tmp_shape[0]-tmp_eval_length)
            idx_end = idx_begin + tmp_eval_length
            dt = 0.03
            t = 0.0
            #Loop over the evaluation sequence
            list_plot_time = []
            list_plot_obs = []
            list_plot_pred = []
            list_plot_ts = []
            for i in range(idx_begin, idx_end, 1):
                #Retrieve current observation and goal frames from dataset
                dict_obs = OrderedDict()
                dict_obs["images"] = torch.tensor(
                    cv2.imdecode(file_dataset[key_images][i], cv2.IMREAD_UNCHANGED),
                    dtype=torch.uint8,
                    device=device)
                for label in file_dataset[name_demo + "/proprio"]:
                    key_proprio = name_demo + "/proprio/" + label
                    dict_obs[label] = torch.tensor(
                        file_dataset[key_proprio][i], 
                        dtype=torch.float32,
                        device=device)
                dict_goal = OrderedDict()
                dict_goal["images"] = torch.tensor(
                    cv2.imdecode(file_dataset[key_images][idx_end], cv2.IMREAD_UNCHANGED),
                    dtype=torch.uint8,
                    device=device)
                for label in file_dataset[name_demo + "/proprio"]:
                    key_proprio = name_demo + "/proprio/" + label
                    dict_goal[label] = torch.tensor(
                        file_dataset[key_proprio][idx_end], 
                        dtype=torch.float32,
                        device=device)
                #Policy inference
                if i%30 == 0 or t <= 0.0:
                    print("compute inference time {:.2f} index {}".format(t, i))
                    policy.compute_inference_policy(
                        t, True,
                        dict_obs, dict_goal)
                pred_ts, pred_trajs, pred_out = policy.interpolate_output(t, ["cmd_pos", "cmd_quat"])
                print("interpolate time {:.2f} index {} last_inference {:.2f}".format(t, i, t-pred_ts))
                list_plot_time.append(t)
                list_plot_obs.append(dict_obs)
                list_plot_pred.append(pred_trajs)
                list_plot_ts.append(pred_ts)
                t += dt
                time.sleep(dt)
            fig, axs = plt.subplots(2, size_act)
            #for j in range(size_act):
            #    axs[0,j].plot(np.array(list_plot_time)[:], np.array(list_plot_cmd)[:,j], marker=".")
            #    axs[0,j].plot(np.array(list_plot_time)[:], np.array(list_plot_pred)[:,j], marker=".")
            #    axs[0,j].grid()
            axs[1,1].plot(np.array(list_plot_time)[:], np.array(list_plot_ts)[:], marker=".")
            #axs[1,0].grid()
            axs[1,1].grid()
            plt.tight_layout()
            plt.show()
    exit()

