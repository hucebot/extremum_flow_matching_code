import os
import time
import copy
import random
import numpy as np
import torch
import platform
import matplotlib
import matplotlib.animation as anm
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import job
from gcrl.agent import GCRLAgentBase, GCRLAgent0, GCRLAgent1, GCRLAgent2, GCRLAgent5, GCRLAgent6
from gcrl.dataset import GCTrajsDataset

from typing import List, Tuple, Dict

#os.environ["MUJOCO_GL"] = "glfw"
import ogbench
import mujoco

#Job parameters
params = {
    #Job mode
    "mode": "",
    #Dataset processing
    "dataset_name": "cube-single-play-v0",
    #Device
    "use_cuda": False,
    #Loading previous agents
    "is_load": False,
    #Trajectories
    "trajs_obs_len": 4,
    "trajs_act_len": 8,
    "trajs_obs_stride": 16,
    "trajs_act_stride": 1,
    "max_goal_dist": 200,
    #Agents
    "randomize_config": False,
    "encoder_proprio_name": "identity",
    "encoder_proprio_size": None,
    #Training
    "epoch": 200000,
    "ema_cutoff": 5000,
    "size_batch": 128,
    "save_model_every": 100000,
    "eval_model_every": 100000,
    #Inference
    "inference_horizon": 7,
    "num_eval_episode": 20,
}
params, logger = job.init(params, "Run OGBench on Extremum Flow Matching Agents")
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

#Load dataset and environment
print("Load dataset:", params["dataset_name"])
env, data_train_dataset, data_eval_dataset = ogbench.make_env_and_datasets(
    params["dataset_name"], 
    dataset_dir='data_static/',
    compact_dataset=True,
)
obs, info = env.reset()
print("Dataset Train:")
for n,v in data_train_dataset.items():
    print("    ", n, v.shape)
print("Dataset Evaluation:")
for n,v in data_eval_dataset.items():
    print("    ", n, v.shape)
size_obs = data_train_dataset["observations"].shape[1]
size_act = data_train_dataset["actions"].shape[1]
params["encoder_proprio_size"] = size_obs
print("size_obs:", size_obs, "size_act:", size_act)

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
    params["encoder_proprio_size"] = random.choice([size_obs, 16, 32, 64])
    params["model_encoder_mlp_hidden"] = random.choice([[32], [64], [128]])
    params["model_flow_unet_hidden"] = random.choice([
        [32*1, 64*1, 128*1],
        [32*2, 64*2, 128*2],
        [32*3, 64*3, 128*3],
        [32*4, 64*4, 128*4],
        ])
    params["model_flow_mlp_hidden"] = random.choice([
        [256, 256, 256],
        [512, 512, 512],
        [1024, 1024, 1024],
        ])
    params["weight_decay"] = random.choice([1e-2, 1e-3, 1e-6])
    params["learning_rate"] = random.choice([2e-4, 1e-4, 5e-5, 1e-5])
    params["critic_ratio_training"] = random.choice([0.1, 0.2, 0.5, 0.7])
    params["trajs_obs_len"] = random.choice([2, 4, 8])
    params["trajs_act_len"] = 8
    params["trajs_obs_stride"] = random.choice([2, 4, 8, 16])
    params["trajs_act_stride"] = 1
    params["max_goal_dist"] = random.choice([50, 75, 100, 150, 200, 250])
    params["ema_cutoff"] = random.choice([500, 1000, 2000, 5000])
    if params["trajs_obs_len"] < 8:
        params["model_planner_name"] = "mlp"
    if params["encoder_proprio_name"] == "identity":
        params["encoder_proprio_size"] = size_obs
#Verbose configuration
print("Agents config:")
for key,value in params.items():
    print("    {0:40} {1}".format(key, value))

#Create Goal Conditioned Dataset for trajectories
dataset_train = GCTrajsDataset(
    dict_obs={
        "proprio": data_train_dataset["observations"],
    },
    array_act=data_train_dataset["actions"],
    array_terminal=(1-data_train_dataset["valids"]),
    trajs_obs_len=params["trajs_obs_len"],
    trajs_act_len=params["trajs_act_len"],
    trajs_obs_stride=params["trajs_obs_stride"],
    trajs_act_stride=params["trajs_act_stride"],
    max_dist_goal=params["max_goal_dist"],
    is_act_vel=True,
    device=device,
)

#Verbose
tmp_dist_train = []
for i in range(dataset_train.count_episodes()):
    tmp_dist_train.append(dataset_train.get_episode(i)[1].size(0))
print("Dataset GCTrajs Train:")
print("    Count episodes:", dataset_train.count_episodes())
print("    Length episodes:", np.array(tmp_dist_train).mean())

def do_set_obs_state_to_env(
        env, 
        obs: np.ndarray,
    ) -> None:
    """Assign state to given environment from observation data (size_obs)"""
    if env.spec.id.startswith("antmaze-"):
        env.unwrapped.set_state(obs[0:15], obs[15:15+14])
    if env.spec.id.startswith("pointmaze-"):
        env.unwrapped.set_xy(obs[0:2])
    if env.spec.id.startswith("cube-single-"):
        env.unwrapped._data.qpos[env.unwrapped._arm_joint_ids] = obs[0:6].clone()
        env.unwrapped._data.qvel[env.unwrapped._arm_joint_ids] = obs[6:6+6].clone()
        env.unwrapped._data.qpos[env.unwrapped._gripper_opening_joint_id] = obs[6+6+3+2].clone()*0.8/3.0
        env.unwrapped._data.joint("object_joint_0").qpos[0:3] = obs[6+6+3+2+1+1:6+6+3+2+1+1+3].clone()/10.0+np.array([0.425, 0.0, 0.0])
        env.unwrapped._data.joint("object_joint_0").qpos[3:3+4] = obs[6+6+3+2+1+1+3:6+6+3+2+1+1+3+4].clone()
        mujoco.mj_forward(env.unwrapped._model, env.unwrapped._data)
    if env.spec.id.startswith("scene-"):
        env.unwrapped._data.qpos[env.unwrapped._arm_joint_ids] = obs[0:6].clone()
        env.unwrapped._data.qvel[env.unwrapped._arm_joint_ids] = obs[6:6+6].clone()
        env.unwrapped._data.qpos[env.unwrapped._gripper_opening_joint_id] = obs[6+6+3+2].clone()*0.8/3.0
        tmp_offset = 6+6+3+2+1+1
        env.unwrapped._data.joint("object_joint_0").qpos[0:3] = obs[tmp_offset:tmp_offset+3].clone()/10.0+np.array([0.425, 0.0, 0.0])
        env.unwrapped._data.joint("object_joint_0").qpos[3:3+4] = obs[tmp_offset+3:tmp_offset+3+4].clone()
        tmp_offset += 3+4+2
        env.unwrapped._cur_button_states[0] = (obs[tmp_offset].clone() > 0)
        env.unwrapped._data.joint("buttonbox_joint_0").qpos[0] = obs[tmp_offset+2].clone()/120.0
        env.unwrapped._data.joint("buttonbox_joint_0").qvel[0] = obs[tmp_offset+3].clone()
        tmp_offset += 4
        env.unwrapped._cur_button_states[1] = (obs[tmp_offset+1].clone() > 0)
        env.unwrapped._data.joint("buttonbox_joint_1").qpos[0] = obs[tmp_offset+2].clone()/120.0
        env.unwrapped._data.joint("buttonbox_joint_1").qvel[0] = obs[tmp_offset+3].clone()
        tmp_offset += 4
        env.unwrapped._data.joint("drawer_slide").qpos[0] = obs[tmp_offset+0].clone()/18.0
        env.unwrapped._data.joint("drawer_slide").qvel[0] = obs[tmp_offset+1].clone()
        env.unwrapped._data.joint("window_slide").qpos[0] = obs[tmp_offset+2].clone()/15.0
        env.unwrapped._data.joint("window_slide").qvel[0] = obs[tmp_offset+3].clone()
        mujoco.mj_forward(env.unwrapped._model, env.unwrapped._data)
        env.unwrapped._apply_button_states()

def do_render_observation(
        env, 
        obs: np.ndarray,
    ) -> np.ndarray:
    """Render with given environment given observation and return a RGB frame (width,height,3)"""
    do_set_obs_state_to_env(env, obs)
    return env.render()

def do_plot_anim_frames(
        list_frames: List[np.ndarray],
    ):
    """Display video animation from given frames (width,height,3) list with matplotlib"""
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(12, 10)
    im = axs.imshow(list_frames[0])
    title = axs.text(0.0,0.9, "", 
        fontsize=12, va="top", bbox=dict(facecolor="white", edgecolor="black", alpha=0.6))
    fig.tight_layout()
    def animate(k):
        title.set_text("Frame: {}/{}".format(str(k), len(list_frames)))
        im.set_array(list_frames[k])
        return im, title
    anim = anm.FuncAnimation(fig, animate, frames=len(list_frames), interval=10, blit=True)
    plt.show()

#Plot dataset episodes
if params["mode"] == "plots":
    #Plot observations
    fig, axs = plt.subplots(size_obs//4+1, 4)
    for i in range(3):
        tmp_episode_obs, tmp_episode_act = dataset_train.get_episode(i)
        for k in range(size_obs):
            axs.flatten()[k].plot(tmp_episode_obs["proprio"][:,k])
            if i == 0:
                axs.flatten()[k].grid()
                axs.flatten()[k].set_title("obs_"+str(k))
    fig.tight_layout()
    plt.show()
    #Plot actions
    fig, axs = plt.subplots(size_act//4+1, 4)
    for i in range(3):
        tmp_episode_obs, tmp_episode_act = dataset_train.get_episode(i)
        for k in range(tmp_episode_act.shape[1]):
            axs.flatten()[k].plot(tmp_episode_act[:,k])
            if i == 0:
                axs.flatten()[k].grid()
                axs.flatten()[k].set_title("act_"+str(k))
    fig.tight_layout()
    plt.show()
    #Plot observations and actions distributions
    batch_obs, batch_goal, batch_dist, batch_traj_obs, batch_traj_act = dataset_train.sample(10000)
    fig, axs = plt.subplots(1, 3, width_ratios=[10, 5, 1])
    for i in range(size_obs):
        axs[0].violinplot(batch_obs["proprio"][:,i], positions=[i], widths=[1.0], showmeans=True, showmedians=True)
        axs[0].boxplot(batch_obs["proprio"][:,i], positions=[i], showfliers=False)
    for i in range(size_act):
        axs[1].violinplot(batch_traj_act[:,0,i], positions=[i], widths=[1.0], showmeans=True, showmedians=True)
        axs[1].boxplot(batch_traj_act[:,0,i], positions=[i], showfliers=False)
    axs[2].violinplot(batch_dist, showmeans=True, showmedians=True)
    axs[0].set_title("observations")
    axs[1].set_title("actions")
    axs[2].set_title("lengths")
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    fig.tight_layout()
    plt.show()
    #Plot sample action trajectories
    batch_obs, batch_goal, batch_dist, batch_traj_obs, batch_traj_act = dataset_train.sample(20)
    fig, axs = plt.subplots(int(np.ceil(size_act/3)), 3)
    for i in range(20):
        for j in range(size_act):
            axs.flatten()[j].plot(batch_traj_act[i,:,j], alpha=0.4, marker="o")
            if i == 0:
                axs.flatten()[j].grid()
                axs.flatten()[j].set_title("act_"+str(j))
    fig.tight_layout()
    plt.show()
    #Plot sample proprioception observation trajectories
    batch_obs, batch_goal, batch_dist, batch_traj_obs, batch_traj_act = dataset_train.sample(20)
    fig, axs = plt.subplots(int(np.ceil(size_obs/6)), 6)
    for i in range(20):
        for j in range(size_obs):
            axs.flatten()[j].plot(batch_traj_obs["proprio"][i,:,j], alpha=0.4, marker="o")
            if i == 0:
                axs.flatten()[j].grid()
                axs.flatten()[j].set_title("obs_"+str(j))
    fig.tight_layout()
    plt.show()
    #Show animated observations
    for i in range(3):
        print("Episode:", i)
        tmp_episode_obs, tmp_episode_act = dataset_train.get_episode(i)
        list_frames = []
        for j in range(tmp_episode_obs["proprio"].shape[0]):
            list_frames.append(do_render_observation(env, tmp_episode_obs["proprio"][j]))
        do_plot_anim_frames(list_frames)
    exit()

#Initialize agents
agents = {
    "agent0": GCRLAgent0(
        config=params, 
        encoder_config=[
            ("proprio", params["encoder_proprio_name"], 
                size_obs, params["encoder_proprio_size"]),
        ],
        size_act=size_act,
        device=device,
    ),
    "agent1_norl": GCRLAgent1(
        config=params, 
        encoder_config=[
            ("proprio", params["encoder_proprio_name"], 
                size_obs, params["encoder_proprio_size"]),
        ],
        size_act=size_act,
        use_rl=False,
        use_merge_traj=False,
        device=device,
    ),
    "agent1_withrl": GCRLAgent1(
        config=params, 
        encoder_config=[
            ("proprio", params["encoder_proprio_name"], 
                size_obs, params["encoder_proprio_size"]),
        ],
        size_act=size_act,
        use_rl=True,
        use_merge_traj=False,
        device=device,
    ),
    "agent2_norl": GCRLAgent2(
        config=params, 
        encoder_config=[
            ("proprio", params["encoder_proprio_name"], 
                size_obs, params["encoder_proprio_size"]),
        ],
        size_act=size_act,
        use_rl=False,
        device=device,
    ),
    "agent2_withrl": GCRLAgent2(
        config=params, 
        encoder_config=[
            ("proprio", params["encoder_proprio_name"], 
                size_obs, params["encoder_proprio_size"]),
        ],
        size_act=size_act,
        use_rl=True,
        device=device,
    ),
    "agent5_norl": GCRLAgent5(
        config=params, 
        encoder_config=[
            ("proprio", params["encoder_proprio_name"], 
                size_obs, params["encoder_proprio_size"]),
        ],
        size_act=size_act,
        use_rl=False,
        device=device,
    ),
    "agent5_withrl": GCRLAgent5(
        config=params, 
        encoder_config=[
            ("proprio", params["encoder_proprio_name"], 
                size_obs, params["encoder_proprio_size"]),
        ],
        size_act=size_act,
        use_rl=True,
        device=device,
    ),
    "agent6_norl": GCRLAgent6(
        config=params, 
        encoder_config=[
            ("proprio", params["encoder_proprio_name"], 
                size_obs, params["encoder_proprio_size"]),
        ],
        size_act=size_act,
        use_rl=False,
        device=device,
    ),
    "agent6_withrl": GCRLAgent6(
        config=params, 
        encoder_config=[
            ("proprio", params["encoder_proprio_name"], 
                size_obs, params["encoder_proprio_size"]),
        ],
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
def do_eval_single(
        env, 
        task_id: int, 
        agent,
        is_render: bool = False,
    ):
    """Simulate a single evaluation run for given task_id"""
    list_trajs_act = []
    list_info = []
    list_frames = []
    #Reset environment
    obs, info = env.reset(options=dict(task_id=task_id, render_goal=is_render))
    goal = info["goal"]
    frame_goal = None
    if is_render:
        frame_goal = info["goal_rendered"]
    done = False
    #Simulation loop
    while not done:
        #Evaluate the policy
        tmp_trajs_act = agent.inference(
            {
                "proprio": torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0),
            }, 
            {
                "proprio": torch.tensor(goal, device=device, dtype=torch.float32).unsqueeze(0),
            },
        )
        list_trajs_act.append(tmp_trajs_act.cpu().clone())
        list_info.append(copy.deepcopy(agent._info))
        for k in range(params["inference_horizon"]):
            obs, reward, terminated, truncated, info = env.step(tmp_trajs_act[0,k,:].cpu().numpy())
            if is_render:
                list_frames.append(env.render())
            done = terminated or truncated
            if done:
                break
    is_success = info["success"]
    return is_success, list_trajs_act, list_frames, frame_goal, list_info

@torch.no_grad()
def do_plot_eval_single(
        env, 
        task_id: int, 
        agent,
    ):
    is_success, list_trajs_act, list_frames, frame_goal, list_info = do_eval_single(
        env, 1, agent, is_render=True)
    print("Episode length:", len(list_frames))
    print("is_success:", is_success)
    #Animate episode
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(24, 14)
    axs[1,0].imshow(list_frames[0])
    axs[1,1].imshow(frame_goal)
    im = axs[0,0].imshow(list_frames[0])
    lines = []
    for i in range(size_act):
        line, = axs[1,2].plot(list_trajs_act[0][0,:,i], marker=".")
        lines.append(line)
    axs[0,2].grid()
    axs[1,2].set_ylim(-1.0, 1.0)
    axs[1,2].grid()
    axs[0,0].set_title("Episode")
    axs[1,0].set_title("Init")
    axs[1,1].set_title("Goal is_success="+str(is_success))
    axs[0,2].set_title("Length")
    title = axs[0,0].text(0.0,0.9, "", 
        fontsize=12, va="top", bbox=dict(facecolor="white", edgecolor="black", alpha=0.6))
    fig.tight_layout()
    def animate(k):
        title.set_text("Frame: {}/{}".format(str(k), len(list_frames)))
        im.set_array(list_frames[k])
        tmp_idx = k//params["inference_horizon"]
        for i in range(size_act):
            lines[i].set_ydata(list_trajs_act[tmp_idx][0,:,i])
        lll = [im, title]
        for i in range(size_act):
            lll.append(lines[i])
        return tuple(lll)
    anim = anm.FuncAnimation(fig, animate, frames=len(list_frames), interval=10, blit=True)
    #anim.save("/tmp/anim.mp4", writer="ffmpeg")
    plt.show()

@torch.no_grad()
def do_eval_stats(
        env, 
        agent,
        num_eval_episode: int,
    ):
    list_is_success = []
    list_length = []
    for k in range(1, 6):
        for i in range(num_eval_episode):
            is_success, list_trajs, list_frames, frame_goal, list_info = do_eval_single(
                env, k, agent, is_render=False)
            list_is_success.append(float(is_success))
            tmp_length = len(list_trajs)*params["inference_horizon"]
            if is_success:
                list_length.append(tmp_length)
            print("task_id", k, "run", i, "len", tmp_length, "is_success", is_success)
    overall_success = np.array(list_is_success).mean()
    success_length = (np.array(list_length).mean() if len(list_length) > 0 else 0.0)
    print("overall_success:", overall_success, "success_length:", success_length)
    return overall_success, success_length

#Run single evaluation simulation
if params["mode"] == "eval_single":
    for label in agents.keys():
        print("==== Eval", label)
        do_plot_eval_single(env, 1, agents[label])
    exit()

#Run statistics evaluation
if params["mode"] == "eval_stats":
    for label in agents.keys():
        print("==== Eval", label)
        do_eval_stats(env, agents[label], params["num_eval_episode"])
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
            #Eval agents
            if epoch > 0 and epoch%params["eval_model_every"] == 0:
                for label in agents.keys():
                    overall_success, success_length = do_eval_stats(
                        env, agents[label], params["num_eval_episode"])
                    logger.add_scalar("eval_success/"+label, overall_success, epoch)
                    logger.add_scalar("eval_length/"+label, success_length, epoch)
    #Finalize logger
    logger.set_parameters(params)
    logger.close()
    exit()

