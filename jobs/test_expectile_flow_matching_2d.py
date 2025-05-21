import numpy as np
import torch
import platform
import copy
import matplotlib
import matplotlib.animation as anm
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import job
from utils.training import EMAModel
from utils.training import scheduler_lr_cosine_warmstart_retry
from models.mlp import MLPNet

#Job parameters
params = {
    #Job mode
    "mode": "",
    #Loading previous model
    "is_load": False,
    #Device
    "use_cuda": False,
    #Target distribution
    "type_dist": "roll",
    #Training
    "epoch": 20000,
    "size_minibatch": 1024,
    "learning_rate": 5e-4,
    "weight_decay": 1e-8,
    "lr_scheduler_step_warmup": 200,
    "lr_scheduler_retry": 1,
    "is_clip_grad": True,
    "norm_clip_grad": 1.0,
    "ema_cutoff": 1000,
    "plot_model_every": 2000,
}
params, logger = job.init(params, "Test Extremum Flow Matching on 2d distribution")
print("Parameters:")
for key,value in params.items():
    print("    {0:30} {1}".format(key, value))
print("Logger output path:", logger.get_output_path())
print("Hostname:", platform.node())

#Default plot parameters
plt.rcParams["figure.figsize"] = [14.0, 14.0]

#Device configuration
if params["use_cuda"]:
    if torch.cuda.is_available():
        print(
            "Using CUDA device: id=", torch.cuda.current_device(), 
            "name=", torch.cuda.get_device_name(torch.cuda.current_device()))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise IOError("CUDA not available")
else:
    device = torch.device("cpu")
print("Using device:", device)

#Synthetic distribution
def generate_synthetic_dist(
        type_dist: str, 
        size_sample: int,
    ) -> torch.Tensor:
    """Sample from 2d distribution"""
    with torch.no_grad():
        """
        samples1 = torch.rand((size_sample,1))
        samples2 = torch.tensor([-0.5,0.3]) + 0.1*torch.randn((size_sample, 2))
        afix = torch.rand(size_sample)
        angles = 10.0*afix
        radius = 0.5*afix
        samples3 = torch.tensor([0.6,0.3]) + torch.stack((radius*torch.cos(angles), radius*torch.sin(angles)), dim=1)
            
        samples4_1 = 0.2*(2.0*torch.rand((size_sample, 2)) - 1.0)
        samples4_2 = torch.tensor([0.0,0.2]) + \
            torch.rand((size_sample, 1))*torch.tensor([0.0, 0.4])
        samples4_3 = torch.rand((size_sample,1))
        samples4 = torch.tensor([0.0,-0.7]) + torch.where(samples4_3 > 0.5, samples4_1, samples4_2)
        
        return torch.where(
            samples1 > 0.6, 
            samples2, 
            torch.where(samples1 > 0.3,
                samples3,
                samples4))
        """
        if type_dist == "normal":
            return torch.randn((size_sample, 2))
        elif type_dist == "uniform":
            return 2.0*torch.rand((size_sample, 2)) - 1.0
        elif type_dist == "bimodal":
            samples1 = torch.tensor([-2.0,0.0]) + 0.5*torch.randn((size_sample, 2))
            samples2 = torch.tensor([2.0,0.0]) + 0.5*torch.randn((size_sample, 2))
            samples3 = torch.rand((size_sample,1))
            return torch.where(samples3 > 0.5, samples1, samples2)
        elif type_dist == "roll":
            samples = torch.rand(size_sample)
            angles = 10.0*samples
            radius = 1.0*samples
            return torch.stack(
                (radius*torch.cos(angles), radius*torch.sin(angles)),
                dim=1)
        elif type_dist == "circle":
            samples = torch.rand(size_sample)
            angles = 2.0*3.1415*samples
            radius = 2.0 + 0.1*torch.randn(size_sample)
            return torch.stack(
                (radius*torch.cos(angles), radius*torch.sin(angles)),
                dim=1)
        elif type_dist == "square_line":
            samples1 = 0.4*(2.0*torch.rand((size_sample, 2)) - 1.0)
            samples2 = torch.tensor([0.0,0.4]) + \
                torch.rand((size_sample, 1))*torch.tensor([0.0, 0.6])
            samples3 = torch.rand((size_sample,1))
            return torch.where(samples3 > 0.5, samples1, samples2)
        elif type_dist == "zero":
            return torch.zeros((size_sample, 2))
        else:
            raise IOError("Invalid distribution")

#Plot target distribution
if params["mode"] == "plots":
    tmp_dst = generate_synthetic_dist(params["type_dist"], 1000)
    fig, axs = plt.subplots(1, 1)
    axs.scatter(tmp_dst[:,0], tmp_dst[:,1], color="blue", alpha=0.2)
    axs.grid()
    axs.axis("equal")
    plt.show()
    exit()

#Define models
model_1 = MLPNet(
    dim_in=1+1,
    dim_hidden=[1024, 1024, 1024],
    dim_out=1,
    activation=torch.nn.SiLU, spectral_norm_on_hidden=False).to(device)
model_2 = MLPNet(
    dim_in=1+1+1,
    dim_hidden=[1024, 1024, 1024],
    dim_out=1,
    activation=torch.nn.SiLU, spectral_norm_on_hidden=False).to(device)
model_3 = MLPNet(
    dim_in=2+1,
    dim_hidden=[1024, 1024, 1024],
    dim_out=2,
    activation=torch.nn.SiLU, spectral_norm_on_hidden=False).to(device)
if params["is_load"]:
    print("Load models")
    model_1.load_state_dict(torch.load("/tmp/model_1.params", weights_only=True))
    model_2.load_state_dict(torch.load("/tmp/model_2.params", weights_only=True))
    model_3.load_state_dict(torch.load("/tmp/model_3.params", weights_only=True))

@torch.no_grad()
def do_plot_model(model_1, model_2):
    tmp_dst = generate_synthetic_dist(params["type_dist"], 1000)
    fig, axs = plt.subplots(1, 1)
    axs.scatter(tmp_dst[:,0], tmp_dst[:,1], color="blue", alpha=0.2)

    with torch.no_grad():
        size_batch = 5000
        
        tmp_src1 = torch.rand(size_batch,1)
        tmp_pred1 = tmp_src1.clone()
        for t in np.arange(0.0, 1.0, 0.05):
            delta_1 = model_1(tmp_pred1, t*torch.ones(size_batch,1))
            tmp_pred1 += 0.05*delta_1

        tmp_src2 = torch.rand(size_batch,1)
        tmp_pred2 = tmp_src2.clone()
        for t in np.arange(0.0, 1.0, 0.05):
            delta_2 = model_2(tmp_pred2, tmp_pred1, t*torch.ones(size_batch,1))
            tmp_pred2 += 0.05*delta_2
    axs.scatter(tmp_pred1, tmp_pred2, c=tmp_src1, alpha=0.2)
    
    with torch.no_grad():
        size_batch = 500
        
        tmp_src1 = 0.0*torch.ones(size_batch,1)
        tmp_pred1 = tmp_src1.clone()
        for t in np.arange(0.0, 1.0, 0.05):
            delta_1 = model_1(tmp_pred1, t*torch.ones(size_batch,1))
            tmp_pred1 += 0.05*delta_1

        tmp_src2 = torch.rand(size_batch,1)
        tmp_pred2 = tmp_src2.clone()
        for t in np.arange(0.0, 1.0, 0.05):
            delta_2 = model_2(tmp_pred2, tmp_pred1, t*torch.ones(size_batch,1))
            tmp_pred2 += 0.05*delta_2
    axs.scatter(tmp_pred1, tmp_pred2, c="red", alpha=0.2)
    
    axs.grid()
    axs.axis("equal")
    return fig, axs

#Plot model
if params["mode"] == "plot_model":
    fig, axs = do_plot_model(model_1, model_2) 
    plt.show()
    exit()

#Anim inference
if params["mode"] == "anim_model":
    with torch.no_grad():
        tmp_dst = generate_synthetic_dist(params["type_dist"], 1000)
        fig, axs = plt.subplots(1, 1)
        axs.scatter(tmp_dst[:,0], tmp_dst[:,1], color="black", alpha=0.2)

        with torch.no_grad():
            size_batch = 8000
            
            tmp_src1 = torch.rand(size_batch,1)
            tmp_pred1 = tmp_src1.clone()
            chain1 = []
            chain1.append(tmp_pred1.clone())
            for t in np.arange(0.0, 1.0, 0.01):
                delta_1 = model_1(tmp_pred1, t*torch.ones(size_batch,1))
                tmp_pred1 += 0.01*delta_1
                chain1.append(tmp_pred1.clone())

            tmp_src2 = torch.rand(size_batch,1)
            tmp_pred2 = tmp_src2.clone()
            chain1[0] = torch.cat([chain1[0], tmp_pred2.clone()], dim=1)
            for i,t in enumerate(np.arange(0.0, 1.0, 0.01)):
                delta_2 = model_2(tmp_pred2, tmp_pred1, t*torch.ones(size_batch,1))
                tmp_pred2 += 0.01*delta_2
                chain1[i+1] = torch.cat([chain1[i+1], tmp_pred2.clone()], dim=1)
        scat1 = axs.scatter(tmp_pred1, tmp_pred2, c=tmp_src1, alpha=0.6)
        
        with torch.no_grad():
            size_batch = 500
            
            tmp_src1 = 0.0*torch.ones(size_batch,1)
            tmp_pred1 = tmp_src1.clone()
            chain2 = []
            chain2.append(tmp_pred1.clone())
            for t in np.arange(0.0, 1.0, 0.01):
                delta_1 = model_1(tmp_pred1, t*torch.ones(size_batch,1))
                tmp_pred1 += 0.01*delta_1
                chain2.append(tmp_pred1.clone())

            tmp_src2 = torch.rand(size_batch,1)
            tmp_pred2 = tmp_src2.clone()
            chain2[0] = torch.cat([chain2[0], tmp_pred2.clone()], dim=1)
            for i,t in enumerate(np.arange(0.0, 1.0, 0.01)):
                delta_2 = model_2(tmp_pred2, tmp_pred1, t*torch.ones(size_batch,1))
                tmp_pred2 += 0.01*delta_2
                chain2[i+1] = torch.cat([chain2[i+1], tmp_pred2.clone()], dim=1)
        scat2 = axs.scatter(tmp_pred1, tmp_pred2, c="red", alpha=1.0)
        
        with torch.no_grad():
            size_batch = 500
            
            tmp_src1 = 1.0*torch.ones(size_batch,1)
            tmp_pred1 = tmp_src1.clone()
            chain3 = []
            chain3.append(tmp_pred1.clone())
            for t in np.arange(0.0, 1.0, 0.01):
                delta_1 = model_1(tmp_pred1, t*torch.ones(size_batch,1))
                tmp_pred1 += 0.01*delta_1
                chain3.append(tmp_pred1.clone())

            tmp_src2 = torch.rand(size_batch,1)
            tmp_pred2 = tmp_src2.clone()
            chain3[0] = torch.cat([chain3[0], tmp_pred2.clone()], dim=1)
            for i,t in enumerate(np.arange(0.0, 1.0, 0.01)):
                delta_2 = model_2(tmp_pred2, tmp_pred1, t*torch.ones(size_batch,1))
                tmp_pred2 += 0.01*delta_2
                chain3[i+1] = torch.cat([chain3[i+1], tmp_pred2.clone()], dim=1)
        scat3 = axs.scatter(tmp_pred1, tmp_pred2, c="red", alpha=1.0)
        
        #axs.grid()
        axs.axis("equal")
        axs.set_xlim(-1.0, 1.2)
        axs.set_ylim(-1.0, 1.2)
        axs.spines[["right", "top", "bottom", "left"]].set_visible(False)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_title("Conditional Model", fontsize=38)
        axs.set_ylabel("Y", fontsize=35)
        axs.set_xlabel("X", fontsize=35)
        plt.tight_layout()
        def animate(i):
            if i < 20:
                scat1.set_offsets(chain1[0])
                scat2.set_offsets(chain2[0])
                scat3.set_offsets(chain3[0])
            elif i < 20+100:
                scat1.set_offsets(chain1[i-20])
                scat2.set_offsets(chain2[i-20])
                scat3.set_offsets(chain3[i-20])
            else:
                scat1.set_offsets(chain1[-1])
                scat2.set_offsets(chain2[-1])
                scat3.set_offsets(chain3[-1])
            return scat1, scat2, scat3
        anim = anm.FuncAnimation(
            fig, animate, frames=20+100+20, interval=20, blit=False)
        anim.save("/tmp/animation_solution.mp4", writer="ffmpeg")
        np.savez("/tmp/data_solution.npz", 
            point_begin_1=chain1[0],
            point_begin_2=chain2[0],
            point_begin_3=chain3[0],
            point_end_1=chain1[-1],
            point_end_2=chain2[-1],
            point_end_3=chain3[-1],
        )
        plt.show()
        exit()

#Anim inference
if params["mode"] == "anim_baseline":
    with torch.no_grad():
        tmp_dst = generate_synthetic_dist(params["type_dist"], 1000)
        fig, axs = plt.subplots(1, 1)
        axs.scatter(tmp_dst[:,0], tmp_dst[:,1], color="black", alpha=0.2)

        with torch.no_grad():
            size_batch = 8000
            
            tmp_src1 = torch.rand(size_batch,2)
            tmp_pred1 = tmp_src1.clone()
            chain1 = []
            chain1.append(tmp_pred1.clone())
            for t in np.arange(0.0, 1.0, 0.01):
                delta_1 = model_3(tmp_pred1, t*torch.ones(size_batch,1))
                tmp_pred1 += 0.01*delta_1
                chain1.append(tmp_pred1.clone())
        scat1 = axs.scatter(tmp_pred1[:,0], tmp_pred1[:,1], c=tmp_src1[:,0], alpha=0.6)
        
        with torch.no_grad():
            size_batch = 500
            
            tmp_src1 = torch.rand(size_batch,2)
            tmp_src1[:,0] = 0.0
            tmp_pred1 = tmp_src1.clone()
            chain2 = []
            chain2.append(tmp_pred1.clone())
            for t in np.arange(0.0, 1.0, 0.01):
                delta_1 = model_3(tmp_pred1, t*torch.ones(size_batch,1))
                tmp_pred1 += 0.01*delta_1
                chain2.append(tmp_pred1.clone())
        scat2 = axs.scatter(tmp_pred1[:,0], tmp_pred1[:,1], c="red", alpha=1.0)
        
        with torch.no_grad():
            size_batch = 500
            
            tmp_src1 = torch.rand(size_batch,2)
            tmp_src1[:,0] = 1.0
            tmp_pred1 = tmp_src1.clone()
            chain3 = []
            chain3.append(tmp_pred1.clone())
            for t in np.arange(0.0, 1.0, 0.01):
                delta_1 = model_3(tmp_pred1, t*torch.ones(size_batch,1))
                tmp_pred1 += 0.01*delta_1
                chain3.append(tmp_pred1.clone())
        scat3 = axs.scatter(tmp_pred1[:,0], tmp_pred1[:,1], c="red", alpha=1.0)
        
        #axs.grid()
        axs.axis("equal")
        axs.set_xlim(-1.0, 1.2)
        axs.set_ylim(-1.0, 1.2)
        axs.spines[["right", "top", "bottom", "left"]].set_visible(False)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_title("Baseline Model", fontsize=38)
        axs.set_ylabel("Y", fontsize=35)
        axs.set_xlabel("X", fontsize=35)
        plt.tight_layout()
        def animate(i):
            if i < 20:
                scat1.set_offsets(chain1[0])
                scat2.set_offsets(chain2[0])
                scat3.set_offsets(chain3[0])
            elif i < 20+100:
                scat1.set_offsets(chain1[i-20])
                scat2.set_offsets(chain2[i-20])
                scat3.set_offsets(chain3[i-20])
            else:
                scat1.set_offsets(chain1[-1])
                scat2.set_offsets(chain2[-1])
                scat3.set_offsets(chain3[-1])
            return scat1, scat2, scat3
        anim = anm.FuncAnimation(
            fig, animate, frames=20+100+20, interval=20, blit=False)
        anim.save("/tmp/animation_baseline.mp4", writer="ffmpeg")
        np.savez("/tmp/data_baseline.npz", 
            point_begin_1=chain1[0],
            point_begin_2=chain2[0],
            point_begin_3=chain3[0],
            point_end_1=chain1[-1],
            point_end_2=chain2[-1],
            point_end_3=chain3[-1],
        )
        plt.show()
        exit()

#Training
if params["mode"] == "train":
    optimizer = torch.optim.AdamW(
        list(model_1.parameters()) + list(model_2.parameters()) + list(model_3.parameters()), 
        betas=(0.9, 0.999),
        weight_decay=params["weight_decay"],
        lr=params["learning_rate"])
    ema_1 = EMAModel(
        cutoff_period=params["ema_cutoff"], 
        warmup_steps=params["lr_scheduler_step_warmup"])
    ema_2 = EMAModel(
        cutoff_period=params["ema_cutoff"], 
        warmup_steps=params["lr_scheduler_step_warmup"])
    ema_3 = EMAModel(
        cutoff_period=params["ema_cutoff"], 
        warmup_steps=params["lr_scheduler_step_warmup"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda epoch: scheduler_lr_cosine_warmstart_retry(
            epoch, params["epoch"], 
            params["lr_scheduler_step_warmup"], 
            params["lr_scheduler_retry"]))
    for epoch in range(params["epoch"]):
        size_batch = params["size_minibatch"]
        tmp_src1 = torch.rand(size_batch,1,device=device)
        tmp_src2 = torch.rand(size_batch,1,device=device)
        tmp_src3 = torch.rand(size_batch,2,device=device)
        tmp_dst = generate_synthetic_dist(params["type_dist"], size_batch).to(device)
        
        tmp_t = torch.rand(size_batch, 1, device=device)
        tmp_middle1 = (1.0-tmp_t)*tmp_src1 + tmp_t*tmp_dst[:,0:1]
        tmp_middle2 = (1.0-tmp_t)*tmp_src2 + tmp_t*tmp_dst[:,1:2]
        tmp_middle3 = (1.0-tmp_t)*tmp_src3 + tmp_t*tmp_dst
        
        delta_1 = model_1(tmp_middle1, tmp_t)
        delta_2 = model_2(tmp_middle2, tmp_dst[:,0:1], tmp_t)
        delta_3 = model_3(tmp_middle3, tmp_t)
        loss = (delta_1 - (tmp_dst[:,0:1]-tmp_src1)).pow(2).mean() + (delta_2 - (tmp_dst[:,1:2]-tmp_src2)).pow(2).mean() + (delta_3 - (tmp_dst-tmp_src3)).pow(2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_1.update(model_1)
        ema_2.update(model_2)
        ema_3.update(model_3)
        scheduler.step()
        if epoch%100 == 0:
            print(epoch, loss.detach().item())
        if epoch%params["plot_model_every"] == 0:
            fig, axs = do_plot_model(
                copy.deepcopy(ema_1.getModel()).cpu(),
                copy.deepcopy(ema_2.getModel()).cpu())
            logger.add_frame_as_fig("train", fig, epoch)
            logger.add_model("model_1", ema_1.getModel(), epoch)
            logger.add_model("model_2", ema_2.getModel(), epoch)
            logger.add_model("model_3", ema_3.getModel(), epoch)
            torch.save(ema_1.getModel().state_dict(), "/tmp/model_1.params")
            torch.save(ema_2.getModel().state_dict(), "/tmp/model_2.params")
            torch.save(ema_3.getModel().state_dict(), "/tmp/model_3.params")
    logger.set_parameters(params)
    logger.close()
    exit()

