import numpy as np
import torch
import platform
import matplotlib
import matplotlib.animation as anm
import matplotlib.pyplot as plt
from scipy import stats

from utils import job
from utils.training import EMAModel
from utils.training import scheduler_lr_cosine_warmstart_retry
from utils.flow_transport import FlowTransport
from models.mlp import MLPNet
from models.unet import ModelDenseSimple

#Job parameters
params = {
    #Job mode
    "mode": "",
    #Loading previous model
    "is_load": False,
    #Device
    "use_cuda": True,
    #Training
    "epoch": 20000,
    "size_minibatch": 1024,
    "learning_rate": 2e-4,
    "weight_decay": 1e-8,
    "lr_scheduler_step_warmup": 100,
    "lr_scheduler_retry": 1,
    "ema_cutoff": 100,
    #Inference
    "forward_step": 20,
}
params, logger = job.init(params, "Compare Expectile Regression and Extremum Flow Matching")
print("Parameters:")
for key,value in params.items():
    print("    {0:30} {1}".format(key, value))
print("Logger output path:", logger.get_output_path())
print("Hostname:", platform.node())

#Default plot parameters
plt.rcParams["figure.figsize"] = [25.0, 14.0]

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

class MixtureModel(stats.rv_continuous):
    def __init__(self, submodels, *args, weights = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise(ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]
    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x)  * weight
        return pdf
    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x)  * weight
        return sf
    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x)  * weight
        return cdf
    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p = self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs

def func_distribution(x):
    return MixtureModel([
        stats.norm(2.0*x+0.3*np.sin(7.0*x), 0.35),
        stats.norm(-2.0*x+0.3*np.sin(7.0*x), 0.35),
        ],
        weights = [1.0, 1.0],
    )

#Generate dataset
array_x = np.linspace(-1.0, 1.0, 5000)
array_y = []
for x in array_x:
    array_y.append(func_distribution(x).rvs(1)[0])
array_y = np.array(array_y)

#Plot dataset 
if True:
    fig, axs = plt.subplots(1, 1)
    axs.scatter(array_x, array_y, color="blue", alpha=0.3)
    axs.grid()
    plt.show()

#Define models
model_1 = MLPNet(
    dim_in=1,
    dim_hidden=[128, 256, 128],
    dim_out=1,
    activation=torch.nn.SiLU, 
    spectral_norm_on_hidden=False,
    norm_layer=False,
).to(device)
model_2 = MLPNet(
    dim_in=1,
    dim_hidden=[128, 256, 128],
    dim_out=1,
    activation=torch.nn.SiLU, 
    spectral_norm_on_hidden=False,
    norm_layer=False,
).to(device)
model_flow = ModelDenseSimple(
    size_channel=1,
    size_length=1,
    size_cond=1,
    size_hidden_list=[1024, 1024, 1024],
).to(device)
model_1.print_parameters()
model_2.print_parameters()
model_flow.print_parameters()

#Training
tensor_x = torch.tensor(array_x, dtype=torch.float32, device=device)
tensor_y = torch.tensor(array_y, dtype=torch.float32, device=device)
flow = FlowTransport(type_transport_func="linear")
ema_flow = EMAModel(
    cutoff_period=params["ema_cutoff"], 
    warmup_steps=params["lr_scheduler_step_warmup"])
optimizer = torch.optim.AdamW(
    list(model_1.parameters()) + list(model_2.parameters()) + list(model_flow.parameters()), 
    betas=(0.9, 0.999),
    weight_decay=params["weight_decay"],
    lr=params["learning_rate"])
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, 
    lambda epoch: scheduler_lr_cosine_warmstart_retry(
        epoch, params["epoch"], 
        params["lr_scheduler_step_warmup"], 
        params["lr_scheduler_retry"]))
for epoch in range(params["epoch"]):
    batch_perm = torch.randperm(tensor_x.size(0), device=device)
    batch_x = tensor_x[batch_perm][0:params["size_minibatch"]].unsqueeze(1)
    batch_y = tensor_y[batch_perm][0:params["size_minibatch"]].unsqueeze(1)
    
    def expectile_loss(u,tau):
        return (tau-(u < 0.0).float()).abs()*u*u

    model_1.train()
    batch_pred_1 = model_1(batch_x)
    loss_1 = expectile_loss(batch_y-batch_pred_1, 0.01).mean()
    
    model_2.train()
    batch_pred_2 = model_2(batch_x)
    loss_2 = expectile_loss(batch_y-batch_pred_2, 0.99).mean()

    model_flow.train()
    loss_flow = flow.train_loss(
        model_flow, 
        torch.rand((params["size_minibatch"], 1, 1), device=device),
        batch_x,
        batch_y.unsqueeze(1))

    loss = loss_1 + loss_2 + loss_flow
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    ema_flow.update(model_flow)
    
    if epoch%10 == 0:
        print(epoch, loss.detach().item())

#Plot model
with torch.no_grad():
    model_1.eval()
    model_2.eval()
    model_flow.eval()
    batch_x = torch.linspace(-1.0, 1.0, 1000, device=device).unsqueeze(1)
    batch_y_1 = model_1(batch_x)
    batch_y_2 = model_2(batch_x)
    batch_flow = []
    for d in np.linspace(0.0, 1.0, 8):
        chain = flow.transport_forward(
            ema_flow.getModel(),
            d*torch.ones_like(batch_x).unsqueeze(1),
            batch_x,
            steps=params["forward_step"])
        batch_flow.append(chain[-1].cpu().numpy())
fig, axs = plt.subplots(1, 1)
axs.scatter(tensor_x.cpu(), tensor_y.cpu(), color="blue", alpha=0.3)
axs.scatter(batch_x.cpu()[:,0], batch_y_1.cpu()[:,0], color="red", alpha=0.3)
axs.scatter(batch_x.cpu()[:,0], batch_y_2.cpu()[:,0], color="green", alpha=0.3)
for c in batch_flow:
    axs.plot(batch_x.cpu()[:,0], c[:,0,0], color="purple", alpha=0.3)
axs.grid()

dict_data = {}
dict_data["array_x"] = array_x
dict_data["array_y"] = array_y
dict_data["batch_x"] = batch_x.cpu().numpy()
dict_data["batch_y_1"] = batch_y_1.cpu().numpy()
dict_data["batch_y_2"] = batch_y_2.cpu().numpy()
for i,c in enumerate(batch_flow):
    dict_data["flow_"+str(i)] = c
np.savez(
    "/tmp/docker_share/data.npz",
    **dict_data,
)
plt.show()

