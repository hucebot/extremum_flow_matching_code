import torch
from collections import OrderedDict

from models.base import BaseModel
from models.unet import ModelUnetResidualConditional
from models.unet import ModelDenseSimple
from models.impala import ImpalaEncoder
from models.mlp import MLPNet
from maths.sample import uniform_unit_ball, uniform_unit_sphere
from utils.training import EMAModel
from utils.training import scheduler_lr_cosine_warmstart_retry
from utils.flow_transport import FlowTransport

from typing import List, Tuple, Dict, Union

class GCRLAgentBase(BaseModel):
    """Base implementation for Flow Matching 
    Goal Conditioned Reinforcement Learning
    """

    @staticmethod
    def get_config() -> Dict:
        """Return Dict of default parameters"""
        config = {
            "epoch": 1000,
            "trajs_obs_len": 16,
            "trajs_act_len": 16,
            "trajs_obs_stride": 1,
            "trajs_act_stride": 1,
            "max_goal_dist": 100,
            "ema_cutoff": 100,
            "size_augment_with_reached": 32,
            "model_encoder_mlp_hidden": [128],
            "model_encoder_impala_hidden": [512],
            "model_encoder_impala_stacks": [3,16,32,32],
            "model_encoder_impala_dropout": 0.2,
            "model_flow_unet_emb_size": 32,
            "model_flow_unet_emb_period_min": 0.002,
            "model_flow_unet_emb_period_max": 10.0,
            "model_flow_unet_hidden": [32*2, 64*2, 128*2],
            "model_flow_mlp_hidden": [1024, 1024, 1024],
            "model_planner_name": "mlp",
            "weight_decay": 1e-6,
            "learning_rate": 1e-4,
            "scheduler_step_warmup": 100,
            "scheduler_retry": 1,
            "norm_clip_grad": 1.0,
            "forward_step": 20,
            "critic_ratio_training": 0.3,
            "critic_ratio_inference": 0.0,
            "inference_sampling_size": 64,
        }
        return config

    def __init__(self, 
            config: Dict,
            encoder_config: Tuple[str,str,int,int],
            size_act: int,
            device: torch.device = None,
        ):
        """Base agant initialization.
        Args:
            config: Configuration dictionary.
            encoder_config: List of tuple (label,name,size_in,size_out) for
                each observation label, encoder model name, 
                and in and out feature sizes.
            size_act: Number of predicted action features.
            device: Optional allocation device for agent models.
        """
        super().__init__()
        #Default empty initialization
        self._device = device
        self._size_act = size_act
        self._config = config
        self._info = dict()
        self._model_encoders = torch.nn.ModuleDict()
        self._model_critic_1 = None
        self._model_critic_2 = None
        self._model_planner = None
        self._model_actor = None
        self._model_world = None
        #Encoder models initialization
        self._size_latent_obs = 0
        for label,name,size_in,size_out in encoder_config:
            self._model_encoders[label] = self._create_model_encoder(
                name=name, 
                size_in=size_in, 
                size_out=size_out)
            self._size_latent_obs += size_out
        #Initialize EMA models
        self._ema_encoders = torch.nn.ModuleDict()
        for label in self._model_encoders.keys():
            self._ema_encoders[label] = EMAModel(
                cutoff_period=self._config["ema_cutoff"], 
                warmup_steps=self._config["scheduler_step_warmup"])
        self._ema_critic_1 = EMAModel(
            cutoff_period=self._config["ema_cutoff"], 
            warmup_steps=self._config["scheduler_step_warmup"])
        self._ema_critic_2 = EMAModel(
            cutoff_period=self._config["ema_cutoff"], 
            warmup_steps=self._config["scheduler_step_warmup"])
        self._ema_planner = EMAModel(
            cutoff_period=self._config["ema_cutoff"], 
            warmup_steps=self._config["scheduler_step_warmup"])
        self._ema_actor = EMAModel(
            cutoff_period=self._config["ema_cutoff"], 
            warmup_steps=self._config["scheduler_step_warmup"])
        self._ema_world = EMAModel(
            cutoff_period=self._config["ema_cutoff"], 
            warmup_steps=self._config["scheduler_step_warmup"])
        #Initialize flow implementation
        self._flow = FlowTransport(type_transport_func="linear")

    def _create_model_encoder(self,
            name: str,
            size_in: int,
            size_out: int,
        ) -> torch.nn.Module:
        """Create and return observation encoder model.
        Args:
            name: model type name.
            size_in: Observation input feature size.
            size_out: Encoded output feature size.
        """
        if name == "identity":
            assert size_in == size_out
            return torch.nn.Identity().to(self._device)
        elif name == "mlp":
            return MLPNet(
                size_in, self._config["model_encoder_mlp_hidden"], size_out,
                activation=torch.nn.SiLU, 
                spectral_norm_on_hidden=False,
                norm_layer=True,
            ).to(self._device)
        elif name == "impala":
            return ImpalaEncoder(
                size_stacks_channel=self._config["model_encoder_impala_stacks"],
                size_fc_in = size_in,
                size_fc_hidden=self._config["model_encoder_impala_hidden"],
                size_fc_out=size_out,
                size_block=1,
                use_layer_norm=True,
                dropout_rate=self._config["model_encoder_impala_dropout"],
            ).to(self._device)
        else:
            raise IOError("Model type name not implemented: " + name)

    def _create_model_flow(self,
            name,
            size_channel: int,
            size_length: int,
            size_cond: int,
        ) -> torch.nn.Module:
        """
        Args:
            name: model type name.
            size_channel: Output feature size.
            size_length: Output trajectory length.
            size_cond: Input conditioning feature size.
        """
        if name == "mlp":
            return ModelDenseSimple(
                size_channel=size_channel,
                size_length=size_length,
                size_cond=size_cond,
                size_hidden_list=self._config["model_flow_mlp_hidden"],
            ).to(self._device)
        elif name == "unet":
            return ModelUnetResidualConditional(
                size_channel=size_channel,
                size_emb_transport=self._config["model_flow_unet_emb_size"],
                size_cond=size_cond,
                size_channel_hidden=self._config["model_flow_unet_hidden"],
                period_min=self._config["model_flow_unet_emb_period_min"],
                period_max=self._config["model_flow_unet_emb_period_max"],
                size_kernel=3,
                size_group_norm=8,
            ).to(self._device)
        else:
            raise IOError("Model type name not implemented: " + name)

    def _init_training(self):
        """Initialize optimizer and scheduler for training.
        To be called after models setup.
        """
        self._all_model_params = []
        for label in self._model_encoders.keys():
            self._all_model_params.extend(list(self._model_encoders[label].parameters()))
        if self._model_critic_1 is not None:
            self._all_model_params.extend(list(self._model_critic_1.parameters()))
        if self._model_critic_2 is not None:
            self._all_model_params.extend(list(self._model_critic_2.parameters()))
        if self._model_planner is not None:
            self._all_model_params.extend(list(self._model_planner.parameters()))
        if self._model_actor is not None:
            self._all_model_params.extend(list(self._model_actor.parameters()))
        if self._model_world is not None:
            self._all_model_params.extend(list(self._model_world.parameters()))
        self._optimizer = torch.optim.AdamW(
            self._all_model_params,
            betas=(0.9, 0.999),
            weight_decay=self._config["weight_decay"],
            lr=self._config["learning_rate"],
        )
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer,
            lambda epoch: scheduler_lr_cosine_warmstart_retry(
                epoch, self._config["epoch"], 
                self._config["scheduler_step_warmup"], 
                self._config["scheduler_retry"]))
        self._update_ema_models(reset=True)

    def _update_ema_models(self, 
            reset: bool,
        ):
        """Update all EMA models.
        Args:
            reset: If True, reset all EMA models.
        """
        for label in self._model_encoders.keys():
            if reset:
                self._ema_encoders[label].reset()
            self._ema_encoders[label].update(self._model_encoders[label])
        if self._model_critic_1 is not None:
            if reset:
                self._ema_critic_1.reset()
            self._ema_critic_1.update(self._model_critic_1)
        if self._model_critic_2 is not None:
            if reset:
                self._ema_critic_2.reset()
            self._ema_critic_2.update(self._model_critic_2)
        if self._model_planner is not None:
            if reset:
                self._ema_planner.reset()
            self._ema_planner.update(self._model_planner)
        if self._model_actor is not None:
            if reset:
                self._ema_actor.reset()
            self._ema_actor.update(self._model_actor)
        if self._model_world is not None:
            if reset:
                self._ema_world.reset()
            self._ema_world.update(self._model_world)

    def _update_train_from_loss(self, 
            loss: torch.Tensor,
        ):
        """Run backward pass and update all 
        model parameters from given scalar loss
        """
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._all_model_params, self._config["norm_clip_grad"])
        self._optimizer.step()
        self._update_ema_models(reset=False)
        self._scheduler.step()

    def save(self, path: str):
        """Save agent model parameters to given path.
        Args:
            path: Path prefix toward the folder to save parameters.
        """
        #Save all EMA model parameters
        for label in self._model_encoders.keys():
            torch.save(
                self._ema_encoders[label].getModel().state_dict(), 
                path+"model_encoder_"+label+".params")
        if self._model_critic_1 is not None:
            torch.save(
                self._ema_critic_1.getModel().state_dict(),
                path+"model_critic_1.params")
        if self._model_critic_2 is not None:
            torch.save(
                self._ema_critic_2.getModel().state_dict(),
                path+"model_critic_2.params")
        if self._model_planner is not None:
            torch.save(
                self._ema_planner.getModel().state_dict(),
                path+"model_planner.params")
        if self._model_actor is not None:
            torch.save(
                self._ema_actor.getModel().state_dict(),
                path+"model_actor.params")
        if self._model_world is not None:
            torch.save(
                self._ema_world.getModel().state_dict(),
                path+"model_world.params")

    def load(self, path: str):
        """Load agent model parameters from given path.
        Args:
            path: Path prefix toward the folder to load parameters.
        """
        for label in self._model_encoders.keys():
            self._model_encoders[label].load_state_dict(
                torch.load(path+"model_encoder_"+label+".params", weights_only=True))
        if self._model_critic_1 is not None:
            self._model_critic_1.load_state_dict(
                torch.load(path+"model_critic_1.params", weights_only=True))
        if self._model_critic_2 is not None:
            self._model_critic_2.load_state_dict(
                torch.load(path+"model_critic_2.params", weights_only=True))
        if self._model_planner is not None:
            self._model_planner.load_state_dict(
                torch.load(path+"model_planner.params", weights_only=True))
        if self._model_actor is not None:
            self._model_actor.load_state_dict(
                torch.load(path+"model_actor.params", weights_only=True))
        if self._model_world is not None:
            self._model_world.load_state_dict(
                torch.load(path+"model_world.params", weights_only=True))
        self._update_ema_models(reset=True)

    def encode_observations(self, 
            batch_obs: Dict[str,torch.Tensor],
            is_eval: bool,
            use_ema: bool,
            is_traj: bool,
        ) -> torch.Tensor:
        """Encode given observation to latent space.
        Args:
            batch_obs: Dictionary of observations to encode (size_batch,...), or
                (size_batch, size_length, ...) if is_traj.
            is_eval: If True, use evaluation mode and do not compute gradients.
            use_ema: If True, use EMA models instead to raw models.
            is_traj: If True, assume input observations are trajectories (size_batch, size_length, ...).
        Returns:
            Encoded latent space observations (size_batch, size_latent_obs), 
            or (size_batch, size_length, size_latent_obs) if is_traj.
        """
        with torch.set_grad_enabled(not is_eval):
            list_latent = []
            for label in self._model_encoders.keys():
                tmp_obs = batch_obs[label]
                if is_traj:
                    size_batch = tmp_obs.size(0)
                    size_length = tmp_obs.size(1)
                    size_other = tmp_obs.size()[2:]
                    tmp_obs = tmp_obs.reshape(size_batch*size_length, *size_other)
                if use_ema:
                    if is_eval:
                        self._ema_encoders[label].eval()
                    else:
                        self._ema_encoders[label].train()
                    list_latent.append(self._ema_encoders[label](tmp_obs))
                else:
                    if is_eval:
                        self._model_encoders[label].eval()
                    else:
                        self._model_encoders[label].train()
                    list_latent.append(self._model_encoders[label](tmp_obs))
            batch_latent_obs = torch.cat(list_latent, dim=-1)
            torch._assert(batch_latent_obs.size(-1) == self._size_latent_obs, "")
            if is_traj:
                batch_latent_obs = batch_latent_obs.reshape(
                    size_batch, size_length, self._size_latent_obs)
            return batch_latent_obs

    @torch.no_grad()
    def _generate_dist_trajs_src(self,
            size_batch: int,
            size_length: int,
            size_channel: int,
        ) -> torch.Tensor:
        """Sample trajectories from noise source distribution 
        with given batch and trajectory sizes.
        Returns sampled trajectory (size_batch, size_length, size_channel).
        """
        return 1.0*uniform_unit_ball(size_batch, size_channel*size_length, self._device) \
            .reshape(size_batch, size_length, size_channel)
    
    @torch.no_grad()
    def _generate_dist_critic_src(self,
            size_batch: int,
            ratio: float = 1.0,
            random: bool = True,
        ) -> torch.Tensor:
        """Sample scalar critic from uniform noise source distribution 
        with given batch and ratio between 0 and 1.
        Returns sampled scalar critic (size_batch, 1, 1).
        """
        if random:
            return ratio*torch.rand(size_batch, 1, 1, device=self._device)
        else:
            return ratio*torch.ones(size_batch, 1, 1, device=self._device)

    @torch.no_grad()
    def merge_trajectories(self,
            trajs1: torch.Tensor,
            dist1: torch.Tensor, 
            trajs2: torch.Tensor, 
            stride: int, 
        ) -> torch.Tensor:
        """
        Merge two given trajectories by appending the second one to the first of the length 
        if the first trajectory is shorter then its sampled duration, else only return the first trajectory.
        Args:
            trajs1: First trajectory to merge (size_batch, size_length, size_channel).
            dist1: Denormalized time length of first trajectory.
            trajs2: Second trajectory to merge (size_batch, size_length, size_channel).
            stride: Sampling stride used by both trajectories.
        Returns the merge trajectory (size_batch, size_length, size_channel).
        """
        assert len(trajs1.size()) == 3 and len(dist1.size()) == 2
        assert trajs1.size() == trajs2.size()
        assert trajs1.size(0) == dist1.size(0) 
        assert stride >= 1
        size_batch = trajs1.size(0)
        size_length = trajs1.size(1)
        size_channel = trajs1.size(2)
        merged_trajs = torch.cat([
            trajs1[:,:,:], 
            trajs2[:,1:,:],
            ], dim=1)
        merged_times = torch.cat([
            torch.clip(torch.arange(0.0, stride*size_length, stride, device=self._device) \
                .unsqueeze(0).repeat(size_batch,1)[:,:], max=dist1),
            dist1+torch.arange(0.0, stride*size_length, stride, device=self._device) \
                .unsqueeze(0).repeat(size_batch,1)[:,1:],
            ], dim=1)
        desired_times = torch.arange(0.0, stride*size_length, stride, device=self._device).unsqueeze(0).repeat(size_batch, 1)
        diff_times = torch.abs(merged_times.unsqueeze(1)-desired_times.unsqueeze(2))
        indices = torch.argmin(diff_times, dim=2)
        result_trajs = torch.gather(merged_trajs, 1, indices.unsqueeze(2).repeat(1,1,size_channel))
        return result_trajs

    def print_parameters(self,
            print_all: bool = False,
        ):
        """Print model parameters"""
        super().print_parameters(print_all)
        for label in self._model_encoders.keys():
            if not isinstance(self._model_encoders[label], torch.nn.Identity):
                print("Model Encoder", label)
                self._model_encoders[label].print_parameters(print_all)
        if self._model_critic_1 is not None:
            print("Model Critic 1")
            self._model_critic_1.print_parameters(print_all)
        if self._model_critic_2 is not None:
            print("Model Critic 2")
            self._model_critic_2.print_parameters(print_all)
        if self._model_planner is not None:
            print("Model Planner")
            self._model_planner.print_parameters(print_all)
        if self._model_actor is not None:
            print("Model Actor")
            self._model_actor.print_parameters(print_all)
        if self._model_world is not None:
            print("Model World")
            self._model_world.print_parameters(print_all)

    def train(self,
            batch_obs: Dict[str, torch.Tensor],
            batch_goal: Dict[str,torch.Tensor],
            batch_dist: torch.Tensor,
            batch_traj_obs: Dict[str,torch.Tensor],
            batch_traj_act: torch.Tensor,
            batch_other_obs: Dict[str,torch.Tensor],
        ) -> float:
        """Train update the agent given batched dataset.
        Args:
            batch_obs: Sampled batched observations dict of (size_batch, ...).
            batch_goal: Sampled batched goal observations dict of (size_batch, ...).
            batch_dist: Sampled batched time step distance from observation to goal (size_batch).
            batch_traj_obs: Sampled batched observation trajectories dict (size_batch, trajs_obs_len, ...).
            batch_traj_act: Sampled batched action trajectories (size_batch, trajs_act_len, size_act).
            batch_other_obs: Sampled batch other observation from whole dataset dict of (size_batch, ...).
        Returns:
            Float loss scalar value.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def inference(self, 
            batch_obs: torch.Tensor, 
            batch_goal: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict]:
        """Compute agent inference.
        Args:
            batch_obs: Batched observations dict of (size_batch, ...).
            batch_goal: Batched goal observations dict of (size_batch, ...).
        Returns:
            Predicted batch action trajectories (size_batch, trajs_act_len, size_act).
        """
        raise NotImplementedError()

    @torch.no_grad()
    def forward(self, 
            batch_obs: torch.Tensor, 
            batch_goal: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict]:
        """Compute agent inference.
        Args:
            batch_obs: Batched observations dict of (size_batch, ...).
            batch_goal: Batched goal observations dict of (size_batch, ...).
        Returns:
            Predicted batch action trajectories (size_batch, trajs_act_len, size_act).
        """
        return self.inference(batch_obs, batch_goal)

class GCRLAgentBC(GCRLAgentBase):
    def __init__(self,
            config: Dict,
            encoder_config: Tuple[str,str,int,int],
            size_act: int,
            device: torch.device = None,
        ):
        super().__init__(
            config=config,
            encoder_config=encoder_config,
            size_act=size_act,
            device=device,
        )
        self._model_actor = self._create_model_flow(
            name="unet", 
            size_channel=self._size_act,
            size_length=self._config["trajs_act_len"],
            size_cond=self._size_latent_obs)
        self._init_training()
    def train(self,
            batch_obs: Dict[str, torch.Tensor],
            batch_goal: Dict[str,torch.Tensor],
            batch_dist: torch.Tensor,
            batch_traj_obs: Dict[str,torch.Tensor],
            batch_traj_act: torch.Tensor,
            batch_other_obs: Dict[str,torch.Tensor],
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        size_batch = tmp_latent_obs.size(0)
        self._model_actor.train()
        loss = self._flow.train_loss(
            self._model_actor,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            tmp_latent_obs,
            batch_traj_act,
        )
        self._update_train_from_loss(loss)
        return loss.detach().cpu().item()
    @torch.no_grad()
    def inference(self, 
            batch_obs: torch.Tensor, 
            batch_goal: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        size_batch = tmp_latent_obs.size(0)
        self._ema_actor.eval()
        chain1 = self._flow.transport_forward(
            self._ema_actor,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            tmp_latent_obs, 
            steps=self._config["forward_step"])

        self._info["latent_obs"] = tmp_latent_obs
        self._info["traj_act"] = chain1[-1]
        return chain1[-1]

class GCRLAgent0(GCRLAgentBase):
    def __init__(self,
            config: Dict,
            encoder_config: Tuple[str,str,int,int],
            size_act: int,
            device: torch.device = None,
        ):
        super().__init__(
            config=config,
            encoder_config=encoder_config,
            size_act=size_act,
            device=device,
        )
        self._model_actor = self._create_model_flow(
            name="unet", 
            size_channel=self._size_act,
            size_length=self._config["trajs_act_len"],
            size_cond=2*self._size_latent_obs)
        self._init_training()
    def train(self,
            batch_obs: Dict[str, torch.Tensor],
            batch_goal: Dict[str,torch.Tensor],
            batch_dist: torch.Tensor,
            batch_traj_obs: Dict[str,torch.Tensor],
            batch_traj_act: torch.Tensor,
            batch_other_obs: Dict[str,torch.Tensor],
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        tmp_latent_goal = self.encode_observations(
            batch_goal,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        size_batch = tmp_latent_obs.size(0)
        self._model_actor.train()
        loss = self._flow.train_loss(
            self._model_actor,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            batch_traj_act,
        )
        self._update_train_from_loss(loss)
        return loss.detach().cpu().item()
    @torch.no_grad()
    def inference(self, 
            batch_obs: torch.Tensor, 
            batch_goal: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        tmp_latent_goal = self.encode_observations(
            batch_goal,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        size_batch = tmp_latent_obs.size(0)
        self._ema_actor.eval()
        chain1 = self._flow.transport_forward(
            self._ema_actor,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            steps=self._config["forward_step"])

        self._info["latent_obs"] = tmp_latent_obs
        self._info["latent_goal"] = tmp_latent_goal
        self._info["traj_act"] = chain1[-1]
        return chain1[-1]

class GCRLAgent1(GCRLAgentBase):
    def __init__(self,
            config: Dict,
            encoder_config: Tuple[str,str,int,int],
            size_act: int,
            use_rl: bool,
            use_merge_traj: bool,
            device: torch.device = None,
        ):
        super().__init__(
            config=config,
            encoder_config=encoder_config,
            size_act=size_act,
            device=device,
        )
        self._use_rl = use_rl
        self._use_merge_traj = use_merge_traj
        self._model_critic_1 = self._create_model_flow(
            name="mlp", 
            size_channel=1,
            size_length=1,
            size_cond=2*self._size_latent_obs)
        self._model_critic_2 = self._create_model_flow(
            name="mlp", 
            size_channel=1,
            size_length=1,
            size_cond=2*self._size_latent_obs)
        self._model_actor = self._create_model_flow(
            name="unet", 
            size_channel=self._size_act,
            size_length=self._config["trajs_act_len"],
            size_cond=2*self._size_latent_obs+1)
        self._init_training()
    def train(self,
            batch_obs: Dict[str, torch.Tensor],
            batch_goal: Dict[str,torch.Tensor],
            batch_dist: torch.Tensor,
            batch_traj_obs: Dict[str,torch.Tensor],
            batch_traj_act: torch.Tensor,
            batch_other_obs: Dict[str,torch.Tensor],
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        tmp_latent_goal = self.encode_observations(
            batch_goal,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        size_batch = tmp_latent_obs.size(0)
        if self._use_rl:
            with torch.no_grad():
                tmp_latent_other = self.encode_observations(
                    batch_other_obs,
                    is_eval=True,
                    use_ema=True,
                    is_traj=False,
                )
                ratio = self._config["critic_ratio_training"]
                self._ema_critic_1.eval()
                chain1 = self._flow.transport_forward(
                    self._ema_critic_1,
                    self._generate_dist_critic_src(size_batch, ratio=ratio),
                    torch.cat([
                        tmp_latent_goal, 
                        tmp_latent_other,
                        ], dim=1),
                    steps=self._config["forward_step"])
                self._ema_critic_2.eval()
                chain2 = self._flow.transport_forward(
                    self._ema_critic_2,
                    self._generate_dist_critic_src(size_batch, ratio=ratio),
                    torch.cat([
                        tmp_latent_goal, 
                        tmp_latent_other,
                        ], dim=1),
                    steps=self._config["forward_step"])
                tmp_sampled_dist = torch.maximum(chain1[-1][:,0,:], chain2[-1][:,0,:])
                tmp_new_dist = batch_dist + tmp_sampled_dist[:,0]
                if self._use_merge_traj:
                    self._ema_actor.eval()
                    chain3 = self._flow.transport_forward(
                        self._ema_actor,
                        self._generate_dist_trajs_src(
                            size_batch, 
                            self._config["trajs_act_len"],
                            self._size_act, 
                        ),
                        torch.cat([
                            tmp_latent_obs, 
                            tmp_latent_goal,
                            tmp_sampled_dist,
                            ], dim=1),
                        steps=self._config["forward_step"])
                    tmp_new_traj_act = self.merge_trajectories(
                        batch_traj_act, 
                        batch_dist.unsqueeze(1)*self._config["max_goal_dist"], 
                        chain3[-1], 
                        self._config["trajs_act_stride"])
                else:
                    tmp_new_traj_act = batch_traj_act
            tmp_latent_obs = torch.cat([
                tmp_latent_obs,
                tmp_latent_obs,
                ], dim=0)
            tmp_latent_goal = torch.cat([
                tmp_latent_goal,
                tmp_latent_other,
                ], dim=0)
            batch_dist = torch.cat([
                batch_dist,
                tmp_new_dist,
                ], dim=0)
            batch_traj_act = torch.cat([
                batch_traj_act,
                tmp_new_traj_act,
                ], dim=0)
        if self._config["size_augment_with_reached"] > 0:
            size_augment = self._config["size_augment_with_reached"]
            tmp_latent_goal = torch.cat([
                tmp_latent_goal,
                tmp_latent_obs[0:size_augment],
                ], dim=0)
            tmp_latent_obs = torch.cat([
                tmp_latent_obs,
                tmp_latent_obs[0:size_augment],
                ], dim=0)
            batch_dist = torch.cat([
                batch_dist,
                torch.zeros_like(batch_dist[0:size_augment]),
                ], dim=0)
            batch_traj_act = torch.cat([
                batch_traj_act,
                batch_traj_act[0:size_augment,0:1,:].repeat(1,self._config["trajs_act_len"],1),
                ], dim=0)
        size_batch = tmp_latent_obs.size(0)
        self._model_critic_1.train()
        loss_critic_1 = self._flow.train_loss(
            self._model_critic_1,
            self._generate_dist_critic_src(size_batch),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            batch_dist.unsqueeze(1).unsqueeze(2),
        )
        self._model_critic_2.train()
        loss_critic_2 = self._flow.train_loss(
            self._model_critic_2,
            self._generate_dist_critic_src(size_batch),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            batch_dist.unsqueeze(1).unsqueeze(2),
        )
        self._model_actor.train()
        loss_actor = self._flow.train_loss(
            self._model_actor,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                batch_dist.unsqueeze(1),
                ], dim=1),
            batch_traj_act,
        )
        loss = 0.1*loss_critic_1 + 0.1*loss_critic_2 + loss_actor
        self._update_train_from_loss(loss)
        return loss.detach().cpu().item()
    @torch.no_grad()
    def inference(self, 
            batch_obs: torch.Tensor, 
            batch_goal: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        tmp_latent_goal = self.encode_observations(
            batch_goal,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        size_batch = tmp_latent_obs.size(0)
        ratio = self._config["critic_ratio_inference"]
        self._ema_critic_1.eval()
        chain1 = self._flow.transport_forward(
            self._ema_critic_1,
            self._generate_dist_critic_src(size_batch, ratio=ratio),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            steps=self._config["forward_step"])
        self._ema_critic_2.eval()
        chain2 = self._flow.transport_forward(
            self._ema_critic_2,
            self._generate_dist_critic_src(size_batch, ratio=ratio),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            steps=self._config["forward_step"])
        self._ema_actor.eval()
        chain3 = self._flow.transport_forward(
            self._ema_actor,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                torch.maximum(chain1[-1][:,0,:], chain2[-1][:,0,:]),
                ], dim=1),
            steps=self._config["forward_step"])
        
        self._info["latent_obs"] = tmp_latent_obs
        self._info["latent_goal"] = tmp_latent_goal
        self._info["critic_1"] = chain1[-1][:,0,0]
        self._info["critic_2"] = chain2[-1][:,0,0]
        self._info["critic_max"] = torch.maximum(chain1[-1][:,0,0], chain2[-1][:,0,0])
        self._info["traj_act"] = chain3[-1]
        return chain3[-1]

class GCRLAgent2(GCRLAgentBase):
    def __init__(self,
            config: Dict,
            encoder_config: Tuple[str,str,int,int],
            size_act: int,
            use_rl: bool,
            device: torch.device = None,
        ):
        super().__init__(
            config=config,
            encoder_config=encoder_config,
            size_act=size_act,
            device=device,
        )
        self._use_rl = use_rl
        self._model_critic_1 = self._create_model_flow(
            name="mlp", 
            size_channel=1,
            size_length=1,
            size_cond=2*self._size_latent_obs)
        self._model_critic_2 = self._create_model_flow(
            name="mlp", 
            size_channel=1,
            size_length=1,
            size_cond=2*self._size_latent_obs)
        self._model_planner = self._create_model_flow(
            name=self._config["model_planner_name"], 
            size_channel=self._size_latent_obs,
            size_length=self._config["trajs_obs_len"],
            size_cond=2*self._size_latent_obs+1)
        self._model_actor = self._create_model_flow(
            name="unet", 
            size_channel=self._size_act,
            size_length=self._config["trajs_act_len"],
            size_cond=self._size_latent_obs+self._config["trajs_obs_len"]*self._size_latent_obs)
        self._init_training()
    def train(self,
            batch_obs: Dict[str, torch.Tensor],
            batch_goal: Dict[str,torch.Tensor],
            batch_dist: torch.Tensor,
            batch_traj_obs: Dict[str,torch.Tensor],
            batch_traj_act: torch.Tensor,
            batch_other_obs: Dict[str,torch.Tensor],
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        tmp_latent_goal = self.encode_observations(
            batch_goal,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        tmp_latent_traj_obs = self.encode_observations(
            batch_traj_obs,
            is_eval=False,
            use_ema=False,
            is_traj=True,
        )
        size_batch = tmp_latent_obs.size(0)
        if self._use_rl:
            with torch.no_grad():
                tmp_latent_other = self.encode_observations(
                    batch_other_obs,
                    is_eval=True,
                    use_ema=True,
                    is_traj=False,
                )
                ratio = self._config["critic_ratio_training"]
                self._ema_critic_1.eval()
                chain1 = self._flow.transport_forward(
                    self._ema_critic_1,
                    self._generate_dist_critic_src(size_batch, ratio=ratio),
                    torch.cat([
                        tmp_latent_goal, 
                        tmp_latent_other,
                        ], dim=1),
                    steps=self._config["forward_step"])
                self._ema_critic_2.eval()
                chain2 = self._flow.transport_forward(
                    self._ema_critic_2,
                    self._generate_dist_critic_src(size_batch, ratio=ratio),
                    torch.cat([
                        tmp_latent_goal, 
                        tmp_latent_other,
                        ], dim=1),
                    steps=self._config["forward_step"])
                tmp_new_dist = batch_dist + torch.maximum(chain1[-1][:,0,0], chain2[-1][:,0,0])
            tmp_latent_obs = torch.cat([
                tmp_latent_obs,
                tmp_latent_obs,
                ], dim=0)
            tmp_latent_goal = torch.cat([
                tmp_latent_goal,
                tmp_latent_other,
                ], dim=0)
            tmp_latent_traj_obs = torch.cat([
                tmp_latent_traj_obs,
                tmp_latent_traj_obs,
                ], dim=0)
            batch_dist = torch.cat([
                batch_dist,
                tmp_new_dist,
                ], dim=0)
            batch_traj_act = torch.cat([
                batch_traj_act,
                batch_traj_act,
                ], dim=0)
        if self._config["size_augment_with_reached"] > 0:
            size_augment = self._config["size_augment_with_reached"]
            tmp_latent_goal = torch.cat([
                tmp_latent_goal,
                tmp_latent_obs[0:size_augment],
                ], dim=0)
            tmp_latent_obs = torch.cat([
                tmp_latent_obs,
                tmp_latent_obs[0:size_augment],
                ], dim=0)
            tmp_latent_traj_obs = torch.cat([
                tmp_latent_traj_obs,
                tmp_latent_traj_obs[0:size_augment,0:1,:].repeat(1,self._config["trajs_obs_len"],1),
                ], dim=0)
            batch_dist = torch.cat([
                batch_dist,
                torch.zeros_like(batch_dist[0:size_augment]),
                ], dim=0)
            batch_traj_act = torch.cat([
                batch_traj_act,
                batch_traj_act[0:size_augment,0:1,:].repeat(1,self._config["trajs_act_len"],1),
                ], dim=0)
        size_batch = tmp_latent_obs.size(0)
        self._model_critic_1.train()
        loss_critic_1 = self._flow.train_loss(
            self._model_critic_1,
            self._generate_dist_critic_src(size_batch),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            batch_dist.unsqueeze(1).unsqueeze(2),
        )
        self._model_critic_2.train()
        loss_critic_2 = self._flow.train_loss(
            self._model_critic_2,
            self._generate_dist_critic_src(size_batch),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            batch_dist.unsqueeze(1).unsqueeze(2),
        )
        self._model_planner.train()
        loss_planner = self._flow.train_loss(
            self._model_planner,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_obs_len"],
                self._size_latent_obs, 
            ),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                batch_dist.unsqueeze(1),
                ], dim=1),
            tmp_latent_traj_obs,
        )
        self._model_actor.train()
        loss_actor = self._flow.train_loss(
            self._model_actor,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            torch.cat([
                tmp_latent_obs, 
                torch.flatten(tmp_latent_traj_obs, start_dim=1),
                ], dim=1),
            batch_traj_act,
        )
        loss = 0.1*loss_critic_1 + 0.1*loss_critic_2 + loss_planner + loss_actor
        self._update_train_from_loss(loss)
        return loss.detach().cpu().item()
    @torch.no_grad()
    def inference(self, 
            batch_obs: torch.Tensor, 
            batch_goal: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        tmp_latent_goal = self.encode_observations(
            batch_goal,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        size_batch = tmp_latent_obs.size(0)
        ratio = self._config["critic_ratio_inference"]
        self._ema_critic_1.eval()
        chain1 = self._flow.transport_forward(
            self._ema_critic_1,
            self._generate_dist_critic_src(size_batch, ratio=ratio),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            steps=self._config["forward_step"])
        self._ema_critic_2.eval()
        chain2 = self._flow.transport_forward(
            self._ema_critic_2,
            self._generate_dist_critic_src(size_batch, ratio=ratio),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            steps=self._config["forward_step"])
        self._ema_planner.eval()
        chain3 = self._flow.transport_forward(
            self._ema_planner,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_obs_len"],
                self._size_latent_obs,
            ),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                torch.maximum(chain1[-1][:,0,:], chain2[-1][:,0,:]),
                ], dim=1),
            steps=self._config["forward_step"])
        tmp_trajs_obs = chain3[-1]
        self._ema_actor.eval()
        chain4 = self._flow.transport_forward(
            self._ema_actor,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            torch.cat([
                tmp_latent_obs, 
                torch.flatten(tmp_trajs_obs, start_dim=1),
                ], dim=1),
            steps=self._config["forward_step"])
        
        self._info["latent_obs"] = tmp_latent_obs
        self._info["latent_goal"] = tmp_latent_goal
        self._info["critic_1"] = chain1[-1][:,0,0]
        self._info["critic_2"] = chain2[-1][:,0,0]
        self._info["critic_max"] = torch.maximum(chain1[-1][:,0,0], chain2[-1][:,0,0])
        self._info["traj_obs"] = chain3[-1]
        self._info["traj_act"] = chain4[-1]
        return chain4[-1]

class GCRLAgent5(GCRLAgentBase):
    def __init__(self,
            config: Dict,
            encoder_config: Tuple[str,str,int,int],
            size_act: int,
            use_rl: bool,
            device: torch.device = None,
        ):
        super().__init__(
            config=config,
            encoder_config=encoder_config,
            size_act=size_act,
            device=device,
        )
        self._use_rl = use_rl
        self._model_critic_1 = self._create_model_flow(
            name="mlp", 
            size_channel=1,
            size_length=1,
            size_cond=2*self._size_latent_obs)
        self._model_critic_2 = self._create_model_flow(
            name="mlp", 
            size_channel=1,
            size_length=1,
            size_cond=2*self._size_latent_obs)
        self._model_planner = self._create_model_flow(
            name=self._config["model_planner_name"], 
            size_channel=self._size_latent_obs,
            size_length=self._config["trajs_obs_len"],
            size_cond=self._size_latent_obs)
        self._model_actor = self._create_model_flow(
            name="unet", 
            size_channel=self._size_act,
            size_length=self._config["trajs_act_len"],
            size_cond=self._size_latent_obs+self._config["trajs_obs_len"]*self._size_latent_obs)
        self._init_training()
    def train(self,
            batch_obs: Dict[str, torch.Tensor],
            batch_goal: Dict[str,torch.Tensor],
            batch_dist: torch.Tensor,
            batch_traj_obs: Dict[str,torch.Tensor],
            batch_traj_act: torch.Tensor,
            batch_other_obs: Dict[str,torch.Tensor],
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        tmp_latent_goal = self.encode_observations(
            batch_goal,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        tmp_latent_traj_obs = self.encode_observations(
            batch_traj_obs,
            is_eval=False,
            use_ema=False,
            is_traj=True,
        )
        size_batch = tmp_latent_obs.size(0)
        if self._use_rl:
            with torch.no_grad():
                tmp_latent_other = self.encode_observations(
                    batch_other_obs,
                    is_eval=True,
                    use_ema=True,
                    is_traj=False,
                )
                ratio = self._config["critic_ratio_training"]
                self._ema_critic_1.eval()
                chain1 = self._flow.transport_forward(
                    self._ema_critic_1,
                    self._generate_dist_critic_src(size_batch, ratio=ratio),
                    torch.cat([
                        tmp_latent_goal, 
                        tmp_latent_other,
                        ], dim=1),
                    steps=self._config["forward_step"])
                self._ema_critic_2.eval()
                chain2 = self._flow.transport_forward(
                    self._ema_critic_2,
                    self._generate_dist_critic_src(size_batch, ratio=ratio),
                    torch.cat([
                        tmp_latent_goal, 
                        tmp_latent_other,
                        ], dim=1),
                    steps=self._config["forward_step"])
                tmp_new_dist = batch_dist + torch.maximum(chain1[-1][:,0,0], chain2[-1][:,0,0])
            tmp_latent_obs = torch.cat([
                tmp_latent_obs,
                tmp_latent_obs,
                ], dim=0)
            tmp_latent_goal = torch.cat([
                tmp_latent_goal,
                tmp_latent_other,
                ], dim=0)
            tmp_latent_traj_obs = torch.cat([
                tmp_latent_traj_obs,
                tmp_latent_traj_obs,
                ], dim=0)
            batch_dist = torch.cat([
                batch_dist,
                tmp_new_dist,
                ], dim=0)
            batch_traj_act = torch.cat([
                batch_traj_act,
                batch_traj_act,
                ], dim=0)
        if self._config["size_augment_with_reached"] > 0:
            size_augment = self._config["size_augment_with_reached"]
            tmp_latent_goal = torch.cat([
                tmp_latent_goal,
                tmp_latent_obs[0:size_augment],
                ], dim=0)
            tmp_latent_obs = torch.cat([
                tmp_latent_obs,
                tmp_latent_obs[0:size_augment],
                ], dim=0)
            tmp_latent_traj_obs = torch.cat([
                tmp_latent_traj_obs,
                tmp_latent_traj_obs[0:size_augment,0:1,:].repeat(1,self._config["trajs_obs_len"],1),
                ], dim=0)
            batch_dist = torch.cat([
                batch_dist,
                torch.zeros_like(batch_dist[0:size_augment]),
                ], dim=0)
            batch_traj_act = torch.cat([
                batch_traj_act,
                batch_traj_act[0:size_augment,0:1,:].repeat(1,self._config["trajs_act_len"],1),
                ], dim=0)
        size_batch = tmp_latent_obs.size(0)
        self._model_critic_1.train()
        loss_critic_1 = self._flow.train_loss(
            self._model_critic_1,
            self._generate_dist_critic_src(size_batch),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            batch_dist.unsqueeze(1).unsqueeze(2),
        )
        self._model_critic_2.train()
        loss_critic_2 = self._flow.train_loss(
            self._model_critic_2,
            self._generate_dist_critic_src(size_batch),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            batch_dist.unsqueeze(1).unsqueeze(2),
        )
        self._model_planner.train()
        loss_planner = self._flow.train_loss(
            self._model_planner,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_obs_len"],
                self._size_latent_obs, 
            ),
            tmp_latent_obs, 
            tmp_latent_traj_obs,
        )
        self._model_actor.train()
        loss_actor = self._flow.train_loss(
            self._model_actor,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            torch.cat([
                tmp_latent_obs, 
                torch.flatten(tmp_latent_traj_obs, start_dim=1),
                ], dim=1),
            batch_traj_act,
        )
        loss = 0.1*loss_critic_1 + 0.1*loss_critic_2 + loss_planner + loss_actor
        self._update_train_from_loss(loss)
        return loss.detach().cpu().item()
    @torch.no_grad()
    def inference(self, 
            batch_obs: torch.Tensor, 
            batch_goal: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        tmp_latent_goal = self.encode_observations(
            batch_goal,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        size_batch = tmp_latent_obs.size(0)
        size_sample = self._config["inference_sampling_size"]
        self._ema_planner.eval()
        chain0 = self._flow.transport_forward(
            self._ema_planner,
            self._generate_dist_trajs_src(
                size_batch*size_sample,
                self._config["trajs_obs_len"],
                self._size_latent_obs, 
            ),
            tmp_latent_obs.repeat_interleave(size_sample,dim=0),
            steps=self._config["forward_step"])

        ratio = self._config["critic_ratio_inference"]
        self._ema_critic_1.eval()
        chain1 = self._flow.transport_forward(
            self._ema_critic_1,
            self._generate_dist_critic_src(size_batch*size_sample, ratio=ratio),
            torch.cat([
                chain0[-1][:,-1,:],
                tmp_latent_goal.repeat_interleave(size_sample,dim=0), 
                ], dim=1),
            steps=self._config["forward_step"])
        self._ema_critic_2.eval()
        chain2 = self._flow.transport_forward(
            self._ema_critic_2,
            self._generate_dist_critic_src(size_batch*size_sample, ratio=ratio),
            torch.cat([
                chain0[-1][:,-1,:],
                tmp_latent_goal.repeat_interleave(size_sample,dim=0), 
                ], dim=1),
            steps=self._config["forward_step"])

        tmp_dist = torch.maximum(chain1[-1][:,0,0], chain2[-1][:,0,0])
        tmp_dist = tmp_dist.reshape(size_batch, size_sample)
        tmp_indices = torch.min(tmp_dist, dim=1, keepdim=False)[1]
        tmp_trajs_obs = chain0[-1].reshape(size_batch, size_sample, self._config["trajs_obs_len"], self._size_latent_obs)
        tmp_trajs_obs = tmp_trajs_obs[:,tmp_indices,:,:]

        self._ema_actor.eval()
        chain3 = self._flow.transport_forward(
            self._ema_actor,
            self._generate_dist_trajs_src(
                size_batch,
                self._config["trajs_act_len"],
                self._size_act,
            ),
            torch.cat([
                tmp_latent_obs, 
                torch.flatten(tmp_trajs_obs[:,0,:,:], start_dim=1),
                ], dim=1),
            steps=self._config["forward_step"])

        self._info["latent_obs"] = tmp_latent_obs
        self._info["latent_goal"] = tmp_latent_goal
        self._info["planner"] = chain0[-1]
        self._info["critic_1"] = chain1[-1][:,0,0]
        self._info["critic_2"] = chain2[-1][:,0,0]
        self._info["critic_max"] = torch.maximum(chain1[-1][:,0,0], chain2[-1][:,0,0])
        self._info["indices"] = tmp_indices
        self._info["traj_obs"] = tmp_trajs_obs
        self._info["traj_act"] = chain3[-1]
        return chain3[-1]

class GCRLAgent6(GCRLAgentBase):
    def __init__(self,
            config: Dict,
            encoder_config: Tuple[str,str,int,int],
            size_act: int,
            use_rl: bool,
            device: torch.device = None,
        ):
        super().__init__(
            config=config,
            encoder_config=encoder_config,
            size_act=size_act,
            device=device,
        )
        self._use_rl = use_rl
        self._model_critic_1 = self._create_model_flow(
            name="mlp", 
            size_channel=1,
            size_length=1,
            size_cond=2*self._size_latent_obs)
        self._model_critic_2 = self._create_model_flow(
            name="mlp", 
            size_channel=1,
            size_length=1,
            size_cond=2*self._size_latent_obs)
        self._model_actor = self._create_model_flow(
            name="unet", 
            size_channel=self._size_act,
            size_length=self._config["trajs_act_len"],
            size_cond=self._size_latent_obs)
        self._model_world = self._create_model_flow(
            name=self._config["model_planner_name"], 
            size_channel=self._size_latent_obs,
            size_length=self._config["trajs_obs_len"],
            size_cond=self._size_latent_obs+self._config["trajs_act_len"]*self._size_act)
        self._init_training()
    def train(self,
            batch_obs: Dict[str, torch.Tensor],
            batch_goal: Dict[str,torch.Tensor],
            batch_dist: torch.Tensor,
            batch_traj_obs: Dict[str,torch.Tensor],
            batch_traj_act: torch.Tensor,
            batch_other_obs: Dict[str,torch.Tensor],
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        tmp_latent_goal = self.encode_observations(
            batch_goal,
            is_eval=False,
            use_ema=False,
            is_traj=False,
        )
        tmp_latent_traj_obs = self.encode_observations(
            batch_traj_obs,
            is_eval=False,
            use_ema=False,
            is_traj=True,
        )
        size_batch = tmp_latent_obs.size(0)
        if self._use_rl:
            with torch.no_grad():
                tmp_latent_other = self.encode_observations(
                    batch_other_obs,
                    is_eval=True,
                    use_ema=True,
                    is_traj=False,
                )
                ratio = self._config["critic_ratio_training"]
                self._ema_critic_1.eval()
                chain1 = self._flow.transport_forward(
                    self._ema_critic_1,
                    self._generate_dist_critic_src(size_batch, ratio=ratio),
                    torch.cat([
                        tmp_latent_goal, 
                        tmp_latent_other,
                        ], dim=1),
                    steps=self._config["forward_step"])
                self._ema_critic_2.eval()
                chain2 = self._flow.transport_forward(
                    self._ema_critic_2,
                    self._generate_dist_critic_src(size_batch, ratio=ratio),
                    torch.cat([
                        tmp_latent_goal, 
                        tmp_latent_other,
                        ], dim=1),
                    steps=self._config["forward_step"])
                tmp_new_dist = batch_dist + torch.maximum(chain1[-1][:,0,0], chain2[-1][:,0,0])
            tmp_latent_obs = torch.cat([
                tmp_latent_obs,
                tmp_latent_obs,
                ], dim=0)
            tmp_latent_goal = torch.cat([
                tmp_latent_goal,
                tmp_latent_other,
                ], dim=0)
            tmp_latent_traj_obs = torch.cat([
                tmp_latent_traj_obs,
                tmp_latent_traj_obs,
                ], dim=0)
            batch_dist = torch.cat([
                batch_dist,
                tmp_new_dist,
                ], dim=0)
            batch_traj_act = torch.cat([
                batch_traj_act,
                batch_traj_act,
                ], dim=0)
        if self._config["size_augment_with_reached"] > 0:
            size_augment = self._config["size_augment_with_reached"]
            tmp_latent_goal = torch.cat([
                tmp_latent_goal,
                tmp_latent_obs[0:size_augment],
                ], dim=0)
            tmp_latent_obs = torch.cat([
                tmp_latent_obs,
                tmp_latent_obs[0:size_augment],
                ], dim=0)
            tmp_latent_traj_obs = torch.cat([
                tmp_latent_traj_obs,
                tmp_latent_traj_obs[0:size_augment,0:1,:].repeat(1,self._config["trajs_obs_len"],1),
                ], dim=0)
            batch_dist = torch.cat([
                batch_dist,
                torch.zeros_like(batch_dist[0:size_augment]),
                ], dim=0)
            batch_traj_act = torch.cat([
                batch_traj_act,
                batch_traj_act[0:size_augment,0:1,:].repeat(1,self._config["trajs_act_len"],1),
                ], dim=0)
        size_batch = tmp_latent_obs.size(0)
        self._model_critic_1.train()
        loss_critic_1 = self._flow.train_loss(
            self._model_critic_1,
            self._generate_dist_critic_src(size_batch),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            batch_dist.unsqueeze(1).unsqueeze(2),
        )
        self._model_critic_2.train()
        loss_critic_2 = self._flow.train_loss(
            self._model_critic_2,
            self._generate_dist_critic_src(size_batch),
            torch.cat([
                tmp_latent_obs, 
                tmp_latent_goal,
                ], dim=1),
            batch_dist.unsqueeze(1).unsqueeze(2),
        )
        self._model_actor.train()
        loss_actor = self._flow.train_loss(
            self._model_actor,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            tmp_latent_obs, 
            batch_traj_act,
        )
        self._model_world.train()
        loss_world = self._flow.train_loss(
            self._model_world,
            self._generate_dist_trajs_src(
                size_batch, 
                self._config["trajs_obs_len"],
                self._size_latent_obs,
            ),
            torch.cat([
                tmp_latent_obs, 
                torch.flatten(batch_traj_act, start_dim=1),
                ], dim=1),
            tmp_latent_traj_obs,
        )
        loss = 0.1*loss_critic_1 + 0.1*loss_critic_2 + loss_actor + loss_world
        self._update_train_from_loss(loss)
        return loss.detach().cpu().item()
    @torch.no_grad()
    def inference(self, 
            batch_obs: torch.Tensor, 
            batch_goal: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict]:
        tmp_latent_obs = self.encode_observations(
            batch_obs,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        tmp_latent_goal = self.encode_observations(
            batch_goal,
            is_eval=True,
            use_ema=True,
            is_traj=False,
        )
        size_batch = tmp_latent_obs.size(0)
        size_sample = self._config["inference_sampling_size"]
        self._ema_actor.eval()
        chain0 = self._flow.transport_forward(
            self._ema_actor,
            self._generate_dist_trajs_src(
                size_batch*size_sample,
                self._config["trajs_act_len"],
                self._size_act, 
            ),
            tmp_latent_obs.repeat_interleave(size_sample,dim=0),
            steps=self._config["forward_step"])

        self._ema_world.eval()
        chain3 = self._flow.transport_forward(
            self._ema_world,
            self._generate_dist_trajs_src(
                size_batch*size_sample,
                self._config["trajs_obs_len"],
                self._size_latent_obs,
            ),
            torch.cat([
                tmp_latent_obs.repeat_interleave(size_sample,dim=0), 
                torch.flatten(chain0[-1], start_dim=1),
                ], dim=1),
            steps=self._config["forward_step"])

        ratio = self._config["critic_ratio_inference"]
        self._ema_critic_1.eval()
        chain1 = self._flow.transport_forward(
            self._ema_critic_1,
            self._generate_dist_critic_src(size_batch*size_sample, ratio=ratio),
            torch.cat([
                chain3[-1][:,-1,:],
                tmp_latent_goal.repeat_interleave(size_sample,dim=0), 
                ], dim=1),
            steps=self._config["forward_step"])
        self._ema_critic_2.eval()
        chain2 = self._flow.transport_forward(
            self._ema_critic_2,
            self._generate_dist_critic_src(size_batch*size_sample, ratio=ratio),
            torch.cat([
                chain3[-1][:,-1,:],
                tmp_latent_goal.repeat_interleave(size_sample,dim=0), 
                ], dim=1),
            steps=self._config["forward_step"])

        tmp_dist = torch.maximum(chain1[-1][:,0,0], chain2[-1][:,0,0])
        tmp_dist = tmp_dist.reshape(size_batch, size_sample)
        tmp_indices = torch.min(tmp_dist, dim=1, keepdim=False)[1]
        tmp_trajs_act = chain0[-1].reshape(size_batch, size_sample, self._config["trajs_act_len"], self._size_act)
        tmp_trajs_act = (tmp_trajs_act[:,tmp_indices,:,:])[:,0,:,:]

        self._info["latent_obs"] = tmp_latent_obs
        self._info["latent_goal"] = tmp_latent_goal
        self._info["actor"] = chain0[-1]
        self._info["world"] = chain3[-1]
        self._info["critic_1"] = chain1[-1][:,0,0]
        self._info["critic_2"] = chain2[-1][:,0,0]
        self._info["critic_max"] = torch.maximum(chain1[-1][:,0,0], chain2[-1][:,0,0])
        self._info["indices"] = tmp_indices
        self._info["traj_act"] = tmp_trajs_act
        return tmp_trajs_act

