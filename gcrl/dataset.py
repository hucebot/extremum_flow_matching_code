import numpy as np
import torch
from collections import OrderedDict

from typing import List, Tuple, Dict, Union

class GCTrajsDataset():
    """Dataset for goal-conditioned learning with trajectory sampling"""

    def __init__(self, 
            dict_obs: Dict[str, np.ndarray],
            array_act: np.ndarray,
            array_terminal: np.ndarray,
            trajs_obs_len: int,
            trajs_act_len: int,
            trajs_obs_stride: int,
            trajs_act_stride: int,
            max_dist_goal: int,
            is_act_vel: bool = True,
            device: torch.device = None,
        ):
        """Initialization.
        dict_obs: Dictionary of observations array (size_data, ...).
        array_act: Low dimensional actions (size_data, size_act).
        array_terminal: Indicator of each episode end (1 at last time step, and 0 elsewhere) (size_data).
        trajs_obs_len: Length of future observations trajectory to sample.
        trajs_act_len: Length of future actions trajectory to sample.
        trajs_obs_stride: Sub-sampling stride of future observations trajectory to sample.
        trajs_act_stride: Sub-sampling stride of future actions trajectory to sample.
        max_dist_goal: Maximum sampled number of steps between observation and goal in the same episode.
        is_act_vel: If True, actions are assumed to be velocity, 
            and short action trajectories are clip to zero.
        device: Optional allocation device for dataset.
        """
        #Sanity checks
        assert trajs_obs_len >= 1 and trajs_act_len >= 1
        assert trajs_obs_stride >= 1 and trajs_act_stride >= 1
        assert max_dist_goal >= 1
        for label,array in dict_obs.items():
            assert len(array.shape) >= 2
            assert array.shape[0] == array_act.shape[0]
        assert len(array_act.shape) == 2
        assert len(array_terminal.shape) == 1
        assert array_act.shape[0] == array_terminal.shape[0]
        self._trajs_obs_len = trajs_obs_len
        self._trajs_act_len = trajs_act_len
        self._trajs_obs_stride = trajs_obs_stride
        self._trajs_act_stride = trajs_act_stride
        self._max_dist_goal = max_dist_goal
        self._is_act_vel = is_act_vel
        #Load data into tensors
        self._obs = OrderedDict()
        for label,array in dict_obs.items():
            if array.dtype == np.float32 or array.dtype == np.float64:
                self._obs[label] = torch.tensor(array, dtype=torch.float32, device=device)
            elif array.dtype == np.uint8:
                self._obs[label] = torch.tensor(array, dtype=torch.uint8, device=device)
            else:
                raise IOError("dtype not supported: " + str(array.dtype))
        self._act = torch.tensor(array_act, dtype=torch.float32, device=device)
        #Split data into individual episodes and extract begin and end indices (size_episode)
        tmp_terminal = torch.tensor(array_terminal, dtype=torch.bool, device=device)
        self._range_episodes_end = torch.nonzero(tmp_terminal)[:,0]
        self._range_episodes_begin = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=device),
            self._range_episodes_end[:-1]+1,
            ], dim=0)
        #Associate each data point to the index of 
        #the end of the associated episode (size_data)
        tmp_sum_index_episode = torch.cumsum(tmp_terminal, dim=0)
        tmp_sum_index_episode[-1] -= 1
        self._indices_episodes_end = self._range_episodes_end[tmp_sum_index_episode]
        #Create the list of all indices that are valid for sampling observations. 
        #They must be at least length of trajectories away from the end of the episode.
        tmp_max_length = max(
            self._trajs_obs_len*self._trajs_obs_stride, 
            self._trajs_act_len*self._trajs_act_stride)
        tmp_indices_not_to_sample = self._range_episodes_end.unsqueeze(-1).repeat(1,tmp_max_length)
        tmp_indices_not_to_sample -= torch.arange(0, tmp_max_length, 1, device=device).unsqueeze(0)
        tmp_indices_not_to_sample = tmp_indices_not_to_sample.flatten()
        tmp_mask_valid_to_sample = torch.ones(tmp_terminal.size(0), dtype=torch.bool, device=device)
        tmp_mask_valid_to_sample[tmp_indices_not_to_sample] = False
        self._indices_valid_to_sample = torch.nonzero(tmp_mask_valid_to_sample)[:,0]

    def count_episodes(self) -> int:
        """Return the number of episode sequences stored in the dataset"""
        return self._range_episodes_begin.size(0)

    def get_episode(self, 
            index: int,
        ) -> Tuple[Dict[str,torch.Tensor], torch.Tensor]:
        """Return the dict of observations and actions 
        for given episode index
        """
        assert index >= 0 and index < self.count_episodes()
        dict_obs = OrderedDict()
        for label,tensor in self._obs.items():
            dict_obs[label] = tensor[self._range_episodes_begin[index]:self._range_episodes_end[index]+1]
        tmp_act = self._act[self._range_episodes_begin[index]:self._range_episodes_end[index]+1]
        return dict_obs, tmp_act

    def sample_obs(self, 
            size_batch: int,
            device: torch.device = None,
        ) -> Dict[str, torch.Tensor]:
        """Uniformly sample a batch of observations from the whole dataset dict of (size_batch,...)"""
        tmp_perm = torch.randperm(self._act.size(0))
        dict_obs = OrderedDict()
        for label,tensor in self._obs.items():
            dict_obs[label] = (tensor[tmp_perm][0:size_batch]).to(device)
        return dict_obs

    def sample(self, 
            size_batch: int,
        ) -> Tuple[
            Dict[str,torch.Tensor], 
            Dict[str,torch.Tensor], 
            torch.Tensor, 
            Dict[str,torch.Tensor], 
            torch.Tensor]:
        """Sample data tuple for training.
        Args:
            size_batch: Size of the batch to sample.
        Return the tuple
            batched observations dict of (size_batch, ...).
            batched goal observations dict of (size_batch, ...).
            batched time step distance from observation to goal (size_batch).
            batched observation trajectories dict (size_batch, trajs_obs_len, ...).
            batched action trajectories (size_batch, trajs_act_len, size_act).
        """
        #Sample observations from valid indices
        tmp_indices_obs = self._indices_valid_to_sample[torch.randperm(self._indices_valid_to_sample.size(0))][0:size_batch]
        batch_obs = OrderedDict()
        for label,tensor in self._obs.items():
            batch_obs[label] = tensor[tmp_indices_obs]
        #Sample goal between observation indices and end of episode clipped by max_length
        tmp_indices_end = self._indices_episodes_end[tmp_indices_obs]
        tmp_indices_end = torch.min(tmp_indices_end, tmp_indices_obs+self._max_dist_goal)
        tmp_indices_begin = tmp_indices_obs
        tmp_indices_goal = torch.randint(2**63 - 1, size=(size_batch,), device=self._act.device) \
            % (tmp_indices_end - tmp_indices_begin) + tmp_indices_begin
        batch_dist = tmp_indices_goal - tmp_indices_obs
        batch_goal = OrderedDict()
        for label,tensor in self._obs.items():
            batch_goal[label] = tensor[tmp_indices_obs + batch_dist]
        #Create indices used to sample observation trajectory and clamped indices 
        #such that the same observation is repeated when goal is reached
        tmp_indices_traj_obs = torch.arange(
            0, self._trajs_obs_len*self._trajs_obs_stride, self._trajs_obs_stride, 
            device=self._act.device).unsqueeze(0).repeat(size_batch,1)
        tmp_indices_traj_obs = torch.clip(tmp_indices_traj_obs, max=batch_dist.unsqueeze(1))
        tmp_indices_traj_obs = tmp_indices_obs.unsqueeze(1).repeat(1,self._trajs_obs_len) + tmp_indices_traj_obs
        tmp_indices_traj_obs = tmp_indices_traj_obs.reshape(size_batch*self._trajs_obs_len)
        #Create indices used to sample action trajectories
        tmp_indices_traj_act = torch.arange(
            0, self._trajs_act_len*self._trajs_act_stride, self._trajs_act_stride, 
            device=self._act.device).unsqueeze(0).repeat(size_batch,1)
        tmp_indices_traj_act_copy = tmp_indices_traj_act
        if not self._is_act_vel:
            tmp_indices_traj_act = torch.clip(tmp_indices_traj_act, max=batch_dist.unsqueeze(1))
        tmp_indices_traj_act = tmp_indices_obs.unsqueeze(1).repeat(1,self._trajs_act_len) + tmp_indices_traj_act
        tmp_indices_traj_act = tmp_indices_traj_act.reshape(size_batch*self._trajs_act_len)
        #Sample action trajectories starting from observation indices
        batch_traj_obs = OrderedDict()
        for label,tensor in self._obs.items():
            batch_traj_obs[label] = tensor[tmp_indices_traj_obs].reshape(size_batch, self._trajs_obs_len, *tensor.size()[1:])
        batch_traj_act = self._act[tmp_indices_traj_act].reshape(size_batch, self._trajs_act_len, *self._act.size()[1:])
        #Set action to zero when the goal is reached assuming velocity command
        if self._is_act_vel:
            batch_traj_act = torch.where(
                tmp_indices_traj_act_copy.unsqueeze(2) < batch_dist.unsqueeze(1).unsqueeze(2),
                batch_traj_act, 
                torch.zeros_like(batch_traj_act))

        return \
            batch_obs, \
            batch_goal, \
            batch_dist/self._max_dist_goal, \
            batch_traj_obs, \
            batch_traj_act

