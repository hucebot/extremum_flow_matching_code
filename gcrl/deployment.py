import numpy as np
import torch
import copy
import math
import threading
from collections import OrderedDict
from typing import List, Dict, Union

class PolicyInference():
    """Implement receding horizon replanning using torch script model
    with optional asynchronous threading inference"""
    def __init__(self, 
            model_policy: Union[str, torch.nn.Module],
            timestep_prediction: float,
            cutoff_period: float,
            device: torch.device = None,
        ):
        """Initialization and loading.
        Args:
            model_policy: Either the path toward pytorch script policy model to load,
                or directly the deployment policy model to use.
            timestep_prediction: Timestep period of predicted 
                output trajectories in seconds.
            cutoff_period: Cutoff period for output trajectories lowpass filtering.
            device: Optional allocation device for agent models.
        """
        #Load deployment model flow
        if isinstance(model_policy, torch.nn.Module):
            self._model_policy = model_policy.to(device)
        else:
            self._model_policy = torch.jit.load(model_policy).to(device)
        #Retrieve predicted output Trajectories parameters
        self._timestep = timestep_prediction
        #Initialize predicted output trajectories state
        self._mutex = threading.Lock()
        self._pred_timestamp = None
        self._pred_outputs = None
        #Lowpass predicted output filtering
        self._cmd_timestamp = None
        self._cmd_filtered = OrderedDict()
        self._cutoff_period = cutoff_period
        #Asynchronous inference thread
        self._thread = None

    def reset(self):
        """Reset inference internal state"""
        if self._thread is not None:
            self._thread.join()
        self._pred_timestamp = None
        self._pred_outputs = None
        self._cmd_timestamp = None
        self._cmd_filtered = OrderedDict()
        self._thread = None

    def reset_filter(self, 
            dict_data: Dict[str, torch.Tensor],
        ):
        """Reset only internal filter with given values"""
        self._cmd_filtered = OrderedDict()
        for label in dict_data.keys():
            self._cmd_filtered[label] = copy.deepcopy(dict_data[label]).cpu()
    
    @torch.no_grad()
    def compute_inference_policy(self, 
            timestamp: float,
            is_asynchronous: bool,
            *args,
        ):
        """Generate a new predicted output trajectory at given timestamp and inputs
        Args:
            timestamp: Current timestamp associated to given inputs.
            is_asynchronous: If True, the inference is computed asynchronously in a thread. 
                If False, the inference is blocking while performing the computation.
            args: Input arguments given to the policy model.
        """
        #Inference function
        def tmp_func():
            #Compute model inference
            tmp_pred = self._model_policy(*args)
            #Switch to newly computed trajectories
            self._mutex.acquire()
            self._pred_timestamp = timestamp
            self._pred_outputs = tmp_pred
            self._mutex.release()
        #Warm start initialization
        if self._pred_timestamp is None:
            for k in range(5):
                tmp_func()
        #Wait previously started inference if not finished
        if self._thread is not None:
            self._thread.join()
        if is_asynchronous:
            #Asynchronous inference
            self._thread = threading.Thread(target=tmp_func)
            self._thread.start()
        else:
            #Blocking inference or first iteration
            tmp_func()

    @torch.no_grad()
    def interpolate_output(self, 
            timestamp: float,
            labels: List[str],
        ) -> Dict[str, np.ndarray]:
        """Interpolate predicted output trajectory at 
        given timestamp and return filtered output value.
        Args:
            timestamp: Timestamp at which to interpolate previously previously output trajectory.
            labels: List of label to interpolate within policy output.
        Returns:
            Timestamp at last policy inference used.
            Dict of named interpolated and filtered output value as numpy array (size_channel). 
            Original output dict of the policy.
        """
        #Get a copy of current trajectories
        self._mutex.acquire()
        tmp_ts = copy.deepcopy(self._pred_timestamp)
        tmp_output = copy.deepcopy(self._pred_outputs)
        self._mutex.release()
        #Sanity check
        assert tmp_ts is not None and tmp_output is not None, "Inference not initialized"
        tmp_delta = (timestamp - tmp_ts)/self._timestep
        assert tmp_delta >= 0.0, "Predicted trajectory is in the future"
        #Filtering initialization
        if self._cmd_timestamp is None:
            self._cmd_timestamp = timestamp
            for label in labels:
                self._cmd_filtered[label] = None
        #Compute filtering coefficient
        dt = timestamp - self._cmd_timestamp
        assert dt >= 0.0, "Interpolation timestamp not progressing"
        omega = 2.0*np.pi/self._cutoff_period
        alpha_filter = (1.0-omega*dt/2.0)/(1.0+omega*dt/2.0)
        #Compute interpolation
        for label in labels:
            #Compute linear interpolation between two predicted timesteps
            idx_low = int(math.floor(tmp_delta))
            idx_up = idx_low+1
            tmp_weight = tmp_delta - math.floor(tmp_delta)
            assert tmp_weight >= 0.0 and tmp_weight <= 1.0
            assert idx_up < tmp_output[label].shape[0], "Predicted trajectory is outdated"
            tmp_val = (1.0-tmp_weight)*tmp_output[label][idx_low,:] + tmp_weight*tmp_output[label][idx_up,:]
            #Conversion to numpy
            tmp_val = tmp_val.cpu().numpy()
            #Apply lowpass filtering
            if self._cmd_filtered[label] is None:
                self._cmd_filtered[label] = tmp_val
            self._cmd_filtered[label] = alpha_filter*self._cmd_filtered[label] + (1.0-alpha_filter)*tmp_val
            #Apply quaternion re-normalization after interpolation and filtering
            if label.endswith("_quat"):
                self._cmd_filtered[label] = self._cmd_filtered[label]/np.linalg.norm(self._cmd_filtered[label])
        self._cmd_timestamp = timestamp
        return tmp_ts, self._cmd_filtered, tmp_output

