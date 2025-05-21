import os
import io
import time
import yaml
import subprocess
import torch
import numpy as np
import matplotlib

from threading import Thread, Lock
from time import sleep
from utils import check_type
from utils import plotting_matplotlib

def _logger_main_writer(logger):
    """Main for writer thread. Regularly flush logger data on disk."""

    while logger._is_continue:
        logger.flush()
        sleep(10.0)

class LoggerWriter():
    """Logger implementation for writing logs of training jobs"""

    def __init__(self, path, overwrite=False):
        """Initialization with new output path directory.
        overwrite: if False, the output path must be non existing."""

        #Check path
        assert isinstance(path, str)
        if not overwrite and (os.path.isdir(path) or os.path.isfile(path)):
            raise IOError("Path already exists: " + path)
        if os.path.isfile(path):
            raise IOError("Path is a file: " + path)
        #Add trailing slash if needed
        self._path = os.path.join(path, "")

        #Lazy initialization
        self._is_init = False

        #Data container
        self._buffer_scalars = {}
        self._last_epoch = None
        self._time_start = time.perf_counter()

        #Output stream
        self._stream = io.StringIO()

        #Start writer thread
        self._is_continue = True
        self._lock = Lock()
        self._thread = Thread(target=_logger_main_writer, args=[self])
        self._thread.daemon = True
        self._thread.start()

    def get_output_path(self):
        """Return the path to output directory"""
        return self._path

    def _lazy_init(self):
        """Initialize the output folder"""

        if self._is_init:
            return
        self._is_init = True

        #Create output folder
        if not os.path.isdir(self._path):
            #Create output path
            os.mkdir(self._path)

    def set_parameters(self, params):
        """Write dict parameters as YAML"""

        #Check input
        assert self._is_continue
        assert isinstance(params, dict)
        #Write parameter as YAML format
        self._lazy_init()
        with open(self._path+"params.yml", "w") as f:
            yaml.dump(params, f, default_flow_style=False)

    def add_model(self, name, model, id_epoch=None):
        """Write given named torch model for given optional epoch"""

        #Check inputs
        assert isinstance(name, str)
        assert isinstance(model, torch.nn.Module)
        assert id_epoch is None or isinstance(id_epoch, int)
            
        #Overwrite model file
        self._lazy_init()
        torch.save(model.state_dict(), self._path+name+".params")
        torch.save(model, self._path+name+".object")
        #Optionally save intermediate checkpoint
        path_models = self._path + "models/"
        if id_epoch is not None:
            if not os.path.isdir(path_models):
                os.mkdir(path_models)
            torch.save(
                model.state_dict(), 
                path_models + name + "." + str(id_epoch).zfill(8) + ".params")
            torch.save(
                model, 
                path_models + name + "." + str(id_epoch).zfill(8) + ".object")
    
    def add_state_dict(self, name, state, id_epoch=None):
        """Write given torch state dict for given optional epoch"""

        #Check inputs
        assert isinstance(name, str)
        assert isinstance(state, dict) or isinstance(state, OrderedDict)
        assert id_epoch is None or isinstance(id_epoch, int)
            
        #Overwrite model file
        self._lazy_init()
        torch.save(state, self._path+name+".params")
        #Optionally save intermediate checkpoint
        path_models = self._path + "models/"
        if id_epoch is not None:
            if not os.path.isdir(path_models):
                os.mkdir(path_models)
            torch.save(
                state, 
                path_models + name + "." + str(id_epoch).zfill(8) + ".params")
    
    def add_script(self, name, script, id_epoch=None):
        """Write given named torch script model for given optional epoch"""

        #Check inputs
        assert isinstance(name, str)
        assert isinstance(script, torch.jit.ScriptModule)
        assert id_epoch is None or isinstance(id_epoch, int)
            
        #Overwrite script file
        self._lazy_init()
        script.save(self._path+name+".pt")
        #Optionally save intermediate checkpoint
        path_scripts = self._path + "scripts/"
        if id_epoch is not None:
            if not os.path.isdir(path_scripts):
                os.mkdir(path_scripts)
            script.save(
                path_scripts + name + "." + str(id_epoch).zfill(8) + ".pt")

    def add_figure(self, name, fig, id_epoch=None):
        """Write given named matplotlib figure for given optional epoch"""
        
        #Check inputs
        assert isinstance(name, str)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert id_epoch is None or isinstance(id_epoch, int)
    
        #Overwrite figure file
        self._lazy_init()
        plotting_matplotlib.save_raw_figure(
            self._path + name + ".fig_object", fig)
        fig.savefig(self._path + name + ".png")
        #Optionally save intermediate checkpoint
        path_figures = self._path + "figures/"
        if id_epoch is not None:
            if not os.path.isdir(path_figures):
                os.mkdir(path_figures)
            plotting_matplotlib.save_raw_figure(
                path_figures + name + "." + str(id_epoch).zfill(8) + ".fig_object", fig)
            fig.savefig(
                path_figures + name + "." + str(id_epoch).zfill(8) + ".png")
        #Close the figure to clean up memory
        fig.clear()
        matplotlib.pyplot.close(fig)
    
    def add_frame_as_fig(self, name, fig, index):
        """Write given named matplotlib figure as image for given index"""
        
        #Check inputs
        assert isinstance(name, str)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert index is None or isinstance(index, int)
    
        #Overwrite figure file
        self._lazy_init()
        path_frames = self._path + "frames/"
        if index is not None:
            if not os.path.isdir(path_frames):
                os.mkdir(path_frames)
            fig.savefig(
                path_frames + name + "." + str(index).zfill(8) + ".png")
        #Close the figure to clean up memory
        fig.clear()
        matplotlib.pyplot.close(fig)

    def convert_frames_to_video(self, name, framerate, is_figure=False):
        """Convert previously generated frames or figures into video with ffmpeg.
        name: Name given to add_frame.
        framerate: Number of frames per seconds.
        is_figure: if True, the name is assumed to be a figure, else a frame."""
        
        if is_figure:
            path_dir = self._path + "figures/"
        else:
            path_dir = self._path + "frames/"
        assert isinstance(name, str)
        assert isinstance(framerate, int)
        assert os.path.isdir(path_dir)
        subprocess.run([
            "ffmpeg", "-nostdin", "-loglevel", "error", 
            "-framerate", str(framerate), 
            "-pattern_type", "glob", "-i", path_dir+name+".*.png", 
            "-c:v", "libx264", "-profile:v", "high", "-crf", "20", "-pix_fmt", "yuv420p", 
            self._path+name+".mp4"], check=True)

    def add_scalar(self, name, value, id_epoch, id_batch=None):
        """Append scalar value with given name and associated 
        epoch and optionally batch id"""

        assert self._is_continue
        assert isinstance(name, str)
        assert isinstance(id_epoch, int) and id_epoch >= 0
        assert id_batch is None or (isinstance(id_batch, int) and id_batch >= 0)
        assert self._last_epoch is None or self._last_epoch <= id_epoch
        if id_batch is None:
            id_batch = -1

        with self._lock:
            if not name in self._buffer_scalars:
                self._buffer_scalars[name] = []
            self._buffer_scalars[name].append([
                id_epoch, id_batch, 
                time.perf_counter()-self._time_start, 
                check_type.get_scalar(value)
            ])
            self._last_epoch = id_epoch

    def add_print(self, s):
        """Add given string as output stream (with trailing end line)"""

        assert isinstance(s, str)
        self._stream.write(s+"\n")

    def flush(self):
        """Write buffered data to disk"""

        #Write output stream
        if len(self._stream.getvalue()) > 0:
            self._lazy_init()
            with open(self._path+"out", "w") as f:
                f.write(self._stream.getvalue())
        
        #Write scalars
        if len(self._buffer_scalars) > 0:
            dict_arrays = {}
            last_epoch = None
            with self._lock:
                #Convert list to numpy arrays
                for key, array in self._buffer_scalars.items():
                    dict_arrays[key] = np.array(array)
                last_epoch = self._last_epoch
            
            #Write as numpy compressed format
            self._lazy_init()
            np.savez_compressed(self._path+"scalars.npz", **dict_arrays)
            
            #Write last as YAML
            dict_last = {}
            dict_last["epoch"] = last_epoch
            for key, array in dict_arrays.items():
                dict_last[key] = float(array[-1,3])
            with open(self._path+"scalars.yml", "w") as f:
                yaml.dump(dict_last, f, default_flow_style=False)

    def close(self):
        """Stop writer thread and flush data"""

        self._is_continue = False
        self.flush()

class LoggerReader():
    """Logger implementation for reading logs of training jobs"""

    def __init__(self, path):
        """Initialization with read only logs directory."""

        #Check path
        assert isinstance(path, str)
        if not os.path.isdir(path):
            raise IOError("Path does not exists: " + path)
        #Add trailing slash if needed
        self._path = os.path.join(path, "")

    def get_parameters(self):
        """Return the dict of parameters"""

        if os.path.isfile(self._path+"params.yml"):
            with open(self._path+"params.yml", "r") as f:
                dict_params = yaml.safe_load(f)
                return dict_params
        else:
            return dict()

    def get_models(self):
        """Return the dict of models loaded with pickle"""

        #Retrieve model names from filenames
        names = [s[:-len(".model_object")] for s in os.listdir(self._path) \
            if s.endswith(".model_object")]
        #Load models with Pickle
        dict_models = {}
        for name in names:
            dict_models[name] = model = torch.load(self._path + name + ".model_object")
        return dict_models

    def get_scalars(self):
        """Return the dict of arrays of scalars"""
        
        if os.path.isfile(self._path+"scalars.npz"):
            with np.load(self._path+"scalars.npz") as data:
                dict_arrays = {}
                for name in data:
                    dict_arrays[name] = data[name]
                return dict_arrays
        else:
            return dict()

    def get_figures(self):
        """Return the dict of list of figures"""

        #Retrieve list of figure names
        names = [s[:-len(".fig_object")] for s in os.listdir(self._path) \
            if s.endswith(".fig_object")]

        dict_figures = {}
        for name in names:
            if not os.path.isdir(self._path+"figures/"):
                break
            tmp_names = [s for s in os.listdir(self._path+"figures/") if s.startswith(name) and s.endswith(".fig_object")]
            tmp_names.sort()
            print(tmp_names)
            dict_figures[name] = []
            for tmp_name in tmp_names:
                dict_figures[name].append(plotting_matplotlib.load_raw_figure(self._path+"figures/"+tmp_name))
        return dict_figures

