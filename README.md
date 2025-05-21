# Extremum Flow Matching for<br>Offline Goal Conditioned Reinforcement Learning

**Website**: https://hucebot.github.io/extremum_flow_matching_website/ \
**Authors**: [Quentin Rouxel](https://leph.io/quentinrouxel/),  [Clemente Donoso](#), [Fei Chen](https://feichenlab.com/), [Serena Ivaldi](https://members.loria.fr/SIvaldi), [Jean-Baptiste Mouret](https://members.loria.fr/JBMouret) \
**ArXiv**: https://arxiv.org/abs/2505.19717 \
**HAL**: https://hal.science/hal-05080807

![Extremum Flow Matching](/assets/policy_architecture_talos_2.png)

This repository contains the python implementation associated with the paper
*Extremum Flow Matching for Offline Goal Conditioned Reinforcement Learning*, developed at [INRIA Nancy](https://www.inria.fr/en/inria-centre-universite-lorraine) (France), [Larsen Team](https://www.inria.fr/en/larsen), and [CLOVER Lab CUHK](https://feichenlab.com) (Hong Kong) in 2024-2025.

The C++ implementation of [SEIKO whole-body Retargeting and Controller](https://github.com/hucebot/seiko_controller_code) was used in real-world experiments on the Talos humanoid robot.

## Installation with Docker

* Build the Docker image on the host machine. Both Docker and NVIDIA Docker support must be installed, and the local user should belong to the `docker` group. 
```
cd docker/
./build.sh
```
* Start the Docker container on the host machine:
```
cd docker/
./launch.sh
```

## Usage inside the Docker container

* Train and reproduce the comparison between Expectile Regression and Extremum Flow Matching on a 1D conditional distribution:
```
source setup.bash
python3 jobs/compare_expectile_regresion_vs_flow_matching.py
```
* Train and reproduce the plots and animations of Extremum Flow Matching on a 2D distribution:
```
source setup.bash
python3 jobs/test_expectile_flow_matching_2d.py --mode train --use_cuda True
python3 jobs/test_expectile_flow_matching_2d.py --mode anim_baseline --use_cuda False --is_load True
python3 jobs/test_expectile_flow_matching_2d.py --mode anim_model --use_cuda False --is_load True
```
* Train proposed agents and run [OGBench](https://seohong.me/projects/ogbench/) comparisons: 
```
source setup.bash
python3 jobs/run_env_ogbench.py --use_cuda True --mode train --dataset_name pointmaze-large-navigate-v0 --prefix_out data_io/ --label ogbench_pointmaze_large_navigate
python3 jobs/run_env_ogbench.py --use_cuda True --mode train --dataset_name pointmaze-large-stitch-v0 --prefix_out data_io/ --label ogbench_pointmaze_large_stitch
python3 jobs/run_env_ogbench.py --use_cuda True --mode train --dataset_name antmaze-large-navigate-v0 --prefix_out data_io/ --label ogbench_antmaze_large_navigate
python3 jobs/run_env_ogbench.py --use_cuda True --mode train --dataset_name antmaze-large-stitch-v0 --prefix_out data_io/ --label ogbench_antmaze_large_stitch
python3 jobs/run_env_ogbench.py --use_cuda True --mode train --dataset_name cube-double-play-v0 --prefix_out data_io/ --label ogbench_cube_double_play
python3 jobs/run_env_ogbench.py --use_cuda True --mode train --dataset_name puzzle-4x4-play-v0 --prefix_out data_io/ --label ogbench_puzzle_4x4_play
python3 jobs/run_env_ogbench.py --use_cuda True --mode train --dataset_name scene-play-v0 --prefix_out data_io/ --label ogbench_scene_play
```
* Train proposed agents and run comparisons on the different collection behavior datasets using a planar pushing task:
```
source setup.bash
python3 jobs/run_env_planar_push.py --use_cuda True --mode train --path_npz_demo ./data_static/dataset_demo_reach_goal.npz --prefix_out data_io/ --label planar_reach_goal
python3 jobs/run_env_planar_push.py --use_cuda True --mode train --path_npz_demo ./data_static/dataset_demo_play_full.npz --prefix_out data_io/ --label planar_play_full
python3 jobs/run_env_planar_push.py --use_cuda True --mode train --path_npz_demo ./data_static/dataset_demo_play_stitch.npz --prefix_out data_io/ --label planar_play_stitch
```
* Files used to train and run the policy online for the Talos humanoid robot experiment:
```
jobs/run_env_talos_manipulation.py
hardware/run_policy_manipulation_talos.py
```

## License

Licensed under the [BSD License](LICENSE)

