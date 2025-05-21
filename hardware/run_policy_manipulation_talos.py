import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PointStamped
import numpy as np
np.float = np.float64
import ros_numpy
from collections import OrderedDict

import time
import math
import h5py
import random
import cv2
import torch
import copy
import threading
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Dict, Union

from gcrl.deployment import PolicyInference

#Device configuration
if True:
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

#Load policy
trajs_act_stride = 3
params_dt = 0.035
policy = PolicyInference(
    model_policy="/tmp/docker_share/agent1_norl.gpu.pt",
    timestep_prediction=params_dt*trajs_act_stride,
    cutoff_period=0.6,
    device=device)
policy.reset()

#Camera color image observation callbacks
image_obs = None
image_goal = None
cmd_pos_obs = None
cmd_quat_obs = None
cmd_pos_goal = None
cmd_quat_goal = None
read_pos_obs = None
read_quat_obs = None
read_pos_goal = None
read_quat_goal = None
def callback_image(msg):
    global image_obs, image_goal
    image_obs = ros_numpy.numpify(msg)
    if image_goal is None:
        image_goal = image_obs.copy()
def callback_pose_cmd(msg):
    global cmd_pos_obs, cmd_quat_obs, cmd_pos_goal, cmd_quat_goal
    cmd_pos_obs = np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
        ], dtype=np.float32)
    cmd_quat_obs = np.array([
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
        ], dtype=np.float32)
    if cmd_pos_goal is None or cmd_quat_goal is None:
        cmd_pos_goal = cmd_pos_obs.copy()
        cmd_quat_goal = cmd_quat_obs.copy()
def callback_pose_read(msg):
    global read_pos_obs, read_quat_obs, read_pos_goal, read_quat_goal
    read_pos_obs = np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
        ], dtype=np.float32)
    read_quat_obs = np.array([
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
        ], dtype=np.float32)
    if read_pos_goal is None or read_quat_goal is None:
        read_pos_goal = read_pos_obs.copy()
        read_quat_goal = read_quat_obs.copy()

#Initialize ROS node and subscribers
rospy.init_node("manipulation_policy", anonymous=True)
rospy.set_param("/use_sim_time", False)
sub_image = rospy.Subscriber("/camera/color/image_raw", Image, callback_image)
sub_pose_cmd = rospy.Subscriber("/talos_controller/cmd_hand_left_pose", PoseStamped, callback_pose_cmd)
sub_pose_read = rospy.Subscriber("/talos_controller/read_hand_left_pose", PoseStamped, callback_pose_read)
#Initialize publisher
pub_pose = rospy.Publisher("/talos_controller/abs_pose_cmd", PoseStamped, queue_size=10)
pub_gripper = rospy.Publisher("/talos_controller/abs_gripper_cmd", PointStamped, queue_size=10)

#Main loop
t = 0.0
is_publish = False
is_warmup = True
last_time = time.time()
last_time_inference = -1.0
rate = rospy.Rate(1.0/params_dt)
print("Starting loop")
while not rospy.is_shutdown():
    #Scheduling
    rate.sleep()
    now_time = time.time()
    real_dt = now_time - last_time
    if image_goal is None or cmd_pos_goal is None or read_pos_goal is None:
        continue
    #Display camera observation
    cv2.imshow("Observation", 
        cv2.resize(cv2.cvtColor(image_obs, cv2.COLOR_RGB2BGR), 
        dsize=(0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("Goal", 
        cv2.resize(cv2.cvtColor(image_goal, cv2.COLOR_RGB2BGR), 
        dsize=(0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_NEAREST))
    key = cv2.waitKey(1)
    #Policy asynchronous inference
    if last_time_inference < 0.0 or t-last_time_inference >= 0.8:
        dict_obs = OrderedDict()
        dict_goal = OrderedDict()
        dict_obs["images"] = torch.tensor(image_obs, dtype=torch.uint8, device=device)
        dict_obs["cmd_pos"] = torch.tensor(cmd_pos_obs, dtype=torch.float32, device=device)
        dict_obs["cmd_quat"] = torch.tensor(cmd_quat_obs, dtype=torch.float32, device=device)
        dict_obs["read_pos"] = torch.tensor(read_pos_obs, dtype=torch.float32, device=device)
        dict_obs["read_quat"] = torch.tensor(read_quat_obs, dtype=torch.float32, device=device)
        dict_goal["images"] = torch.tensor(image_goal, dtype=torch.uint8, device=device)
        dict_goal["cmd_pos"] = torch.tensor(cmd_pos_goal, dtype=torch.float32, device=device)
        dict_goal["cmd_quat"] = torch.tensor(cmd_quat_goal, dtype=torch.float32, device=device)
        dict_goal["read_pos"] = torch.tensor(read_pos_goal, dtype=torch.float32, device=device)
        dict_goal["read_quat"] = torch.tensor(read_quat_goal, dtype=torch.float32, device=device)
        print("compute inference t={:.3f}".format(t))
        is_async = not is_warmup
        policy.compute_inference_policy(
            t, is_async,
            dict_obs, dict_goal)
        last_time_inference = t
        is_warmup = False
    #Interpolate command at current time from computed trajectories
    pred_ts, pred_interpolated, pred_out = policy.interpolate_output(t, ["cmd_pos", "cmd_quat", "cmd_gripper"])
    #Verbose and data log
    print("is_publish={} time={:.3f} real_dt={:.3f} t={:.3f} last_inference={:.3f}".format(
        is_publish, now_time, real_dt, t, t-pred_ts))
    #Process keyboard input
    #Quit 'escape' or 'q'
    if key == 113 or key == 27:
        break
    #Set goal frame 'g'
    if key == 103:
        image_goal = image_obs.copy()
        cmd_pos_goal = cmd_pos_obs.copy()
        cmd_quat_goal = cmd_quat_obs.copy()
        read_pos_goal = read_pos_obs.copy()
        read_quat_goal = read_quat_obs.copy()
        print("Reset goal", cmd_pos_goal, cmd_quat_goal, read_pos_goal, read_quat_goal)
    #Publish mode 'm'
    if key == 109:
        is_publish = (True if not is_publish else False)
    #Publish command message
    if is_publish:
        msg_cmd_pose = PoseStamped()
        msg_cmd_pose.pose.position.x = pred_interpolated["cmd_pos"][0]
        msg_cmd_pose.pose.position.y = pred_interpolated["cmd_pos"][1]
        msg_cmd_pose.pose.position.z = pred_interpolated["cmd_pos"][2]
        msg_cmd_pose.pose.orientation.x = pred_interpolated["cmd_quat"][0]
        msg_cmd_pose.pose.orientation.y = pred_interpolated["cmd_quat"][1]
        msg_cmd_pose.pose.orientation.z = pred_interpolated["cmd_quat"][2]
        msg_cmd_pose.pose.orientation.w = pred_interpolated["cmd_quat"][3]
        pub_pose.publish(msg_cmd_pose)
        msg_cmd_gripper = PointStamped()
        msg_cmd_gripper.point.x = pred_interpolated["cmd_gripper"][0]
        msg_cmd_gripper.point.y = 0.0
        msg_cmd_gripper.point.z = 0.0
        pub_gripper.publish(msg_cmd_gripper)
    else:
        policy.reset_filter({
            "cmd_pos": dict_obs["cmd_pos"],
            "cmd_quat": dict_obs["cmd_quat"],
            "cmd_gripper": torch.tensor([0.0]),
            "read_pos": dict_obs["read_pos"],
            "read_quat": dict_obs["read_quat"],
        })
        msg_cmd_pose = PoseStamped()
        msg_cmd_pose.pose.position.x = cmd_pos_obs[0]
        msg_cmd_pose.pose.position.y = cmd_pos_obs[1]
        msg_cmd_pose.pose.position.z = cmd_pos_obs[2]
        msg_cmd_pose.pose.orientation.x = cmd_quat_obs[0]
        msg_cmd_pose.pose.orientation.y = cmd_quat_obs[1]
        msg_cmd_pose.pose.orientation.z = cmd_quat_obs[2]
        msg_cmd_pose.pose.orientation.w = cmd_quat_obs[3]
        pub_pose.publish(msg_cmd_pose)
        msg_cmd_gripper = PointStamped()
        msg_cmd_gripper.point.x = 0.0
        msg_cmd_gripper.point.y = 0.0
        msg_cmd_gripper.point.z = 0.0
        pub_gripper.publish(msg_cmd_gripper)
    #Update state
    t += min(real_dt, 0.5)
    last_time = now_time

