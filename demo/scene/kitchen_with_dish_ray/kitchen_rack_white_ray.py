import mujoco,cv2,pyvista
import numpy as np
import sys
sys.path.append('../../../')
from utils.mujoco_parser import MuJoCoParserClass
from utils.util import sample_xyzs,rpy2r,r2rpy,r2quat,compute_view_params,get_interp_const_vel_traj, printmd
from pyvirtualdisplay.smartdisplay import SmartDisplay
np.set_printoptions(precision=2,suppress=True,linewidth=100)
print ("MuJoCo version:[%s]"%(mujoco.__version__))
from utils.util import fill_object, get_geom_region_type1, get_geom_region_type2, get_geom_region_type3, get_geom_region_type4, passthrough_filter, remove_duplicates_with_threshold, \
                        r2quat, rpy2r, passthrough_filter, remove_duplicates_with_threshold
import ray
import time
import os
import sys
sys.path.append('../../../')
from utils.mujoco_parser import MuJoCoParserClass
from utils.mujoco_parser_w_ray import MuJoCoParserClassRay

SCREEN_CAPTURE_RESOLUTION = (1200, 800)
virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
virtual_display.start()

xml_path = '../../../asset/scene_kitchen_dish_rack_white.xml'

config = dict()
config['num_cpus'] = 48
config['num_gpus'] = 4
config['file_name'] = "20230914" # config['file_name']

start_time = time.time()
num_cpus = config['num_cpus']
num_gpus = config['num_gpus']
num_envs = num_cpus #config['num_envs']
file_name = config['file_name']

# 0. Set the environment
start_ray = start_time
ray.shutdown()
ray.init(num_cpus=num_cpus, num_gpus=num_gpus, _temp_dir="/home/root/ray/", include_dashboard=False, ignore_reinit_error=True)
print(f"Ray is initialized with {num_cpus} cpus and {num_gpus} gpus with {num_envs} environments")
end_ray = time.time()
print(f"Time taken for Ray initialization: {end_ray - start_ray:.4f} seconds")

# Create remote instances
start_mujoco_envs = time.time()
mujoco_envs = [MuJoCoParserClassRay.remote(name=f'UR5e in Ray idx[{i}]', rel_xml_path=xml_path, VERBOSE=False, USE_MUJOCO_VIEWER=False, MODE='offscreen', env_id=i) for i in range(num_envs)]

idxs_forward = ray.get(mujoco_envs[0].get_idxs.remote())
init_ur_q = np.array([np.deg2rad(-90), np.deg2rad(-130), np.deg2rad(120), np.deg2rad(100), np.deg2rad(45), np.deg2rad(-90)])
for i in range(num_envs):
    mujoco_envs[i].init_set_state.remote(scene='kitchen_white')
    mujoco_envs[i].forward.remote(q=init_ur_q, joint_idxs=idxs_forward)
end_mujoco_envs = time.time()
print(f"Time taken for MuJoCo initialization: {end_mujoco_envs - start_mujoco_envs:.4f} seconds")

# 1. Sample all of the possible positions
start_sample_pcd = time.time()
positions = ray.get(mujoco_envs[0].get_sampled_position.remote(offset=[0,0,0.12], scene='kitchen_white', threshold=0.03, 
                    resolution_table=(9,9,9), resolution_shelf=(6,6,6), resolution_obj=(15,15,1),
                    x_range=[0.40, 1.15], z_range=[0.7, 1.3]))
splited_positions = np.array_split(positions, num_envs)
end_sample_pcd = time.time()
print(f"Time taken for sampled positions: {end_sample_pcd - start_sample_pcd:.4f} seconds")
len_feasible = len(positions)
np.save('./data/white/filtered_pcd.npy', positions)

# 3. Get the feasible positions
start_feasibility_check = time.time()
remaining_envs = list(enumerate(mujoco_envs))
target_object_name = 'kitchen-plate'
# Asynchronous call.
for env_idx, env in remaining_envs:
# for env_idx, env in tqdm(remaining_envs, desc="Processing environments", total=len(remaining_envs)):
    env.get_feasible_position.remote(splited_positions[env_idx], scene='kitchen_white', var_threshold=150, VERBOSE=True, 
                                     quat_bounds = [0.64, 0.79], perturb_tick = 500, noise_scale = 0.0015,
                                     target_obj_name=target_object_name, offset=np.array([0, 0, 0.01]), init_q=init_ur_q, nstep=100, end_tick=3000)
end_feasibility_check = time.time()
print(f"Time taken for feasibility check: {end_feasibility_check - start_feasibility_check:.4f} seconds")

start_the_rest = time.time()
for i in range(num_envs):
    print(f"The # of Feasible Set: [Worker: {i}]:[{len(ray.get(mujoco_envs[i].get_feasible_place_positions.remote()))}]")

feasible_points = []
quat_ranges = []
result_each_worker = [mujoco_envs[i].get_feasible_place_positions.remote() for i in range(num_envs)]
results = ray.get(result_each_worker)

for pcd, quat_range in results:
    print(f"fea: {pcd}, quat: {quat_range}")

    if len(pcd) > 0:
        # print(pcd, quat_range)
        feasible_points.append(pcd)
        quat_ranges.append(quat_range)
    else:
        print(f"Empty result: {pcd}")
if len(feasible_points) > 0:
    feasible_points = np.concatenate(feasible_points, axis=0)
    print(f"The # of concatenated feasible_points: {feasible_points.shape}")
if len(quat_ranges) > 0:
    quat_ranges = np.concatenate(quat_ranges, axis=0)
    print(f"The # of concatenated quat_ranges: {quat_ranges.shape}")
print(f"The # of concatenated feasible_points: {feasible_points.shape}")

# if the file have already existed, then add the new letter to the file name
print(f"num_cpus: {num_cpus}, num_gpus: {num_gpus}, num_envs: {num_envs}")
if os.path.isfile(f'./data/white/feasible_pcd_{file_name}_c{num_cpus}_g{num_gpus}_worker{num_envs}.npy'):
    np.save(f'./data/white/feasible_pcd_{file_name}_c{num_cpus}_g{num_gpus}_worker{num_envs}_2.npy', feasible_points)
    print(f"[Renamed] The file was successfully saved!: {f'./data/white/feasible_pcd_{file_name}_c{num_cpus}_g{num_gpus}_worker{num_envs}_2.npy'}")
else:
    np.save(f'./data/white/feasible_pcd_{file_name}_c{num_cpus}_g{num_gpus}_worker{num_envs}.npy', feasible_points)
    print(f"The file was successfully saved!: {f'./data/white/feasible_pcd_{file_name}_c{num_cpus}_g{num_gpus}_worker{num_envs}.npy'}")

if os.path.isfile(f'./data/white/quat_range_{file_name}_c{num_cpus}_g{num_gpus}_worker{num_envs}.npy'):
    np.save(f'./data/white/quat_range_{file_name}_c{num_cpus}_g{num_gpus}_worker{num_envs}_2.npy', quat_ranges)
    print(f"[Renamed] The file was successfully saved!: {f'./data/white/quat_range_{file_name}_c{num_cpus}_g{num_gpus}_worker{num_envs}_2.npy'}")
else:
    np.save(f'./data/white/quat_range_{file_name}_c{num_cpus}_g{num_gpus}_worker{num_envs}.npy', quat_ranges)
    print(f"The file was successfully saved!: {f'./data/white/quat_range_{file_name}_c{num_cpus}_g{num_gpus}_worker{num_envs}.npy'}")
end_the_rest = time.time()
print(f"Time taken for get feasible pcds: {end_the_rest - start_the_rest:.4f} seconds")

virtual_display.stop()
print(f"Time taken for Ray initialization: {end_ray - start_ray:.4f} seconds")
print(f"Time taken for MuJoCo initialization: {end_mujoco_envs - start_mujoco_envs:.4f} seconds")
print(f"Time taken for sampled positions: {end_sample_pcd - start_sample_pcd:.4f} seconds")
print(f"Time taken for feasibility check: {end_feasibility_check - start_feasibility_check:.4f} seconds")
print(f"Time taken for get feasible pcds: {end_the_rest - start_the_rest:.4f} seconds")
print(f"Total time taken: {time.time() - start_time:.4f} seconds")
check_time = end_sample_pcd - start_sample_pcd + end_feasibility_check - start_feasibility_check + end_the_rest - start_the_rest
print(f"check: {check_time:.4f} seconds")
print(f"len positions: {len_feasible}")
virtual_display.stop()

print(f"shape of feasible_points: {feasible_points.shape}")