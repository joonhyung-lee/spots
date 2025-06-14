import os
import cv2
import mujoco_viewer
import numpy as np
import ray
import time
# import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from utils import MUJOCO_PARSER_ROOT_DIR
sys.path.append(MUJOCO_PARSER_ROOT_DIR)
from utils.mujoco_parser import MuJoCoParserClass
from utils.util import (rpy2r,r2rpy,r2quat, fill_object, get_geom_region_type1, get_geom_region_type2, get_geom_region_type3, get_geom_region_type4, get_geom_region_type5, get_geom_region_type8,
                        sample_xyzs, passthrough_filter, remove_duplicates_with_threshold)
# print(f"MUJOCO_PARSER_ROOT_DIR={MUJOCO_PARSER_ROOT_DIR}")

INIT_UR_Q = np.array([np.deg2rad(-90), np.deg2rad(-130), np.deg2rad(120), np.deg2rad(100), np.deg2rad(45), np.deg2rad(-90)])

@ray.remote(max_restarts=-1)
class MuJoCoParserClassRay(MuJoCoParserClass):
    """
        MuJoCo Parser class
    """
    def __init__(self,name='Robot',rel_xml_path=None,USE_MUJOCO_VIEWER=False,VERBOSE=False, MODE='offscreen',env_id=0):
        """
            Initialize MuJoCo parser with ray
        """
        super().__init__(name=name,rel_xml_path=rel_xml_path,USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER,VERBOSE=VERBOSE, MODE=MODE)
        self.cnt = 0
        # self.quat_w_min = 0
        # self.quat_w_max = 0
        self.quat_range = 0
        self.quat_range_list = []
        self.DONE_FLAG = False
        self.feasible_place_positions = []


    def get_mode(self):
        """
            Get mode
        """
        return self.MODE.copy()
    
    def get_idxs(self):
        """
            Get state of the robot
        """
        return [self.idxs_forward.copy(), self.idxs_jacobian.copy(), self.idxs_step.copy()]

    def check_done(self):
        """
            Check done flag
        """
        if self.DONE_FLAG:
            return True
        else:
            return False

    def get_done(self):
        """
            Get done flag
        """
        return self.DONE_FLAG
    
    def get_cnt(self):
        """
            Get cnt
        """
        return self.cnt.copy()
    
    def get_feasible_place_positions(self):
        """
            Get feasible place positions
        """
        return self.feasible_place_positions.copy(), self.quat_range_list
    
    def get_physical_score(self):
        """
            Get physical score
        """
        return self.feasible_place_positions.copy()

    def set_viewer(self, azimuth, elevation, distance, lookat):
        self.viewer.cam.azimuth = azimuth
        self.viewer.cam.elevation = elevation
        self.viewer.cam.distance = distance
        self.viewer.cam.lookat = lookat

    def reset_env(self, flag=False):
        """
            Reset done flag
        """
        self.DONE_FLAG = flag
        self.cnt = 0
        self.feasible_place_positions = []

    def init_set_state(self, scene='kitchen'):
        """
            Initialize set environment configuration
        """
        if scene == 'kitchen_white':
            # Move tables and robot base
            self.model.body('base_table').pos = np.array([0,0,0])
            self.model.body('base').pos = np.array([0.18,0,0.79])
            self.model.body('avoiding_object_table').pos = np.array([0.38+0.45,0,0])
            self.model.body('right_object_table').pos = np.array([-0.05,-0.80,0])
            self.model.body('left_object_table').pos = np.array([-1.5,0.80,0])

            # Place objects
            obj_box_names = [body_name for body_name in self.body_names
                        if body_name is not None and (body_name.startswith("obj_box"))]
            n_box_obj = len(obj_box_names)
            self.place_objects_random(n_obj=n_box_obj, obj_names=obj_box_names, x_range=[0.80, 1.15], y_range=[-3.15, -2.15], COLORS=False, VERBOSE=True)

            jntadr_mug_cup = self.model.body('mug_cup').jntadr[0]
            self.model.joint(jntadr_mug_cup).qpos0[:3] = np.array([ 0.9, -0.3, 0.85])
            self.model.joint(jntadr_mug_cup).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, np.random.rand() * 360])))

            jntadr_tray = self.model.body('tray').jntadr[0]
            self.model.joint(jntadr_tray).qpos0[:3] = np.array([0.9,-0.3,0.80])
            # self.model.joint(jntadr_tray).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, 0])))

            jntadr_dish_rack = self.model.body('dish_rack').jntadr[0]
            pos_dish_rack = np.array([0.9,0.35,0.82])
            self.model.joint(jntadr_dish_rack).qpos0[:3] = pos_dish_rack
            dish_rack_random_rot = 0 # np.random.rand() * 360
            self.model.joint(jntadr_dish_rack).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, dish_rack_random_rot])))

            # Target dish
            self.model.joint(self.model.body('kitchen-plate').jntadr[0]).qpos0[:3] = np.array([0.2, -0.7, 0.8])
            # Set the dishes
            self.model.joint(self.model.body('kitchen-plate-2').jntadr[0]).qpos0[:3] = pos_dish_rack + np.array([0.0,0.02, 0.2])
            self.model.joint(self.model.body('kitchen-plate-2').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 90, 360])))
            self.model.joint(self.model.body('kitchen-plate-3').jntadr[0]).qpos0[:3] = pos_dish_rack + np.array([-0.05, 0.02, 0.2])
            self.model.joint(self.model.body('kitchen-plate-3').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 90, 360])))
            self.model.joint(self.model.body('kitchen-plate-4').jntadr[0]).qpos0[:3] = pos_dish_rack + np.array([-3.12, 0.02, 0.2])
            # self.model.joint(self.model.body('kitchen-plate-4').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 90, 360])))

        elif scene == 'kitchen_small':
            # Move tables and robot base
            self.model.body('base_table').pos = np.array([0,0,0])
            self.model.body('base').pos = np.array([0.18,0,0.79])
            self.model.body('avoiding_object_table').pos = np.array([0.38+0.45,0,0])
            self.model.body('right_object_table').pos = np.array([-0.05,-0.80,0])
            self.model.body('left_object_table').pos = np.array([-1.5,0.80,0])

            # Place objects
            obj_box_names = [body_name for body_name in self.body_names
                        if body_name is not None and (body_name.startswith("obj_box"))]
            n_box_obj = len(obj_box_names)
            self.place_objects_random(n_obj=n_box_obj, obj_names=obj_box_names, x_range=[0.80, 1.15], y_range=[-3.15, -2.15], COLORS=False, VERBOSE=True)

            jntadr_mug_cup = self.model.body('mug_cup').jntadr[0]
            self.model.joint(jntadr_mug_cup).qpos0[:3] = np.array([ 0.9, -0.3, 0.85])
            self.model.joint(jntadr_mug_cup).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, np.random.rand() * 360])))

            jntadr_tray = self.model.body('tray').jntadr[0]
            self.model.joint(jntadr_tray).qpos0[:3] = np.array([0.9,-0.3,0.80])
            # self.model.joint(jntadr_tray).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, 0])))

            jntadr_dish_rack = self.model.body('rack_base').jntadr[0]
            pos_dish_rack = np.array([0.9,0.35,0.82])
            self.model.joint(jntadr_dish_rack).qpos0[:3] = pos_dish_rack
            dish_rack_random_rot = 0 # np.random.rand() * 360
            self.model.joint(jntadr_dish_rack).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, dish_rack_random_rot])))

            # Target dish
            self.model.joint(self.model.body('kitchen-plate').jntadr[0]).qpos0[:3] = np.array([0.2, -0.7, 0.8])
            # Set the dishes
            self.model.joint(self.model.body('kitchen-plate-2').jntadr[0]).qpos0[:3] = pos_dish_rack + np.array([0.,0.0, 0.15])
            self.model.joint(self.model.body('kitchen-plate-2').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 90, 90])))
            self.model.joint(self.model.body('kitchen-plate-3').jntadr[0]).qpos0[:3] = pos_dish_rack + np.array([0., -0.1, 0.15])
            self.model.joint(self.model.body('kitchen-plate-3').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 90, 90])))
            self.model.joint(self.model.body('kitchen-plate-4').jntadr[0]).qpos0[:3] = pos_dish_rack + np.array([-2.5, 0.1, 0.15])
            self.model.joint(self.model.body('kitchen-plate-4').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 90, 90])))

        elif scene == 'kitchen_black':
            # Move tables and robot base
            self.model.body('base_table').pos = np.array([0,0,0])
            self.model.body('base').pos = np.array([0.18,0,0.79])
            self.model.body('avoiding_object_table').pos = np.array([0.38+0.45,0,0])
            self.model.body('right_object_table').pos = np.array([-0.05,-0.80,0])
            self.model.body('left_object_table').pos = np.array([-1.5,0.80,0])

            # Place objects
            obj_box_names = [body_name for body_name in self.body_names
                        if body_name is not None and (body_name.startswith("obj_box"))]
            n_box_obj = len(obj_box_names)
            self.place_objects_random(n_obj=n_box_obj, obj_names=obj_box_names, x_range=[0.80, 1.15], y_range=[-3.15, -2.15], COLORS=False, VERBOSE=True)

            jntadr_mug_cup = self.model.body('mug_cup').jntadr[0]
            self.model.joint(jntadr_mug_cup).qpos0[:3] = np.array([ 0.9, -0.3, 0.85])
            self.model.joint(jntadr_mug_cup).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, np.random.rand() * 360])))

            jntadr_tray = self.model.body('tray').jntadr[0]
            self.model.joint(jntadr_tray).qpos0[:3] = np.array([0.9,-0.3,0.80])
            # self.model.joint(jntadr_tray).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, 0])))

            jntadr_dish_rack = self.model.body('kitchen-drainer').jntadr[0]
            pos_dish_rack = np.array([0.9,0.35,0.82])
            self.model.joint(jntadr_dish_rack).qpos0[:3] = pos_dish_rack
            dish_rack_random_rot = 0 # np.random.rand() * 360
            self.model.joint(jntadr_dish_rack).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, dish_rack_random_rot])))

            # Target dish
            self.model.joint(self.model.body('kitchen-plate').jntadr[0]).qpos0[:3] = np.array([0.2, -0.7, 0.8])
            # Set the dishes
            self.model.joint(self.model.body('kitchen-plate-2').jntadr[0]).qpos0[:3] = pos_dish_rack + np.array([0.,-0.06, 0.25])
            self.model.joint(self.model.body('kitchen-plate-2').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 90, 90])))
            self.model.joint(self.model.body('kitchen-plate-3').jntadr[0]).qpos0[:3] = pos_dish_rack + np.array([0., -0.12, 0.25])
            self.model.joint(self.model.body('kitchen-plate-3').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 90, 90])))
            self.model.joint(self.model.body('kitchen-plate-4').jntadr[0]).qpos0[:3] = pos_dish_rack + np.array([-3.5, 0., 0.25])
            self.model.joint(self.model.body('kitchen-plate-4').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 90, 90])))

        elif scene == 'office_bookshelf':
            # Move tables and robot base
            self.model.body('base_table').pos = np.array([0,0,0])
            self.model.body('avoiding_object_table').pos = np.array([0.38+0.45,0,0])
            self.model.body('base').pos = np.array([0.18,0,0.79])
            self.model.body('right_object_table').pos = np.array([-0.05,-0.80,0])
            self.model.body('left_object_table').pos = np.array([-0.05,0.80,0])

            jntadr_mug_cup = self.model.body('mug_cup').jntadr[0]
            self.model.joint(jntadr_mug_cup).qpos0[:3] = np.array([0.5, -0.4, 0.81])
            self.model.joint(jntadr_mug_cup).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, np.random.rand() * 360])))
            self.model.body('shelf_row1').pos = np.array([0.8,0.3,0.75])
            self.model.body('shelf_row1').quat = r2quat(rpy2r(np.radians([0, 0, 90])))

            obj_box_names = [body_name for body_name in self.body_names
                        if body_name is not None and (body_name.startswith("obj_box"))]
            obj_wine_names = [body_name for body_name in self.body_names
                            if body_name is not None and (body_name.startswith("obj_wine"))]
            obj_book_names = [body_name for body_name in self.body_names
                        if body_name is not None and (body_name.startswith("book"))]

            # Place objects
            n_box_obj = len(obj_box_names)
            self.place_objects_random(n_obj=n_box_obj, obj_names=obj_box_names, x_range=[0.80, 1.15], y_range=[-3.15, -2.15], COLORS=False, VERBOSE=True)
            n_wine_obj = 3
            self.place_objects_random(n_obj=n_wine_obj, obj_names=obj_wine_names, x_range=[0.80, 0.85], y_range=[0.3, 0.5], z_range=[1.35,1.36], min_dist=0.05, COLORS=False, VERBOSE=True)

            self.model.joint(self.model.body('mug_cup').jntadr[0]).qpos0[:3] = np.array([-8, 0.45, 1.55])

            # Book1 is target_object
            self.model.joint(self.model.body('book2').jntadr[0]).qpos0[:3] = np.array([0.80, 0.30, 1.03])
            self.model.joint(self.model.body('book3').jntadr[0]).qpos0[:3] = np.array([0.82, 0.35, 1.03])
            self.model.joint(self.model.body('book4').jntadr[0]).qpos0[:3] = np.array([0.79, 0.40, 1.03])
            self.model.joint(self.model.body('book5').jntadr[0]).qpos0[:3] = np.array([0.84, 0.45, 1.03])
            self.model.joint(self.model.body('book6').jntadr[0]).qpos0[:3] = np.array([0.8, 0.25, 0.8])
            self.model.joint(self.model.body('book7').jntadr[0]).qpos0[:3] = np.array([0.81, 0.29, 0.80])
            self.model.joint(self.model.body('book8').jntadr[0]).qpos0[:3] = np.array([0.79, 0.38, 0.80])
            self.model.joint(self.model.body('book9').jntadr[0]).qpos0[:3] = np.array([0.8, 0.45, 0.80])
            self.model.joint(self.model.body('book10').jntadr[0]).qpos0[:3] = np.array([0.83, 0.49, 0.80])

            self.model.joint(self.model.body('book1').jntadr[0]).qpos0[:3] = np.array([0.2, -0.4, 0.8])
            self.model.joint(self.model.body('book1').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.array([1.57, 0, 0])))
        elif scene == "bookshelf_genre":
            # Move tables and robot base
            self.model.body('base_table').pos = np.array([0,0,0])
            self.model.body('avoiding_object_table').pos = np.array([0.38+0.45,0,0])
            self.model.body('base').pos = np.array([-1.18,0,0.79])
            self.model.body('right_object_table').pos = np.array([-0.05,-0.80,0])
            self.model.body('left_object_table').pos = np.array([-5,0.80,0])

            shelf_pos = np.array([0.8,0.0,0.75])
            shelf_2nd_pos = shelf_pos + np.array([0.,0.,0.25])
            self.model.body('shelf_row1').pos = shelf_pos
            self.model.body('shelf_row1').quat = r2quat(rpy2r(np.radians([0, 0, 90])))

            obj_box_names = [body_name for body_name in self.body_names
                        if body_name is not None and (body_name.startswith("obj_box"))]
            obj_wine_names = [body_name for body_name in self.body_names
                            if body_name is not None and (body_name.startswith("obj_wine"))]

            # Place objects
            n_wine_obj = 2
            self.place_objects_random(n_obj=n_wine_obj, obj_names=obj_wine_names, x_range=[0.80, 0.85], y_range=[-0.2, 0.2], z_range=[1.25,1.26], min_dist=0.05, COLORS=False, VERBOSE=True)

            # Book1 is target_object
            self.model.joint(self.model.body('book2').jntadr[0]).qpos0[:3] = shelf_2nd_pos + np.array([0, 0.225, 0.02])
            self.model.joint(self.model.body('book3').jntadr[0]).qpos0[:3] = shelf_2nd_pos + np.array([0, 0.16, 0.02])
            self.model.joint(self.model.body('book4').jntadr[0]).qpos0[:3] = shelf_2nd_pos + np.array([0, 0.11, 0.02])
            self.model.joint(self.model.body('book6').jntadr[0]).qpos0[:3] = shelf_pos + np.array([0, -0.22, 0.02])
            self.model.joint(self.model.body('book7').jntadr[0]).qpos0[:3] = shelf_pos + np.array([0, -0.16, 0.02])
            self.model.joint(self.model.body('book8').jntadr[0]).qpos0[:3] = shelf_pos + np.array([0, -0.12, 0.02])
            self.model.joint(self.model.body('book9').jntadr[0]).qpos0[:3] = shelf_pos + np.array([0, -0.07, 0.02])

            self.model.joint(self.model.body('book1').jntadr[0]).qpos0[:3] = np.array([0.2, -0.4, 0.8])
            self.model.joint(self.model.body('book1').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.array([1.57, 0, 0])))

        elif scene == "bookshelf_genre_random":
            shelf_pos = np.array([0.8,0.0,0.75])
            shelf_2nd_pos = shelf_pos + np.array([0.,0.,0.25])
            shelf_3rd_pos = shelf_pos + np.array([0.,0.,0.5])
            self.model.body('shelf_row1').pos = shelf_pos
            self.model.body('shelf_row1').quat = r2quat(rpy2r(np.radians([0, 0, 90])))

            obj_box_names = [body_name for body_name in self.body_names
                        if body_name is not None and (body_name.startswith("obj_box"))]
            obj_wine_names = [body_name for body_name in self.body_names
                            if body_name is not None and (body_name.startswith("obj_wine"))]
            obj_book_names = [body_name for body_name in self.body_names
                        if body_name is not None and (body_name.startswith("book"))]

            random_bookshelf_row1 = sample_xyzs(n_sample=4,
                        x_range=[0.80, 0.82],y_range=[-0.2, 0.2],z_range=[0.78, 0.79],min_dist=0.08)
            random_bookshelf_row2 = sample_xyzs(n_sample=4,
                        x_range=[0.80, 0.82],y_range=[-0.2, 0.2],z_range=[shelf_2nd_pos[2]+0.03, shelf_2nd_pos[2]+0.03],min_dist=0.08)
            random_bookshelf_row3 = sample_xyzs(n_sample=4,
                        x_range=[0.80, 0.82],y_range=[-0.2, 0.2],z_range=[shelf_3rd_pos[2]+0.15, shelf_3rd_pos[2]+0.25],min_dist=0.08)
            # Place objects
            # random indices
            random_bookshelf_row1_idx = np.random.permutation(random_bookshelf_row1)
            random_bookshelf_row2_idx = np.random.permutation(random_bookshelf_row2)
            random_bookshelf_row3_idx = np.random.permutation(random_bookshelf_row3)

            # Book1 is target_object
            self.model.joint(self.model.body('book2').jntadr[0]).qpos0[:3] = random_bookshelf_row1_idx[0]
            self.model.joint(self.model.body('book3').jntadr[0]).qpos0[:3] = random_bookshelf_row1_idx[1]
            self.model.joint(self.model.body('book4').jntadr[0]).qpos0[:3] = random_bookshelf_row1_idx[2]
            self.model.joint(self.model.body('book6').jntadr[0]).qpos0[:3] = random_bookshelf_row1_idx[3]
            self.model.joint(self.model.body('book7').jntadr[0]).qpos0[:3] = random_bookshelf_row2_idx[0]
            self.model.joint(self.model.body('book8').jntadr[0]).qpos0[:3] = random_bookshelf_row2_idx[1]
            self.model.joint(self.model.body('book9').jntadr[0]).qpos0[:3] = random_bookshelf_row1_idx[2]
            self.model.joint(self.model.body('obj_wine_01').jntadr[0]).qpos0[:3] = random_bookshelf_row3_idx[0]
            self.model.joint(self.model.body('obj_wine_02').jntadr[0]).qpos0[:3] = random_bookshelf_row3_idx[1]

        elif scene == "bookshelf_modified":
            # Move tables and robot base
            self.model.body('base_table').pos = np.array([0,0,0])
            self.model.body('avoiding_object_table').pos = np.array([0.38+0.45,0,0])
            self.model.body('base').pos = np.array([-1.18,0,0.79])
            self.model.body('right_object_table').pos = np.array([-0.05,-0.80,0])
            self.model.body('left_object_table').pos = np.array([-5,0.80,0])

            shelf_pos = np.array([0.8,0.0,0.75])
            self.model.body('shelf_row1').pos = shelf_pos
            self.model.body('shelf_row1').quat = r2quat(rpy2r(np.radians([0, 0, 90])))
            obj_wine_names = [body_name for body_name in self.body_names
                            if body_name is not None and (body_name.startswith("obj_wine"))]

            # Place objects
            n_wine_obj = 2
            self.place_objects_random(n_obj=n_wine_obj, obj_names=obj_wine_names, x_range=[0.80, 0.85], y_range=[-0.2, 0.2], z_range=[1.25,1.26], min_dist=0.05, COLORS=False, VERBOSE=True)

            # Book1 is target_object
            self.model.joint(self.model.body('book6').jntadr[0]).qpos0[:3] = shelf_pos + np.array([0, -0.22, 0.02])
            self.model.joint(self.model.body('book7').jntadr[0]).qpos0[:3] = shelf_pos + np.array([0, -0.16, 0.02])
            self.model.joint(self.model.body('book8').jntadr[0]).qpos0[:3] = shelf_pos + np.array([0, -0.12, 0.02])
            self.model.joint(self.model.body('book9').jntadr[0]).qpos0[:3] = shelf_pos + np.array([0, -0.07, 0.02])

            self.model.joint(self.model.body('book1').jntadr[0]).qpos0[:3] = np.array([0.2, -0.4, 0.8])
            self.model.joint(self.model.body('book1').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.array([1.57, 0, 0])))

        elif scene == "category_pringles":
            # Move tables and robot base
            self.model.body('base_table').pos = np.array([0,0,0])
            self.model.body('avoiding_object_table').pos = np.array([0.38+0.45,0,0])
            self.model.body('base').pos = np.array([-18,0,0.79])
            self.model.body('right_object_table').pos = np.array([-0.05,-0.80,0])
            self.model.body('left_object_table').pos = np.array([-5,0.80,0])

            shelf_pos = np.array([0.9,0.0,0.8])
            self.model.body('shelf_row1').pos = shelf_pos

            # Place objects
            tray_pos = shelf_pos + np.array([ 0.05, -0.12, 0.20])
            self.model.joint(self.model.body('tray_gray').jntadr[0]).qpos0[:3] = tray_pos
            self.model.joint(self.model.body('tray_gray').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([90, 0, 90])))
            self.model.joint(self.model.body('ycb-apple-2').jntadr[0]).qpos0[:3] = tray_pos + np.array([-0.05,-0.05,0.05])
            self.model.joint(self.model.body('cider').jntadr[0]).qpos0[:3] = shelf_pos + np.array([-0.05,0.15,0.30])
            self.model.joint(self.model.body('mug_cup').jntadr[0]).qpos0[:3] = shelf_pos + np.array([0.0,-0.1,0.03])
            self.model.joint(self.model.body('mug_cup').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, 90])))
            self.model.joint(self.model.body('pocky').jntadr[0]).qpos0[:3] = shelf_pos + np.array([-0.0,-0.07,0.5])
            self.model.joint(self.model.body('pocky').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([-90, 90, 0])))

            # Do Not use
            self.model.joint(self.model.body('obj_wine_01').jntadr[0]).qpos0[:3] = np.array([-20,1,1])
            self.model.joint(self.model.body('fat_wine_glass').jntadr[0]).qpos0[:3] = shelf_pos + np.array([-10.0,0.15,0.02])
            self.model.joint(self.model.body('fat_wine_glass').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([90, 0, 0])))
            self.model.joint(self.model.body('tray').jntadr[0]).qpos0[:3] = np.array([-5,1,1])
            self.model.joint(self.model.body('ycb-orange-2').jntadr[0]).qpos0[:3] = tray_pos + np.array([-50,0.05,0.05])
            self.model.joint(self.model.body('chocopie').jntadr[0]).qpos0[:3] = shelf_pos + np.array([-50.0,0.07,0.5])
            self.model.joint(self.model.body('chocopie').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([90, 0, 90])))
            self.model.joint(self.model.body('ycb-lemon').jntadr[0]).qpos0[:3] = np.array([-10.0, -0.6, 0.8])
            self.model.joint(self.model.body('ycb-lemon').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, -90])))
            self.model.joint(self.model.body('orange_can').jntadr[0]).qpos0[:3] = np.array([-10.05,-0.5,0.85])
            self.model.joint(self.model.body('orange_can').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, 135])))
            self.model.joint(self.model.body('ycb-apple').jntadr[0]).qpos0[:3] = np.array([-5,-5,1])
            self.model.joint(self.model.body('ycb-banana').jntadr[0]).qpos0[:3] = np.array([-5,-5,1])
            self.model.joint(self.model.body('ycb-orange').jntadr[0]).qpos0[:3] = np.array([-5,-5,1])
            self.model.joint(self.model.body('kitchen-bowl-red').jntadr[0]).qpos0[:3] = np.array([-5,-5,1])
            self.model.joint(self.model.body('kitchen-bowl-blue').jntadr[0]).qpos0[:3] = np.array([-5,-5,1])
            self.model.joint(self.model.body('redbullcan').jntadr[0]).qpos0[:3] = np.array([-5,-5,1])
            self.model.joint(self.model.body('coke').jntadr[0]).qpos0[:3] = np.array([-5,-5,1])


            # Target objects
            self.model.joint(self.model.body('pringles').jntadr[0]).qpos0[:3] = np.array([0.2, -0.6, 0.8])
            self.model.joint(self.model.body('pringles').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([90, 0, 135])))

        elif scene == 'category_tray':
            # Move tables and robot base
            self.model.body('base_table').pos = np.array([0,0,0])
            self.model.body('avoiding_object_table').pos = np.array([0.38+0.45,0,0])
            self.model.body('base').pos = np.array([0.18,0,0.79])
            self.model.body('right_object_table').pos = np.array([-0.05,-0.80,0])
            self.model.body('left_object_table').pos = np.array([-0.05,0.80,0])

            # Place objects
            tray_pos = np.array([ 0.9, 0.25, 0.8])
            tray_gray_pos = np.array([ 0.9, 0., 0.8])
            bowl_red_pos = np.array([ 0.9, -0.25, 0.8])
            self.model.joint(self.model.body('tray').jntadr[0]).qpos0[:3] = tray_pos
            self.model.joint(self.model.body('tray').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([90, 0, 90])))
            self.model.joint(self.model.body('tray_gray').jntadr[0]).qpos0[:3] = tray_gray_pos
            self.model.joint(self.model.body('tray_gray').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([90, 0, 90])))
            self.model.joint(self.model.body('kitchen-bowl-red').jntadr[0]).qpos0[:3] = bowl_red_pos

            self.model.joint(self.model.body('ycb-apple-2').jntadr[0]).qpos0[:3] = bowl_red_pos + np.array([0,0,0.1])
            self.model.joint(self.model.body('mug_cup').jntadr[0]).qpos0[:3] = tray_pos + np.array([0.05,0,0.05])
            self.model.joint(self.model.body('hershey_cookie').jntadr[0]).qpos0[:3] = tray_gray_pos + np.array([0.05,0,0.05])
            self.model.joint(self.model.body('hershey_cookie').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.radians([90, -90, 0])))

            self.model.joint(self.model.body('kitchen-bowl-blue').jntadr[0]).qpos0[:3] = np.array([-5,0,0.5])
            self.model.joint(self.model.body('ycb-banana-2').jntadr[0]).qpos0[:3] = np.array([-9,0,0]) # sorted_pose[2] + spawn_z_offset
            self.model.joint(self.model.body('ycb-lemon-2').jntadr[0]).qpos0[:3] = np.array([-5,0,0.5])
            self.model.joint(self.model.body('ycb-orange-2').jntadr[0]).qpos0[:3] = np.array([-5,0,0.5])

            # Target objects
            self.model.joint(self.model.body('pringles').jntadr[0]).qpos0[:3] = np.array([0.1, -0.6, 0.8])

        super().reset()

    def get_sampled_position(self, scene='kitchen', offset=[0,0,0.05], threshold=0.05, resolution_table=(9,9,9), resolution_shelf=(6,6,6), resolution_obj=(2,2,2),
                             x_range=[0.40, 1.15], z_range=[0.7, 1.3], add_table=False):
        if scene == 'kitchen_white':
            obj_drainer_region = fill_object(get_geom_region_type4(self, 'dish_rack'), resolution_obj)
            # obj_table_region = fill_object(get_geom_region_type3(self, 'avoiding_object_table'), resolution_table)

            stacked_geom_region = np.vstack([
                obj_drainer_region # ,
                # obj_table_region#, obj_mugcup_region
            ])
        elif scene == 'kitchen_small':
            obj_drainer_region = fill_object(get_geom_region_type5(self, 'rack_base'), resolution_obj)

            stacked_geom_region = np.vstack([
                obj_drainer_region
            ])

        elif scene == 'kitchen_black':
            obj_drainer_region = fill_object(get_geom_region_type4(self, 'kitchen-drainer'), resolution_obj)

            stacked_geom_region = np.vstack([
                obj_drainer_region
            ])
            
        elif scene == 'office_bookshelf':
            obj_shelf_row1_region = fill_object(get_geom_region_type1(self, 'shelf_row1'), resolution_obj)
            obj_shelf_row2_region = fill_object(get_geom_region_type1(self, 'shelf_row2'), resolution_obj)
            obj_shelf_row3_region = fill_object(get_geom_region_type1(self, 'shelf_row3'), resolution_obj)

            stacked_geom_region = np.vstack([
                obj_shelf_row1_region, obj_shelf_row2_region, obj_shelf_row3_region,
            ])

        elif scene == 'bookshelf_modified':
            obj_shelf_row1_region = fill_object(get_geom_region_type1(self, 'shelf_row1'), resolution_obj)
            obj_shelf_row2_region = fill_object(get_geom_region_type1(self, 'shelf_row2'), resolution_obj)

            stacked_geom_region = np.vstack([
                obj_shelf_row1_region, obj_shelf_row2_region
            ])

        elif scene == 'category_pringles':
            obj_shelf_row1_region = fill_object(get_geom_region_type1(self, 'shelf_row1'), resolution_obj)
            obj_shelf_row2_region = fill_object(get_geom_region_type1(self, 'shelf_row2'), resolution_obj)
            obj_shelf_row3_region = fill_object(get_geom_region_type1(self, 'shelf_row3'), resolution_obj)

            stacked_geom_region = np.vstack([
                obj_shelf_row1_region, obj_shelf_row2_region, obj_shelf_row3_region,
            ])

        elif scene == 'category_tray':
            obj_shelf_row1_region = fill_object(get_geom_region_type8(self, 'tray'), resolution_obj)
            obj_shelf_row2_region = fill_object(get_geom_region_type8(self, 'tray_gray'), resolution_obj)
            obj_shelf_row3_region = fill_object(get_geom_region_type1(self, 'kitchen-bowl-red'), resolution_obj)

            stacked_geom_region = np.vstack([
                obj_shelf_row1_region, obj_shelf_row2_region, obj_shelf_row3_region,
            ])

        if add_table:
            obj_table_region = fill_object(get_geom_region_type3(self, 'avoiding_object_table'), resolution_table)
            stacked_geom_region = np.vstack([
                stacked_geom_region, obj_table_region
            ])
        pcd_masked = passthrough_filter(stacked_geom_region, axis=0, interval=x_range)
        pcd_masked = passthrough_filter(pcd_masked, axis=2, interval=z_range)
        pcd_masked += np.array(offset)

        filtered_pcd = remove_duplicates_with_threshold(pcd_masked, threshold)
        print(f"Stacked geometry regions's shape: {stacked_geom_region.shape}")
        print(f"Masked geometry regions's shape: {pcd_masked.shape}")
        print(f"Removed duplicated geometry regions's shape: {filtered_pcd.shape}")
        print(f"All of the # of positions is: {len(filtered_pcd)}")
        np.save('./data/filtered_pcd.npy', filtered_pcd)
        return filtered_pcd
        
    def get_feasible_position(self, positions, scene='kitchen', target_obj_name='obj_box_01', MODE='offscreen',var_threshold=0.0005, offset=np.array([0, 0, 0.05]), init_q=INIT_UR_Q, nstep=1, end_tick=3000, perturb_tick=500, noise_scale=0.01, VERBOSE=False, quat_bounds=None):
        super().reset()
        self.MODE = MODE
        self.DONE_FLAG = False
        target_obj_jntadr = self.model.body(target_obj_name).jntadr[0]
        target_obj_qposadr = self.model.jnt_qposadr[target_obj_jntadr]
        p_list = []
        R_list = []
        quat_list = []
        body1s_list, body2s_list = [], []
        self.feasible_place_positions = []

        for p_idx, p in enumerate(positions):
            # print(f"The # of positions: [{p_idx+1}/{len(positions)}]")
            super().forward(q=init_q,joint_idxs=self.idxs_forward)
            p_copy = p.copy()
            p_copy += offset
            start = self.tick
            _,f_contacts,_,_,body1s,body2s = self.get_contact_info(must_include_prefix=target_obj_name)
            self.data.qpos[target_obj_qposadr:target_obj_qposadr+3] = p_copy
            if scene == 'kitchen_white':
                self.data.qpos[target_obj_qposadr+3:target_obj_qposadr+7] = r2quat(rpy2r(np.radians([0, 90, 360])))
            elif scene == 'kitchen_small':
                self.data.qpos[target_obj_qposadr+3:target_obj_qposadr+7] = r2quat(rpy2r(np.radians([0, 90, 90])))
            elif scene == 'kitchen_black':
                self.data.qpos[target_obj_qposadr+3:target_obj_qposadr+7] = r2quat(rpy2r(np.radians([0, 90, 90])))
            elif scene == 'office_bookshelf':
                self.data.qpos[target_obj_qposadr+3:target_obj_qposadr+7] = r2quat(rpy2r(np.radians([0, 0, 0])))
            elif scene == 'category_pringles':
                self.data.qpos[target_obj_qposadr+3:target_obj_qposadr+7] = r2quat(rpy2r(np.radians([90, 0, 0])))
            elif scene == 'category_tray':
                self.data.qpos[target_obj_qposadr+3:target_obj_qposadr+7] = r2quat(rpy2r(np.radians([90, 0, 0])))
            while self.tick - start < end_tick:
                if self.tick - start > perturb_tick:
                    noise = np.random.normal(0, noise_scale, 4)
                    print(f"noise: {noise}")
                    self.data.qpos[target_obj_qposadr+3: target_obj_qposadr+7] += noise

                super().forward(q=init_q,joint_idxs=self.idxs_forward)
                super().step(ctrl=init_q,ctrl_idxs=self.idxs_step, nstep=nstep)
                p_obj, R_obj = super().get_pR_body(body_name=target_obj_name)
                quat_ = r2quat(R_obj)
                quat_list.append(quat_)
                p_list.append(p_obj)
                R_list.append(r2rpy(R_obj, unit='deg'))
                if self.MODE == 'window':
                    self.MODE == 'window'
                    super().plot_sphere(p=p_copy, r=0.01, rgba=[1,0,0,0.3])
                    super().render(render_every=10)
                elif self.MODE == 'offscreen':
                    self.MODE = 'offscreen'
                    # rgb, depth = super().grab_rgb_depth_img_offscreen()

            p_obj, R_obj = super().get_pR_body(body_name=target_obj_name)
            R_final = r2rpy(R_obj, unit='deg')
            p_list.append(p_obj)
            R_list.append(R_final)
            super().reset()

            if scene == 'kitchen_white':
                lower_bound = quat_bounds[0]    # 0.70
                upper_bound = quat_bounds[1]    # 0.79
                in_range = np.logical_and(np.array(quat_list)[5:][:,0] >= lower_bound, np.array(quat_list)[5:][:,0] <= upper_bound)
                all_in_range = np.all(in_range)
                if all_in_range:
                    print(f"Feasible: {p}")
                    self.feasible_place_positions.append(p)
                    self.quat_range = [np.max(np.array(quat_list)[5:][:,0]) - np.min(np.array(quat_list)[5:][:,0])]
                    self.quat_range_list.append(self.quat_range)
            elif scene == 'kitchen_small':
                lower_bound = quat_bounds[0]    # 0.3
                upper_bound = quat_bounds[1]    # 0.60
                in_range = np.logical_and(np.array(quat_list)[10:][:,0] >= lower_bound, np.array(quat_list)[10:][:,0] <= upper_bound)
                all_in_range = np.all(in_range)
                if all_in_range:
                    print(f"Feasible: {p}")
                    self.feasible_place_positions.append(p)
                    self.quat_range = [np.max(np.array(quat_list)[5:][:,0]) - np.min(np.array(quat_list)[5:][:,0])]
                    self.quat_range_list.append(self.quat_range)
            elif scene == 'kitchen_black':
                lower_bound = quat_bounds[0]    # 0.49
                upper_bound = quat_bounds[1]    # 0.56
                in_range = np.logical_and(np.array(quat_list)[10:][:,0] >= lower_bound, np.array(quat_list)[10:][:,0] <= upper_bound)
                all_in_range = np.all(in_range)
                if all_in_range:
                    print(f"Feasible: {p}")
                    self.feasible_place_positions.append(p)
                    self.quat_range = [np.max(np.array(quat_list)[5:][:,0]) - np.min(np.array(quat_list)[5:][:,0])]
                    self.quat_range_list.append(self.quat_range)
            elif scene == 'office_bookshelf':
                lower_bound = quat_bounds[0]    # 0.99
                upper_bound = quat_bounds[1]    # 1.00
                in_range = np.logical_and(np.array(quat_list)[5:][:,0] >= lower_bound, np.array(quat_list)[5:][:,0] <= upper_bound)
                wo_contact = np.logical_and(np.max(np.linalg.norm(np.array(f_contacts)[:,0])) >= 0, np.max(np.linalg.norm(np.array(f_contacts)[:,0])) <= 35)
                all_in_range = np.all(in_range)
                all_wo_contact = np.all(wo_contact)
                print(f"quat_max: {np.max(np.array(quat_list)[5:][:,0])}")
                print(f"quat_min: {np.min(np.array(quat_list)[5:][:,0])}")

                if all_in_range:
                    print(f"Feasible: {p}")
                    self.feasible_place_positions.append(p)
                    self.quat_range = [np.max(np.array(quat_list)[5:][:,0]) - np.min(np.array(quat_list)[5:][:,0])]
                    self.quat_range_list.append(self.quat_range)

            elif scene == 'category_tray' or scene =='category_pringles':
                lower_bound = quat_bounds[0]    # 0.49
                upper_bound = quat_bounds[1]    # 0.56
                in_range = np.logical_and(np.array(quat_list)[10:][:,0] >= lower_bound, np.array(quat_list)[10:][:,0] <= upper_bound)
                all_in_range = np.all(in_range)
                if all_in_range:
                    print(f"Feasible: {p}")
                    self.feasible_place_positions.append(p)
                    self.quat_range = [np.max(np.array(quat_list)[5:][:,0]) - np.min(np.array(quat_list)[5:][:,0])]
                    self.quat_range_list.append(self.quat_range)

            else:
                delta_R = np.diff(R_list, axis=0)
                norm_delta_R = np.linalg.norm(delta_R, axis=1)
                var_R = np.var(norm_delta_R)
                if var_R < var_threshold:
                    self.feasible_place_positions.append(p)
            if self.cnt == len(positions)-1:
                self.DONE_FLAG = True
            self.cnt += 1