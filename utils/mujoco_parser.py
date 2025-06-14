import os

import cv2
import mujoco_viewer
import numpy as np
# import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils.util import (compute_view_params, get_rotation_matrix_from_two_points,
                  meters2xyz, pr2t, r2w, rpy2r, r2quat, trim_scale, sample_xyzs)
import mujoco
import ray

class MuJoCoParserClass(object):
    """
        MuJoCo Parser class
    """
    def __init__(self,name='Robot',rel_xml_path=None,USE_MUJOCO_VIEWER=False,VERBOSE=True,MODE="offscreen"):
        """
            Initialize MuJoCo parser
        """
        self.name         = name
        self.rel_xml_path = rel_xml_path
        self.VERBOSE      = VERBOSE
        self.MODE         = MODE
        # Constants
        self.tick         = 0
        self.render_tick  = 0
        # Parse an xml file
        if self.rel_xml_path is not None:
            self._parse_xml()
        # Viewer
        self.USE_MUJOCO_VIEWER = USE_MUJOCO_VIEWER
        if self.USE_MUJOCO_VIEWER==True and self.MODE=='window':
            self.init_viewer(MODE='window')
        elif self.MODE=='offscreen':
            self.init_viewer(MODE='offscreen')
        # Initial joint position
        self.qpos0 = self.data.qpos
        # Reset
        self.reset()
        # Print
        if self.VERBOSE:
            self.print_info()
        self.is_running = False

    def _parse_xml(self):
        """
            Parse an xml file
        """
        self.full_xml_path    = os.path.abspath(os.path.join(os.getcwd(),self.rel_xml_path))
        self.model            = mujoco.MjModel.from_xml_path(self.full_xml_path)
        self.data             = mujoco.MjData(self.model)
        self.dt               = self.model.opt.timestep
        self.HZ               = int(1/self.dt)
        self.n_geom           = self.model.ngeom # number of geometries
        self.geom_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,x)
                                for x in range(self.model.ngeom)]
        self.n_body           = self.model.nbody # number of bodies
        self.body_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,x)
                                for x in range(self.n_body)]
        self.body_name_idx    = [self.body_names.index(x) for x in self.body_names]
        self.n_dof            = self.model.nv # degree of freedom
        self.n_joint          = self.model.njnt     # number of joints 
        self.joint_names      = [mujoco.mj_id2name(self.model,mujoco.mjtJoint.mjJNT_HINGE,x)
                                 for x in range(self.n_joint)]
        self.joint_types      = self.model.jnt_type # joint types
        self.joint_ranges     = self.model.jnt_range # joint ranges
        self.rev_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names  = [self.joint_names[x] for x in self.rev_joint_idxs]
        self.n_rev_joint      = len(self.rev_joint_idxs)
        self.rev_joint_mins   = self.joint_ranges[self.rev_joint_idxs,0]
        self.rev_joint_maxs   = self.joint_ranges[self.rev_joint_idxs,1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins
        self.pri_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names  = [self.joint_names[x] for x in self.pri_joint_idxs]
        self.pri_joint_mins   = self.joint_ranges[self.pri_joint_idxs,0]
        self.pri_joint_maxs   = self.joint_ranges[self.pri_joint_idxs,1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins
        self.n_pri_joint      = len(self.pri_joint_idxs)
        # Actuator
        self.n_ctrl           = self.model.nu # number of actuators (or controls)
        self.ctrl_names       = []
        for addr in self.model.name_actuatoradr:
            ctrl_name = self.model.names[addr:].decode().split('\x00')[0]
            self.ctrl_names.append(ctrl_name) # get ctrl name
        self.ctrl_joint_idxs = []
        self.ctrl_joint_names = []
        for ctrl_idx in range(self.n_ctrl):
            transmission_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid # transmission index
            joint_idx = self.model.jnt_qposadr[transmission_idx][0] # index of the joint when the actuator acts on a joint
            self.ctrl_joint_idxs.append(joint_idx)
            self.ctrl_joint_names.append(self.joint_names[transmission_idx[0]])
        self.ctrl_qpos_idxs = self.ctrl_joint_idxs
        self.ctrl_qvel_idxs = []
        for ctrl_idx in range(self.n_ctrl):
            transmission_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid # transmission index
            joint_idx = self.model.jnt_dofadr[transmission_idx][0] # index of the joint when the actuator acts on a joint
            self.ctrl_qvel_idxs.append(joint_idx)
        self.ctrl_ranges      = self.model.actuator_ctrlrange # control range
        # Sensors
        self.n_sensor         = self.model.nsensor
        self.sensor_names     = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SENSOR,x)
                                for x in range(self.n_sensor)]
        # Site (sites are where sensors usually located)
        self.n_site           = self.model.nsite
        self.site_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SITE,x)
                                for x in range(self.n_site)]

        self.idxs_forward = [self.model.joint(joint_name).qposadr[0] for joint_name in self.rev_joint_names[:6]]
        self.idxs_jacobian = [self.model.joint(joint_name).dofadr[0] for joint_name in self.rev_joint_names[:6]]
        list1, list2 = self.ctrl_joint_idxs, self.idxs_forward
        self.idxs_step = []
        for i in range(len(list2)):
            if list2[i] in list1:
                self.idxs_step.append(list1.index(list2[i]))


    def print_info(self):
        """
            Printout model information
        """
        print ("dt:[%.4f] HZ:[%d]"%(self.dt,self.HZ))
        print ("n_dof (=nv):[%d]"%(self.n_dof))
        print ("n_geom:[%d]"%(self.n_geom))
        print ("geom_names:%s"%(self.geom_names))
        print ("n_body:[%d]"%(self.n_body))
        print ("body_names:%s"%(self.body_names))
        print ("n_joint:[%d]"%(self.n_joint))
        print ("joint_names:%s"%(self.joint_names))
        print ("joint_types:%s"%(self.joint_types))
        print ("joint_ranges:\n%s"%(self.joint_ranges))
        print ("n_rev_joint:[%d]"%(self.n_rev_joint))
        print ("rev_joint_idxs:%s"%(self.rev_joint_idxs))
        print ("rev_joint_names:%s"%(self.rev_joint_names))
        print ("rev_joint_mins:%s"%(self.rev_joint_mins))
        print ("rev_joint_maxs:%s"%(self.rev_joint_maxs))
        print ("rev_joint_ranges:%s"%(self.rev_joint_ranges))
        print ("n_pri_joint:[%d]"%(self.n_pri_joint))
        print ("pri_joint_idxs:%s"%(self.pri_joint_idxs))
        print ("pri_joint_names:%s"%(self.pri_joint_names))
        print ("pri_joint_mins:%s"%(self.pri_joint_mins))
        print ("pri_joint_maxs:%s"%(self.pri_joint_maxs))
        print ("pri_joint_ranges:%s"%(self.pri_joint_ranges))
        print ("n_ctrl:[%d]"%(self.n_ctrl))
        print ("ctrl_names:%s"%(self.ctrl_names))
        print ("ctrl_joint_idxs:%s"%(self.ctrl_joint_idxs))
        print ("ctrl_joint_names:%s"%(self.ctrl_joint_names))
        print ("ctrl_qvel_idxs:%s"%(self.ctrl_qvel_idxs))
        print ("ctrl_ranges:\n%s"%(self.ctrl_ranges))
        print ("n_sensor:[%d]"%(self.n_sensor))
        print ("sensor_names:%s"%(self.sensor_names))
        print ("n_site:[%d]"%(self.n_site))
        print ("site_names:%s"%(self.site_names))

    def init_viewer(self,viewer_title='MuJoCo',viewer_width=1200,viewer_height=800,viewer_hide_menus=True, MODE="offscreen"):
        """
            Initialize viewer
        """
        if MODE == "window":
            self.USE_MUJOCO_VIEWER = True
            self.viewer = mujoco_viewer.MujocoViewer(
                    self.model,self.data,mode='window',title=viewer_title,
                    width=viewer_width,height=viewer_height,hide_menus=viewer_hide_menus)
        elif MODE == "offscreen":
            self.viewer = mujoco_viewer.MujocoViewer(self.model,self.data,mode='offscreen')


    def update_viewer(self,azimuth=None,distance=None,elevation=None,lookat=None,
                      VIS_TRANSPARENT=None,VIS_CONTACTPOINT=None,
                      contactwidth=None,contactheight=None,contactrgba=None,
                      VIS_JOINT=None,jointlength=None,jointwidth=None,jointrgba=None,
                      CALL_MUJOCO_FUNC=True):
        """
            Initialize viewer
        """
        if azimuth is not None:
            self.viewer.cam.azimuth = azimuth
        if distance is not None:
            self.viewer.cam.distance = distance
        if elevation is not None:
            self.viewer.cam.elevation = elevation
        if lookat is not None:
            self.viewer.cam.lookat = lookat
        if VIS_TRANSPARENT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = VIS_TRANSPARENT
        if VIS_CONTACTPOINT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = VIS_CONTACTPOINT
        if contactwidth is not None:
            self.model.vis.scale.contactwidth = contactwidth
        if contactheight is not None:
            self.model.vis.scale.contactheight = contactheight
        if contactrgba is not None:
            self.model.vis.rgba.contactpoint = contactrgba
        if VIS_JOINT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = VIS_JOINT
        if jointlength is not None:
            self.model.vis.scale.jointlength = jointlength
        if jointwidth is not None:
            self.model.vis.scale.jointwidth = jointwidth
        if jointrgba is not None:
            self.model.vis.rgba.joint = jointrgba
        # Call MuJoCo functions for immediate modification
        if CALL_MUJOCO_FUNC:
            # Forward
            mujoco.mj_forward(self.model,self.data)
            # Update scene and render
            mujoco.mjv_updateScene(
                self.model,self.data,self.viewer.vopt,self.viewer.pert,self.viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,self.viewer.scn)
            mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)

    def get_viewer_cam_info(self,VERBOSE=False):
        """
            Get viewer cam information
        """
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance
        cam_elevation = self.viewer.cam.elevation
        cam_lookat    = self.viewer.cam.lookat.copy()
        if VERBOSE:
            print ("cam_azimuth:[%.2f] cam_distance:[%.2f] cam_elevation:[%.2f] cam_lookat:%s]"%
                (cam_azimuth,cam_distance,cam_elevation,cam_lookat))
        return cam_azimuth,cam_distance,cam_elevation,cam_lookat

    def is_viewer_alive(self):
        """
            Check whether a viewer is alive
        """
        return self.viewer.is_alive

    def reset(self):
        """
            Reset
        """
        mujoco.mj_resetData(self.model,self.data)
        # To initial position
        self.data.qpos = self.qpos0
        mujoco.mj_forward(self.model,self.data)
        self.tick        = 0
        self.render_tick = 0

    def step(self,ctrl=None,ctrl_idxs=None,nstep=1,INCREASE_TICK=True):
        """
            Forward dynamics
        """
        if ctrl is not None:
            if ctrl_idxs is None:
                self.data.ctrl[:] = ctrl
            else:
                self.data.ctrl[ctrl_idxs] = ctrl
        mujoco.mj_step(self.model,self.data,nstep=nstep)
        if INCREASE_TICK:
            self.tick = self.tick + 1*nstep

    def forward(self,q=None,joint_idxs=None,INCREASE_TICK=True):
        """
            Forward kinematics
        """
        if q is not None:
            if joint_idxs is not None:
                self.data.qpos[joint_idxs] = q
            else:
                self.data.qpos = q
        mujoco.mj_forward(self.model,self.data)
        if INCREASE_TICK:
            self.tick = self.tick + 1

    def get_sim_time(self):
        """
            Get simulation time (sec)
        """
        return self.data.time

    def render(self,render_every=1):
        """
            Render
        """
        # print(f"Here is: {self.USE_MUJOCO_VIEWER, self.MODE}")
        # if self.USE_MUJOCO_VIEWER and self.MODE == "window":
        if self.MODE == "window":
            if ((self.render_tick % render_every) == 0) or (self.render_tick == 0):
                self.viewer.render()
            self.render_tick = self.render_tick + 1
        elif self.MODE == "offscreen":
            rgbd = self.grab_rgb_depth_img_offscreen()
            self.render_tick = self.render_tick + 1
            return rgbd
        else:
            print ("[%s] Viewer NOT initialized."%(self.name))

    def grab_image(self,resize_rate=None,interpolation=cv2.INTER_NEAREST):
        """
            Grab the rendered iamge
        """
        img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)
        mujoco.mjr_readPixels(img, None,self.viewer.viewport,self.viewer.ctx)
        img = np.flipud(img) # flip image
        # Resize
        if resize_rate is not None:
            h = int(img.shape[0]*resize_rate)
            w = int(img.shape[1]*resize_rate)
            img = cv2.resize(img,(w,h),interpolation=interpolation)
        return img.copy()

    def close_viewer(self):
        """
            Close viewer
        """
        self.USE_MUJOCO_VIEWER = False
        self.viewer.close()

    def get_p_body(self,body_name):
        """
            Get body position
        """
        return self.data.body(body_name).xpos.copy()

    def get_R_body(self,body_name):
        """
            Get body rotation matrix
        """
        return self.data.body(body_name).xmat.reshape([3,3]).copy()

    def get_pR_body(self,body_name):
        """
            Get body position and rotation matrix
        """
        p = self.get_p_body(body_name)
        R = self.get_R_body(body_name)
        return p,R
    
    def get_p_joint(self,joint_name):
        """
            Get joint position
        """
        body_id = self.model.joint(joint_name).bodyid[0] # first body ID
        return self.get_p_body(self.body_names[body_id])

    def get_R_joint(self,joint_name):
        """
            Get joint rotation matrix
        """
        body_id = self.model.joint(joint_name).bodyid[0] # first body ID
        return self.get_R_body(self.body_names[body_id])
    
    def get_pR_joint(self,joint_name):
        """
            Get joint position and rotation matrix
        """
        p = self.get_p_joint(joint_name)
        R = self.get_R_joint(joint_name)
        return p,R
    
    def get_p_sensor(self,sensor_name):
        """
             Get sensor position
        """
        sensor_id = self.model.sensor(sensor_name).id # get sensor ID
        sensor_objtype = self.model.sensor_objtype[sensor_id] # get attached object type (i.e., site)
        sensor_objid = self.model.sensor_objid[sensor_id] # get attached object ID
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # get the site name
        p = self.data.site(site_name).xpos.copy() # get the position of the site
        return p
    
    def get_R_sensor(self,sensor_name):
        """
             Get sensor position
        """
        sensor_id = self.model.sensor(sensor_name).id
        sensor_objtype = self.model.sensor_objtype[sensor_id]
        sensor_objid = self.model.sensor_objid[sensor_id]
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid)
        R = self.data.site(site_name).xmat.reshape([3,3]).copy()
        return R
    
    def get_pR_sensor(self,sensor_name):
        """
            Get body position and rotation matrix
        """
        p = self.get_p_sensor(sensor_name)
        R = self.get_R_sensor(sensor_name)
        return p,R

    def get_q(self,joint_idxs=None):
        """
            Get joint position in (radian)
        """
        if joint_idxs is None:
            q = self.data.qpos
        else:
            q = self.data.qpos[joint_idxs]
        return q.copy()

    def get_J_body(self,body_name):
        """
            Get Jocobian matrices of a body
        """
        J_p = np.zeros((3,self.model.nv)) # nv: nDoF
        J_R = np.zeros((3,self.model.nv))
        mujoco.mj_jacBody(self.model,self.data,J_p,J_R,self.data.body(body_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_ik_ingredients(self,body_name,p_trgt=None,R_trgt=None,IK_P=True,IK_R=True, w_weight=1):
        """
            Get IK ingredients
        """
        J_p,J_R,J_full = self.get_J_body(body_name=body_name)
        p_curr,R_curr = self.get_pR_body(body_name=body_name)
        if (IK_P and IK_R):
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_full
            err   = np.concatenate((p_err,w_weight*w_err))
        elif (IK_P and not IK_R):
            p_err = (p_trgt-p_curr)
            J     = J_p
            err   = p_err
        elif (not IK_P and IK_R):
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_R
            err   = w_err
        else:
            J   = None
            err = None
        return J,err

    def damped_ls(self,J,err,eps=1e-6,stepsize=1.0,th=5*np.pi/180.0):
        """
            Dampled least square for IK
        """
        dq = stepsize*np.linalg.solve(a=(J.T@J)+eps*np.eye(J.shape[1]),b=J.T@err)
        dq = trim_scale(x=dq,th=th)
        return dq

    def onestep_ik(self,body_name,p_trgt=None,R_trgt=None,IK_P=True,IK_R=True,
                   joint_idxs=None,stepsize=1,eps=1e-1,th=5*np.pi/180.0):
        """
            Solve IK for a single step
        """
        J,err = self.get_ik_ingredients(
            body_name=body_name,p_trgt=p_trgt,R_trgt=R_trgt,IK_P=IK_P,IK_R=IK_R)
        dq = self.damped_ls(J,err,stepsize=stepsize,eps=eps,th=th)
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        q = self.get_q(joint_idxs=joint_idxs)
        q = q + dq[joint_idxs]
        # FK
        self.forward(q=q,joint_idxs=joint_idxs)
        return q, err
    
    def solve_ik(self,body_name,p_trgt,R_trgt,IK_P,IK_R,q_init,idxs_forward, idxs_jacobian,
                 RESET=False,DO_RENDER=False,render_every=1,th=1*np.pi/180.0,err_th=1e-6,w_weight=1.0, stepsize=1.0):
        """
            Solve IK
        """
        if RESET:
            self.reset()
        q_backup = self.get_q(joint_idxs=idxs_forward)
        q = q_init.copy()
        self.forward(q=q,joint_idxs=idxs_forward)
        tick = 0
        while True:
            tick = tick + 1
            J,err = self.get_ik_ingredients(
                body_name=body_name,p_trgt=p_trgt,R_trgt=R_trgt,IK_P=IK_P,IK_R=IK_R, w_weight=w_weight)
            dq = self.damped_ls(J,err,stepsize=stepsize,eps=1e-1,th=th)
            q = q + dq[idxs_jacobian]
            self.forward(q=q,joint_idxs=idxs_forward)
            # Terminate condition
            err_norm = np.linalg.norm(err)
            if err_norm < err_th:
                break
            # Render
            if DO_RENDER:
                if ((tick-1)%render_every) == 0:
                    p_tcp,R_tcp = self.get_pR_body(body_name=body_name)
                    self.plot_T(p=p_tcp,R=R_tcp,PLOT_AXIS=True,axis_len=0.1,axis_width=0.005)
                    self.plot_T(p=p_trgt,R=R_trgt,PLOT_AXIS=True,axis_len=0.2,axis_width=0.005)
                    self.render(render_every=render_every)
        # Back to back-uped position
        q_ik = self.get_q(joint_idxs=idxs_forward)
        self.forward(q=q_backup,joint_idxs=idxs_forward)
        return q_ik

    def solve_ik_repel(self,body_name,p_trgt,R_trgt,IK_P,IK_R,q_init,idxs_forward, idxs_jacobian,
                    RESET=False,DO_RENDER=False,render_every=1,th=1*np.pi/180.0,err_th=1e-6,w_weight=1.0, stepsize=1.0, eps=0.1,
                    repulse = 30, VERBOSE=False, inc_prefix = None, exc_prefix = None):
        """
            Solve IK
        """
        if RESET:
            self.reset()
        q_backup = self.get_q(joint_idxs=idxs_forward)
        q = q_init.copy()
        self.forward(q=q,joint_idxs=idxs_forward)
        tick = 0
        while True:
            tick = tick + 1
            J,err = self.get_ik_ingredients(
                body_name=body_name,p_trgt=p_trgt,R_trgt=R_trgt,IK_P=IK_P,IK_R=IK_R, w_weight=w_weight)
            dq = self.damped_ls(J,err,stepsize=stepsize,eps=eps,th=th)
            clipped_dq = np.clip(dq[idxs_jacobian], -0.1, 0.1)
            q = q + clipped_dq
            # limit with joint limits
            q = np.clip(q, self.joint_ranges[idxs_forward,0], self.joint_ranges[idxs_forward,1])
            self.forward(q=q,joint_idxs=idxs_forward)

            p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(must_include_prefix=inc_prefix, must_exclude_prefix=exc_prefix)

            body1s_ = [obj_ for obj_ in body1s if obj_ not in ["ur_rg2_gripper_finger1_finger_tip_link","ur_rg2_gripper_finger2_finger_tip_link"]]
            body2s_ = [obj_ for obj_ in body2s if obj_ not in ["ur_rg2_gripper_finger1_finger_tip_link","ur_rg2_gripper_finger2_finger_tip_link"]]
            
            if len(body1s_) > 0:
                if VERBOSE:
                    print(body1s_, body2s_)
                    print(f"Collision with {body1s_[0]} and {body2s_}")
                # clipping the gradient
                clipped_dq = np.clip(dq[idxs_jacobian], -0.1, 0.1)
                q = q - clipped_dq * repulse
                q = np.clip(q, self.joint_ranges[idxs_forward,0], self.joint_ranges[idxs_forward,1])
            
            # Terminate condition
            err_norm = np.linalg.norm(err)
            if err_norm < err_th:
                break
            # Render
            if DO_RENDER:
                if ((tick-1)%render_every) == 0:
                    p_tcp,R_tcp = self.get_pR_body(body_name=body_name)
                    self.plot_T(p=p_tcp,R=R_tcp,PLOT_AXIS=True,axis_len=0.1,axis_width=0.005)
                    self.plot_T(p=p_trgt,R=R_trgt,PLOT_AXIS=True,axis_len=0.2,axis_width=0.005)
                    self.render(render_every=render_every)
                if VERBOSE:
                    self.plot_T(p=np.array([0,0,2.5]),R=np.eye(3,3),
                                PLOT_AXIS=False,label='[%.4f] err'%(err_norm))
        # Back to back-uped position
        q_ik = self.get_q(joint_idxs=idxs_forward)
        self.forward(q=q_backup,joint_idxs=idxs_forward)
        
        return q_ik

    def solve_augmented_ik(self, ik_body_names, ik_p_trgts, ik_R_trgts,
                        IK_Ps, IK_Rs, q_init, idxs_forward, idxs_jacobian,
                        RESET=False,DO_RENDER=False,render_every=1,th=1*np.pi/180.0,err_th=1e-6):
        if RESET:
            self.reset()
        q_backup = self.get_q(joint_idxs=idxs_forward)
        q = q_init.copy()
        self.forward(q=q,joint_idxs=idxs_forward)
        tick = 0
        while True:
            tick = tick + 1
            # Numerical IK
            J_aug,err_aug = [],[]
            for ik_idx,ik_body_name in enumerate(ik_body_names):
                p_trgt,R_trgt = ik_p_trgts[ik_idx],ik_R_trgts[ik_idx]
                IK_P,IK_R = IK_Ps[ik_idx],IK_Rs[ik_idx]
                J,err = self.get_ik_ingredients(
                    body_name=ik_body_name,p_trgt=p_trgt,R_trgt=R_trgt,IK_P=IK_P,IK_R=IK_R)
                if (J is None) and (err is None): continue
                if len(J_aug) == 0:
                    J_aug,err_aug = J,err
                else:
                    J_aug   = np.concatenate((J_aug,J),axis=0)
                    err_aug = np.concatenate((err_aug,err),axis=0)
            dq = self.damped_ls(J_aug,err_aug,stepsize=1,eps=1e-1,th=5*np.pi/180.0)
            # Update q and FK
            q = q + dq[idxs_jacobian]
            self.forward(q=q,joint_idxs=idxs_forward)

            # Terminate condition
            err_norm = np.linalg.norm(err_aug)
            if err_norm < err_th:
                break
            # Render
            if DO_RENDER:
                if ((tick-1)%render_every) == 0:
                    for ik_idx,ik_body_name in enumerate(ik_body_names):
                        p_trgt,R_trgt = ik_p_trgts[ik_idx],ik_R_trgts[ik_idx]
                        IK_P,IK_R = IK_Ps[ik_idx],IK_Rs[ik_idx]
                        if (IK_P is None) and (IK_R is None): continue
                        self.plot_T(p=self.get_p_body(body_name=ik_body_name),R=self.get_R_body(body_name=ik_body_name),
                                PLOT_AXIS=IK_R,axis_len=0.2,axis_width=0.01,
                                PLOT_SPHERE=IK_P,sphere_r=0.05,sphere_rgba=[1,0,0,0.9],  label=f'augmented error: {np.linalg.norm(err_aug)}')
                        self.plot_T(p=p_trgt,R=R_trgt,
                                PLOT_AXIS=IK_R,axis_len=0.2,axis_width=0.01,
                                PLOT_SPHERE=IK_P,sphere_r=0.05,sphere_rgba=[0,0,1,0.9])
                    self.plot_T(p=[0,0,0],R=np.eye(3,3),PLOT_AXIS=True,axis_len=1.0)
                    self.render()
        # Back to back-uped position
        q_ik = self.get_q(joint_idxs=idxs_forward)
        self.forward(q=q_backup,joint_idxs=idxs_forward)

        return q_ik

    def plot_sphere(self,p,r,rgba=[1,1,1,1],label=''):
        """
            Add sphere
        """
        self.viewer.add_marker(
            pos   = p,
            size  = [r,r,r],
            rgba  = rgba,
            type  = mujoco.mjtGeom.mjGEOM_SPHERE,
            label = label)

    def plot_T(self,p,R,
               PLOT_AXIS=True,axis_len=1.0,axis_width=0.01,
               PLOT_SPHERE=False,sphere_r=0.05,sphere_rgba=[1,0,0,0.5],axis_rgba=None,
               label=None):
        """
            Plot coordinate axes
        """
        if PLOT_AXIS:
            if axis_rgba is None:
                rgba_x = [1.0,0.0,0.0,0.9]
                rgba_y = [0.0,1.0,0.0,0.9]
                rgba_z = [0.0,0.0,1.0,0.9]
            else:
                rgba_x = axis_rgba
                rgba_y = axis_rgba
                rgba_z = axis_rgba
            # X axis
            R_x = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([1,0,0]))
            p_x = p + R_x[:,2]*axis_len/2
            self.viewer.add_marker(
                pos   = p_x,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_x,
                rgba  = rgba_x,
                label = ''
            )
            R_y = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,1,0]))
            p_y = p + R_y[:,2]*axis_len/2
            self.viewer.add_marker(
                pos   = p_y,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_y,
                rgba  = rgba_y,
                label = ''
            )
            R_z = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,0,1]))
            p_z = p + R_z[:,2]*axis_len/2
            self.viewer.add_marker(
                pos   = p_z,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_z,
                rgba  = rgba_z,
                label = ''
            )
        if PLOT_SPHERE:
            self.viewer.add_marker(
                pos   = p,
                size  = [sphere_r,sphere_r,sphere_r],
                rgba  = sphere_rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = '')
        if label is not None:
            self.viewer.add_marker(
                pos   = p,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.01],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label)
            
    def plot_box(self,p=np.array([0,0,0]),R=np.eye(3),
                 xlen=1.0,ylen=1.0,zlen=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_BOX,
            size  = [xlen,ylen,zlen],
            rgba  = rgba,
            label = ''
        )
        
    def plot_capsule(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size  = [r,r,h],
            rgba  = rgba,
            label = ''
        )
        
    def plot_cylinder(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,h],
            rgba  = rgba,
            label = ''
        )
    
    def plot_ellipsoid(self,p=np.array([0,0,0]),R=np.eye(3),rx=1.0,ry=1.0,rz=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size  = [rx,ry,rz],
            rgba  = rgba,
            label = ''
        )
        
    def plot_arrow(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,h*2],
            rgba  = rgba,
            label = ''
        )

    def plot_arrow_uv(self,p,uv,r_stem=0.03,len_arrow=0.3,rgba=[1,0,0,1],label=''):
        """
            Plot arrow
        """
        p_a = np.copy(np.array([0,0,1]))
        p_b = np.copy(uv)
        p_a_norm = np.linalg.norm(p_a)
        p_b_norm = np.linalg.norm(p_b)
        if p_a_norm > 1e-9: p_a = p_a/p_a_norm
        if p_b_norm > 1e-9: p_b = p_b/p_b_norm
        v = np.cross(p_a,p_b)
        S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        if np.linalg.norm(v) == 0:
            R = np.eye(3,3)
        else:
            R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))

        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r_stem,r_stem,len_arrow],
            rgba  = rgba,
            label = label
        )

    def plot_line(self,p=np.array([0,0,0]),R=np.eye(3),h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = h,
            rgba  = rgba,
            label = ''
        )
        
    def plot_arrow_fr2to(self,p_fr,p_to,r=1.0,rgba=[0.5,0.5,0.5,0.5]):
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,np.linalg.norm(p_to-p_fr)*2],
            rgba  = rgba,
            label = ''
        )

    def plot_line_fr2to(self,p_fr,p_to,rgba=[0.5,0.5,0.5,0.5]):
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = np.linalg.norm(p_to-p_fr),
            rgba  = rgba,
            label = ''
        )
    
    def plot_cylinder_fr2to(self,p_fr,p_to,r=0.01,rgba=[0.5,0.5,0.5,0.5]):
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = (p_fr+p_to)/2,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,np.linalg.norm(p_to-p_fr)/2],
            rgba  = rgba,
            label = ''
        )
            
    def plot_body_T(self,body_name,
               PLOT_AXIS=True,axis_len=1.0,axis_width=0.01,
               PLOT_SPHERE=False,sphere_r=0.05,sphere_rgba=[1,0,0,0.5],axis_rgba=None,
               label=None):
        """
            Plot coordinate axes on a body
        """
        p,R = self.get_pR_body(body_name=body_name)
        self.plot_T(p,R,PLOT_AXIS=PLOT_AXIS,axis_len=axis_len,axis_width=axis_width,
                    PLOT_SPHERE=PLOT_SPHERE,sphere_r=sphere_r,sphere_rgba=sphere_rgba,axis_rgba=axis_rgba,
                    label=label)
        
    def plot_joint_T(self,joint_name,
               PLOT_AXIS=True,axis_len=1.0,axis_width=0.01,
               PLOT_SPHERE=False,sphere_r=0.05,sphere_rgba=[1,0,0,0.5],axis_rgba=None,
               label=None):
        """
            Plot coordinate axes on a joint
        """
        p,R = self.get_pR_joint(joint_name=joint_name)
        self.plot_T(p,R,PLOT_AXIS=PLOT_AXIS,axis_len=axis_len,axis_width=axis_width,
                    PLOT_SPHERE=PLOT_SPHERE,sphere_r=sphere_r,sphere_rgba=sphere_rgba,axis_rgba=axis_rgba,
                    label=label)

    def plot_arrow_contact(self,p,uv,r_arrow=0.03,h_arrow=0.3,rgba=[1,0,0,1],label=''):
        """
            Plot arrow
        """
        p_a = np.copy(np.array([0,0,1]))
        p_b = np.copy(uv)
        p_a_norm = np.linalg.norm(p_a)
        p_b_norm = np.linalg.norm(p_b)
        if p_a_norm > 1e-9: p_a = p_a/p_a_norm
        if p_b_norm > 1e-9: p_b = p_b/p_b_norm
        v = np.cross(p_a,p_b)
        S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        if np.linalg.norm(v) == 0:
            R = np.eye(3,3)
        else:
            R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))

        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r_arrow,r_arrow,h_arrow],
            rgba  = rgba,
            label = label
        )

    def get_body_names(self,prefix='obj_'):
        """
            Get body names with prefix
        """
        body_names = [x for x in self.body_names if x[:len(prefix)]==prefix]
        return body_names

    def get_contact_info(self,must_include_prefix=None,must_exclude_prefix=None):
        """
            Get contact information
        """
        p_contacts = []
        f_contacts = []
        geom1s = []
        geom2s = []
        body1s = []
        body2s = []
        for c_idx in range(self.data.ncon):
            contact   = self.data.contact[c_idx]
            # Contact position and frame orientation
            p_contact = contact.pos # contact position
            R_frame   = contact.frame.reshape(( 3,3))
            # Contact force
            f_contact_local = np.zeros(6,dtype=np.float64)
            mujoco.mj_contactForce(self.model,self.data,0,f_contact_local)
            f_contact = R_frame @ f_contact_local[:3] # in the global coordinate
            # Contacting geoms
            contact_geom1 = self.geom_names[contact.geom1]
            contact_geom2 = self.geom_names[contact.geom2]
            contact_body1 = self.body_names[self.model.geom_bodyid[contact.geom1]]
            contact_body2 = self.body_names[self.model.geom_bodyid[contact.geom2]]
            # Append
            if must_include_prefix is not None:
                if (contact_body1[:len(must_include_prefix)] == must_include_prefix) or (contact_body2[:len(must_include_prefix)] == must_include_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
                    body1s.append(contact_body1)
                    body2s.append(contact_body2)
            elif must_exclude_prefix is not None:
                if (contact_body1[:len(must_exclude_prefix)] != must_exclude_prefix) and (contact_body2[:len(must_exclude_prefix)] != must_exclude_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
                    body1s.append(contact_body1)
                    body2s.append(contact_body2)
            else:
                p_contacts.append(p_contact)
                f_contacts.append(f_contact)
                geom1s.append(contact_geom1)
                geom2s.append(contact_geom2)
                body1s.append(contact_body1)
                body2s.append(contact_body2)
        return p_contacts,f_contacts,geom1s,geom2s,body1s,body2s

    def plot_contact_info(self,must_include_prefix=None,h_arrow=0.3,rgba_arrow=[1,0,0,1],
                          PRINT_CONTACT_BODY=False,PRINT_CONTACT_GEOM=False,VERBOSE=False):
        """
            Plot contact information
        """
        # Get contact information
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        # Render contact informations
        for (p_contact,f_contact,geom1,geom2,body1,body2) in zip(p_contacts,f_contacts,geom1s,geom2s,body1s,body2s):
            f_norm = np.linalg.norm(f_contact)
            f_uv = f_contact / (f_norm+1e-8)
            # h_arrow = 0.3 # f_norm*0.05
            self.plot_arrow_contact(p=p_contact,uv=f_uv,r_arrow=0.01,h_arrow=h_arrow,rgba=rgba_arrow,
                        label='')
            self.plot_arrow_contact(p=p_contact,uv=-f_uv,r_arrow=0.01,h_arrow=h_arrow,rgba=rgba_arrow,
                        label='')
            if PRINT_CONTACT_BODY:
                label = '[%s]-[%s]'%(body1,body2)
            elif PRINT_CONTACT_GEOM:
                label = '[%s]-[%s]'%(geom1,geom2)
            else:
                label = '' 
            self.plot_sphere(p=p_contact,r=0.02,rgba=[1,0.2,0.2,1],label=label)
        # Print
        if VERBOSE:
            self.print_contact_info(must_include_prefix=must_include_prefix)
            
    def print_contact_info(self,must_include_prefix=None):
        """ 
            Print contact information
        """
        # Get contact information
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        for (p_contact,f_contact,geom1,geom2,body1,body2) in zip(p_contacts,f_contacts,geom1s,geom2s,body1s,body2s):
            print ("Tick:[%d] Body contact:[%s]-[%s]"%(self.tick,body1,body2))

    def open_interactive_viewer(self):
        """
            Open interactive viewer
        """
        from mujoco import viewer
        viewer.launch(self.model)

    def get_T_viewer(self,fovy=45):
        """
            Get viewer pose
        """
        cam_lookat    = self.viewer.cam.lookat
        cam_elevation = self.viewer.cam.elevation
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance

        p_lookat = cam_lookat
        R_lookat = rpy2r(np.deg2rad([0,-cam_elevation,cam_azimuth]))
        T_lookat = pr2t(p_lookat,R_lookat)
        T_viewer = T_lookat @ pr2t(np.array([-cam_distance,0,0]),np.eye(3))
        return T_viewer

    def grab_rgb_depth_img_offscreen(self):
        """
            Grab RGB and Depth images in offscreen mode
        """
        assert self.MODE == 'offscreen'
        # viewer_azimuth,viewer_distance,viewer_elevation,viewer_lookat = self.get_viewer_cam_info()
        self.update_viewer(azimuth=30,distance=3.0,
                            elevation=-30,lookat=[1, 0, 0.71])
        offscreen_rgb_img, offscreen_depth_img = self.viewer.read_pixels(depth=True)

        # Rescale depth image
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        scaled_depth_img = near / (1 - offscreen_depth_img * (1 - near / far))
        offscreen_depth_img = scaled_depth_img.squeeze()
        return offscreen_rgb_img, offscreen_depth_img


    def grab_rgb_depth_img(self):
        """
            Grab RGB and Depth images
        """
        rgb_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        depth_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_img,depth_img,self.viewer.viewport,self.viewer.ctx)
        rgb_img,depth_img = np.flipud(rgb_img),np.flipud(depth_img)

        # Rescale depth image
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        scaled_depth_img = near / (1 - depth_img * (1 - near / far))
        depth_img = scaled_depth_img.squeeze()
        return rgb_img,depth_img
    
    def get_pcd_from_depth_img(self,depth_img,fovy=45):
        """
            Get point cloud data from depth image
        """
        # Get camera pose
        T_viewer = self.get_T_viewer(fovy=fovy)

        # Camera intrinsic
        img_height = depth_img.shape[0]
        img_width = depth_img.shape[1]
        focal_scaling = 0.5*img_height/np.tan(fovy*np.pi/360)
        cam_matrix = np.array(((focal_scaling,0,img_width/2),
                            (0,focal_scaling,img_height/2),
                            (0,0,1)))

        # Estimate 3D point from depth image
        xyz_img = meters2xyz(depth_img,cam_matrix) # [H x W x 3]
        xyz_transpose = np.transpose(xyz_img,(2,0,1)).reshape(3,-1) # [3 x N]
        xyzone_transpose = np.vstack((xyz_transpose,np.ones((1,xyz_transpose.shape[1])))) # [4 x N]

        # To world coordinate
        xyzone_world_transpose = T_viewer @ xyzone_transpose
        xyz_world_transpose = xyzone_world_transpose[:3,:] # [3 x N]
        xyz_world = np.transpose(xyz_world_transpose,(1,0)) # [N x 3]
        return xyz_world,xyz_img
    
    def get_egocentric_rgb_depth_pcd(self,p_ego=None,p_trgt=None,rsz_rate=50,fovy=45,
                                     BACKUP_AND_RESTORE_VIEW=False):
        """
            Get egocentric 1) RGB image, 2) Depth image, 3) Point Cloud Data
        """
        if BACKUP_AND_RESTORE_VIEW:
            # Backup camera information
            viewer_azimuth,viewer_distance,viewer_elevation,viewer_lookat = self.get_viewer_cam_info()

        if (p_ego is not None) and (p_trgt is not None):
            cam_azimuth,cam_distance,cam_elevation,cam_lookat = compute_view_params(
                camera_pos=p_ego,target_pos=p_trgt,up_vector=np.array([0,0,1]))
            self.update_viewer(azimuth=cam_azimuth,distance=cam_distance,
                               elevation=cam_elevation,lookat=cam_lookat)
        
        # Grab RGB and depth image
        rgb_img,depth_img = self.grab_rgb_depth_img() # get rgb and depth images

        # Resize
        if rsz_rate is not None:
            h_rsz,w_rsz = depth_img.shape[0]//rsz_rate,depth_img.shape[1]//rsz_rate
            depth_img_rsz = cv2.resize(depth_img,(w_rsz,h_rsz),interpolation=cv2.INTER_NEAREST)
        else:
            depth_img_rsz = depth_img
        # Get PCD
        pcd,xyz_img = self.get_pcd_from_depth_img(depth_img_rsz,fovy=fovy) # [N x 3]

        if BACKUP_AND_RESTORE_VIEW:
            # Restore camera information
            self.update_viewer(azimuth=viewer_azimuth,distance=viewer_distance,
                               elevation=viewer_elevation,lookat=viewer_lookat)
        return rgb_img,depth_img,pcd,xyz_img

    def place_objects_random(self, n_obj, obj_names, x_range=[0.75, 1.35], y_range=[-0.38,0.38],z_range=[0.81,0.81], min_dist=0.1, COLORS=False, VERBOSE=False):
        xyzs = sample_xyzs(n_sample=n_obj,
                    x_range=x_range,y_range=y_range,z_range=z_range,min_dist=min_dist)
        # if COLORS:
        #     colors = np.array([plt.cm.gist_rainbow(x) for x in np.linspace(0,1,n_obj)])

        for obj_idx, obj_name in enumerate(obj_names):
            if obj_idx == (n_obj):
                break
            jntadr = self.model.body(obj_name).jntadr[0]
            self.model.joint(jntadr).qpos0[:3] = xyzs[obj_idx, :]
            # if COLORS:
            #     geomadr = self.model.body(obj_name).geomadr[0]
            #     self.model.geom(geomadr).rgba = colors[obj_idx]
        
        if VERBOSE:
            for obj_idx, obj_name in enumerate(obj_names):
                if obj_idx == (n_obj):
                    break
                print(f"{obj_name}: {xyzs[obj_idx, :]}")

    def set_objects(self, xyzs_dict, COLORS=False, VERBOSE=True):
        """Set objects to the given positions."""
        n_obj = len(xyzs_dict.keys())
        # if COLORS:
        #     colors = np.array([plt.cm.gist_rainbow(x) for x in np.linspace(0,1,n_obj)])

        for obj_idx, obj_name in enumerate(xyzs_dict.keys()):
            jntadr = self.model.body(obj_name).jntadr[0]
            self.model.joint(jntadr).qpos0[:3] = xyzs_dict[obj_name]
            if COLORS:
                geomadr = self.model.body(obj_name).geomadr[0]
                self.model.geom(geomadr).rgba = colors[obj_idx]

        if VERBOSE:
            for obj_idx, obj_name in enumerate(xyzs_dict.keys()):
                if obj_idx == (n_obj):
                    break
                print(f"{obj_name}: {xyzs_dict[obj_name]}")

    def get_tick(self):
        """
            Get tick
        """
        tick = int(self.get_sim_time()/self.dt)
        return tick

    def loop_every(self,HZ=1):
        """
            Loop every
        """
        # tick = int(self.get_sim_time()/self.dt)
        FLAG = (self.tick-1)%(int(1/self.dt/HZ))==0
        return FLAG
    
    def get_sensor_value(self,sensor_name):
        """
            Read sensor value
        """
        data = self.data.sensor(sensor_name).data
        return data.copy()

    def get_sensor_values(self,sensor_names=None):
        """
            Read multiple sensor values
        """
        if sensor_names is None:
            sensor_names = self.sensor_names
        data = np.array([self.get_sensor_value(sensor_name) for sensor_name in self.sensor_names]).squeeze()
        return data.copy()
    
    def get_qpos_joint(self,joint_name):
        """
            Get joint position
        """
        addr = self.model.joint(joint_name).qposadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        qpos = self.data.qpos[addr:addr+L]
        return qpos
    
    def get_qvel_joint(self,joint_name):
        """
            Get joint velocity
        """
        addr = self.model.joint(joint_name).dofadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        if L > 1: L = 6
        qvel = self.data.qvel[addr:addr+L]
        return qvel
    
    def get_qpos_joints(self,joint_names):
        """
            Get multiple joint positions from 'joint_names'
        """
        return np.array([self.get_qpos_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def get_qvel_joints(self,joint_names):
        """
            Get multiple joint velocities from 'joint_names'
        """
        return np.array([self.get_qvel_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def viewer_pause(self):
        """
            Viewer pause
        """
        self.viewer._paused = True
        
    def viewer_resume(self):
        """
            Viewer resume
        """
        self.viewer._paused = False
        
    """
        RRT-Star feasibility checker functions
    """
    def is_point_feasible(self,point_sample):
        q_bu = self.get_q(joint_idxs=self.idxs_jacobian)  # q_backup
        self.forward(q=point_sample,joint_idxs=self.idxs_forward)
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(must_include_prefix='ur_', must_exclude_prefix=None)
        # self.forward(q=q_bu,joint_idxs=idxs_forward)
        if len(p_contacts) == 0: # exclude about rg2 gripper tip contact.
            # print(geom1s, geom2s)
            # print(body1s, body2s)
            return True
        else:
            # print(f"Collision Detect! {geom1s} and {geom2s} were collided.")
            return False
    def is_point_to_point_connectable(self,point_fr,point_to):
        for alpha in np.linspace(0,1,5):
            point_check = alpha*point_fr + (1-alpha)*point_to
            if not self.is_point_feasible(point_check):
                return False
        return True

    def set_figure(self, fig_idx=0, title=None, line_name=None, xlabel=None, rgba=np.array([0,0,0,0]), gridsize=(5,5), legend=True):
        """
            Set figure using mjvFigure.
        """
        self.viewer.add_line_to_fig(line_name=line_name, fig_idx=fig_idx)
        figure = self.viewer.figs[0]
        figure.title = title
        figure.xlabel = xlabel
        figure.gridsize = gridsize
        figure.figurergba = rgba
        figure.flg_legend = legend

        return figure

    def plot_data(self, name="None", data=None, fig_idx=0):
        """
            Plot data on the figure.
        """
        self.viewer.add_data_to_line(line_name=name, line_data=data, fig_idx=fig_idx)

    def set_office_scene(self):
        # Move tables and robot base
        self.model.body('base_table').pos = np.array([0,0,0])
        self.model.body('avoiding_object_table').pos = np.array([0.38+0.45,0,0])
        self.model.body('base').pos = np.array([0.18,0,0.79])
        self.model.body('right_object_table').pos = np.array([-0.05,-0.80,0])
        self.model.body('left_object_table').pos = np.array([-0.05,0.80,0])

        jntadr_mug_cup = self.model.body('mug_cup').jntadr[0]
        self.model.joint(jntadr_mug_cup).qpos0[:3] = np.array([0.5, -0.4, 0.81])
        self.model.joint(jntadr_mug_cup).qpos0[3:] = r2quat(rpy2r(np.radians([0, 0, np.random.rand() * 360])))
        self.model.body('shelf_row1').pos = np.array([0.8,0.3,0.77])
        self.model.body('shelf_row1').quat = r2quat(rpy2r(np.radians([0, 0, 90])))

        obj_box_names = [body_name for body_name in self.body_names
                    if body_name is not None and (body_name.startswith("obj_box"))]

        # Place objects
        n_box_obj = len(obj_box_names)
        self.place_objects_random(n_obj=n_box_obj, obj_names=obj_box_names, x_range=[0.80, 1.15], y_range=[-3.15, -2.15], COLORS=False, VERBOSE=True)

        self.model.joint(self.model.body('book1').jntadr[0]).qpos0[:3] = np.array([0.82, 0.38, 0.85])
        self.model.joint(self.model.body('book2').jntadr[0]).qpos0[:3] = np.array([0.82, 0.50, 0.87])
        self.model.joint(self.model.body('book3').jntadr[0]).qpos0[:3] = np.array([0.82, 0.42, 0.90])
        self.model.joint(self.model.body('book1').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.array([1.57, 0, 0])))
        self.model.joint(self.model.body('book2').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.array([1.57, 0, 0])))
        self.model.joint(self.model.body('book3').jntadr[0]).qpos0[3:] = r2quat(rpy2r(np.array([1.57, 0, 0])))

        jntadr_coke = self.model.body('coke').jntadr[0]
        self.model.joint(jntadr_coke).qpos0[:3] = np.array([0.98,-0.15,0.79])

        # Apply changes
        self.reset()