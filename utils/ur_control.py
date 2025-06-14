import sys
sys.path.append("..")
import numpy as np 
import math 
from math import pi
import time s
""" FOR MODERN DRIVER """
import roslib; roslib.load_manifest('ur_driver')
import rospy
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
""" FOR ONROBOT RG2 """
from pymodbus.client.sync import ModbusTcpClient

class RealRobot:
    def __init__(self):
        self.client      = None
        self.JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.arm_pub     = rospy.Publisher('arm_controller/command', JointTrajectory, queue_size = 10)
        self.up_offset   = np.array([0, 0, 0.25])        
        self.client      = None 
        
    def rest_pose(self):
        try:  
            q = [-4.55611547e-01, -1.1069, 2.46218, -1.35530, 1.11438, -1.59319056e-03]
            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.JOINT_NAMES
            joint_states = rospy.wait_for_message("joint_states", JointState)
            joints_pos   = joint_states.position
            g.trajectory.points = [
                JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=q, velocities=[0]*6, time_from_start=rospy.Duration(3))]  
            self.client.send_goal(g)
            self.client.wait_for_result()
        except KeyboardInterrupt:
            self.client.cancel_goal()
            raise
        except:
            raise

    def capture_pose(self):
        try: 
            q = [(-90)/180*math.pi, (-132.46)/180*math.pi, (122.85)/180*math.pi, (99.65)/180*math.pi, (45)/180*math.pi, (-90.02)/180*math.pi]
            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.JOINT_NAMES
            joint_states = rospy.wait_for_message("joint_states", JointState)
            joints_pos   = joint_states.position
            g.trajectory.points = [
                JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=q, velocities=[0]*6, time_from_start=rospy.Duration(3))]  
            self.client.send_goal(g)
            self.client.wait_for_result()
        except KeyboardInterrupt:
            self.client.cancel_goal()
            raise
        except:
            raise

    def capture_pose2(self):
        try: 
            q = [(-180)/180*math.pi, (-132.46)/180*math.pi, (122.85)/180*math.pi, (99.65)/180*math.pi, (45)/180*math.pi, (-90.02)/180*math.pi]
            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.JOINT_NAMES
            joint_states = rospy.wait_for_message("joint_states", JointState)
            joints_pos   = joint_states.position
            g.trajectory.points = [
                JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=q, velocities=[0]*6, time_from_start=rospy.Duration(3))]  
            self.client.send_goal(g)
            self.client.wait_for_result()
        except KeyboardInterrupt:
            self.client.cancel_goal()
            raise
        except:
            raise

    def pnp_prepose(self):
        try: 
            q = [(-77.51)/180*math.pi, (-107.42)/180*math.pi, (116.83)/180*math.pi, (42.62)/180*math.pi, (95.31)/180*math.pi, (6.46)/180*math.pi]
            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.JOINT_NAMES
            joint_states = rospy.wait_for_message("joint_states", JointState)
            joints_pos   = joint_states.position
            g.trajectory.points = [
                JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=q, velocities=[0]*6, time_from_start=rospy.Duration(3))]  
            self.client.send_goal(g)
            self.client.wait_for_result()
        except KeyboardInterrupt:
            self.client.cancel_goal()
            raise
        except:
            raise

    def pnp_grasp1(self):
        try: 
            q = [(-84.27)/180*math.pi, (-34.48)/180*math.pi, (34.14)/180*math.pi, (90.57)/180*math.pi, (89.35)/180*math.pi, (-174.77)/180*math.pi]
            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.JOINT_NAMES
            joint_states = rospy.wait_for_message("joint_states", JointState)
            joints_pos   = joint_states.position
            g.trajectory.points = [
                JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=q, velocities=[0]*6, time_from_start=rospy.Duration(3))]  
            self.client.send_goal(g)
            self.client.wait_for_result()
        except KeyboardInterrupt:
            self.client.cancel_goal()
            raise
        except:
            raise

    def pnp_grasp2(self):
        try: 
            q = [(-84.30)/180*math.pi, (-26.78)/180*math.pi, (33.9)/180*math.pi, (86.85)/180*math.pi, (88.77)/180*math.pi, (-174.75)/180*math.pi]
            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.JOINT_NAMES
            joint_states = rospy.wait_for_message("joint_states", JointState)
            joints_pos   = joint_states.position
            g.trajectory.points = [
                JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=q, velocities=[0]*6, time_from_start=rospy.Duration(3))]  
            self.client.send_goal(g)
            self.client.wait_for_result()
        except KeyboardInterrupt:
            self.client.cancel_goal()
            raise
        except:
            raise

    def start_pose(self):
        try: 
            q = [-3.35916187e-01, -13.90421628e-01,  2.52584599e+00, -1.13542436e+00, 1.23408381e+00, -1.59279665e-03]
            g = FollowJointTrajectoryGoal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = self.JOINT_NAMES
            joint_states = rospy.wait_for_message("joint_states", JointState)
            joints_pos   = joint_states.position
            g.trajectory.points = [
                JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                JointTrajectoryPoint(positions=q, velocities=[0]*6, time_from_start=rospy.Duration(3))]  
            self.client.send_goal(g)
            self.client.wait_for_result()
        except KeyboardInterrupt:
            self.client.cancel_goal()
            raise
        except:
            raise

    def move_trajectory(self, joint_list):
        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = self.JOINT_NAMES
        for i, q in enumerate(joint_list):
            if i==0:
                joint_states = rospy.wait_for_message("joint_states", JointState)
                joints_pos   = joint_states.position
                g.trajectory.points = [
                    JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                    JointTrajectoryPoint(positions=q, velocities=[0]*6, time_from_start=rospy.Duration(3))]  
                d=3
            else:
                vel = (q-prev_q) #num_interpol # TODO: CHECK VELOCITY
                g.trajectory.points.append(
                    JointTrajectoryPoint(positions=q, velocities=vel,time_from_start=rospy.Duration(d))) 
            prev_q = q
            d+=0.002
        try:
            print("MOVE")
            self.client.send_goal(g)
            self.client.wait_for_result()
        except:
            raise

    def move_slow_trajectory(self, joint_list):
        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = self.JOINT_NAMES
        for i, q in enumerate(joint_list):
            if i==0:
                joint_states = rospy.wait_for_message("joint_states", JointState)
                joints_pos   = joint_states.position
                g.trajectory.points = [
                    JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6, time_from_start=rospy.Duration(0.0)),
                    JointTrajectoryPoint(positions=q, velocities=[0]*6, time_from_start=rospy.Duration(3))]  
                d=3
            else:
                vel = (q-prev_q) #num_interpol # TODO: CHECK VELOCITY
                g.trajectory.points.append(
                    JointTrajectoryPoint(positions=q, velocities=vel,time_from_start=rospy.Duration(d))) 
            prev_q = q
            d+=0.007
        try:
            print("MOVE")
            self.client.send_goal(g)
            self.client.wait_for_result()
        except:
            raise

    def main(self, joint_list=None): 
        try: 
            # rospy.init_node("test_move", anonymous=True, disable_signals=True)
            self.client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
            print("Waiting for server...")
            self.client.wait_for_server()
            print("Connected to server")
            parameters = rospy.get_param(None)
            index = str(parameters).find('prefix')
            if (index > 0):
                prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
                for i, name in enumerate(self.JOINT_NAMES):
                    self.JOINT_NAMES[i] = prefix + name
            self.move_trajectory(joint_list=joint_list)

            # order = input("Continue? y/n: ")
            # if order=="y": 
            #     self.move_trajectory(joint_list=joint_list)
            # else: 
            #     print("Stop")
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise


    def main_slow(self, joint_list=None): 
        try: 
            # rospy.init_node("test_move", anonymous=True, disable_signals=True)
            self.client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
            print("Waiting for server...")
            self.client.wait_for_server()
            print("Connected to server")
            parameters = rospy.get_param(None)
            index = str(parameters).find('prefix')
            if (index > 0):
                prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
                for i, name in enumerate(self.JOINT_NAMES):
                    self.JOINT_NAMES[i] = prefix + name
            self.move_slow_trajectory(joint_list=joint_list)

            # order = input("Continue? y/n: ")
            # # if order=="y": 
            #     self.move_slow_trajectory(joint_list=joint_list)
            # else: 
            #     print("Stop")
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise

    def move_capture_pose(self): 
        try: 
            # rospy.init_node("test_move", anonymous=True, disable_signals=True)
            self.client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
            print("Waiting for server...")
            self.client.wait_for_server()
            print("Connected to server")
            parameters = rospy.get_param(None)
            index = str(parameters).find('prefix')
            if (index > 0):
                prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
                for i, name in enumerate(self.JOINT_NAMES):
                    self.JOINT_NAMES[i] = prefix + name
            self.capture_pose()

            # order = input("Continue? y/n: ")
            # if order=="y": 
            #     print("Press Y")
            #     self.capture_pose()
            # else: 
            #     print("Stop")
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise

    def move_capture_pose2(self): 
        try: 
            # rospy.init_node("test_move", anonymous=True, disable_signals=True)
            self.client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
            print("Waiting for server...")
            self.client.wait_for_server()
            print("Connected to server")
            parameters = rospy.get_param(None)
            index = str(parameters).find('prefix')
            if (index > 0):
                prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
                for i, name in enumerate(self.JOINT_NAMES):
                    self.JOINT_NAMES[i] = prefix + name
            self.capture_pose2()

            # order = input("Continue? y/n: ")
            # if order=="y": 
            #     print("Press Y")
            #     self.capture_pose2()
            # else: 
            #     print("Stop")
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise

    def move_pnp_prepose(self): 
        try: 
            # rospy.init_node("test_move", anonymous=True, disable_signals=True)
            self.client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
            print("Waiting for server...")
            self.client.wait_for_server()
            print("Connected to server")
            parameters = rospy.get_param(None)
            index = str(parameters).find('prefix')
            if (index > 0):
                prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
                for i, name in enumerate(self.JOINT_NAMES):
                    self.JOINT_NAMES[i] = prefix + name
            self.pnp_prepose()

            # order = input("Continue? y/n: ")
            # if order=="y": 
            #     print("Press Y")
            #     self.pnp_prepose()
            # else: 
            #     print("Stop")
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise

    def move_pnp_grasp1(self): 
        try: 
            # rospy.init_node("test_move", anonymous=True, disable_signals=True)
            self.client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
            print("Waiting for server...")
            self.client.wait_for_server()
            print("Connected to server")
            parameters = rospy.get_param(None)
            index = str(parameters).find('prefix')
            if (index > 0):
                prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
                for i, name in enumerate(self.JOINT_NAMES):
                    self.JOINT_NAMES[i] = prefix + name
            self.pnp_grasp1()
            # order = input("Continue? y/n: ")
            # if order=="y": 
            #     print("Press Y")
            #     self.pnp_grasp1()
            # else: 
            #     print("Stop")
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise

    def move_pnp_grasp2(self): 
        try: 
            # rospy.init_node("test_move", anonymous=True, disable_signals=True)
            self.client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
            print("Waiting for server...")
            self.client.wait_for_server()
            print("Connected to server")
            parameters = rospy.get_param(None)
            index = str(parameters).find('prefix')
            if (index > 0):
                prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
                for i, name in enumerate(self.JOINT_NAMES):
                    self.JOINT_NAMES[i] = prefix + name
            self.pnp_grasp2()
            # order = input("Continue? y/n: ")
            # if order=="y": 
            #     print("Press Y")
            #     self.pnp_grasp2()
            # else: 
            #     print("Stop")
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise

    def move_rest_pose(self): 
        try: 
            # rospy.init_node("test_move", anonymous=True, disable_signals=True)
            self.client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
            print("Waiting for server...")
            self.client.wait_for_server()
            print("Connected to server")
            parameters = rospy.get_param(None)
            index = str(parameters).find('prefix')
            if (index > 0):
                prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
                for i, name in enumerate(self.JOINT_NAMES):
                    self.JOINT_NAMES[i] = prefix + name
            self.rest_pose()

            # order = input("Continue? y/n: ")
            # if order=="y": 
            #     print("Press Y")
            #     self.rest_pose()
            # else: 
            #     print("Stop")
        except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise

    def gripper_open(self, force,width,onrobot_ip='192.168.1.1'):
        graspclient = ModbusTcpClient(onrobot_ip)
        slave = 65
        graspclient.write_registers(0,[force,width,1],unit=slave)
        time.sleep(1)
        graspclient.close()
    
    def gripper_close(self, force,width,onrobot_ip='192.168.1.1'):
        graspclient = ModbusTcpClient(onrobot_ip)
        slave = 65
        graspclient.write_registers(0,[force,width,1],unit=slave)
        time.sleep(1)
        graspclient.close()

def grasped(graspclient):
    slave = 65
    flag = graspclient.read_input_registers(268,1,unit=slave).registers[0]
    flag = (flag&0x02) == 2

    if flag:
        print("Grasp detected: True")
    return flag

def Graspable(graspclient):
    slave = 65
    flag = graspclient.read_input_registers(268,1,unit=slave).registers[0]
    flag = (flag&0x08) == 8

    if flag:
        print("Grasp availablity: False")
    
    return flag

def resetTool(graspclient):
    print('Tool reseting')
    toolslave = 63
    graspclient.write_register(0,2,unit=toolslave)
    time.sleep(3)
    print("Reset Fininshed", end='\r')


def closeGrasp(force,width,graspclient):
    # If grasped, reset&openGrasp
    if grasped(graspclient):
        resetTool(graspclient)
        openGrasp(400,1000,graspclient)
    # If S1 activated, reset
    if Graspable(graspclient):
        resetTool(graspclient)
    slave = 65
    graspclient.write_registers(0,[force,width,1],unit=slave)
    time.sleep(1)

def openGrasp(force,width,graspclient):
    # If S1 activated, reset
    FLAG = True 
    while FLAG:
        FAIL = Graspable(graspclient)
        if FAIL:
            resetTool(graspclient)
        else: 
            FLAG=False
    # if Graspable(graspclient):
    #     resetTool(graspclient)
    slave = 65
    graspclient.write_registers(0,[force,width,1],unit=slave)
    time.sleep(1)

