import numpy as np 

def rot_e():
    e = np.array([[1, 	       0, 	      0],
             	  [0,          1,         0],
             	  [0,          0,         1]])
    return e


def rot_x(rad):
    roll = np.array([[1, 	       0, 	         0],
             		 [0, np.cos(rad), -np.sin(rad)],
             		 [0, np.sin(rad),  np.cos(rad)]])
    return roll 


def rot_y(rad):
    pitch = np.array([[np.cos(rad), 0, np.sin(rad)],
                      [0,		    1, 	         0],
                      [-np.sin(rad),0, np.cos(rad)]])
    return pitch


def rot_z(rad):
    yaw = np.array([[np.cos(rad), -np.sin(rad),  0],
         	        [np.sin(rad),  np.cos(rad),  0],
              		[0, 			         0,  1]])
    return yaw 


def Rotation_E(): 
    e = np.array([[1, 	       0, 	      0,    0],
             	  [0,          1,         0,    0],
             	  [0,          0,         1,    0],
             	  [0,		   0,	      0,    0]])
    return e


def Rotation_X(rad):
    roll = np.array([[1, 	       0, 	      0,    0],
             		 [0, np.cos(rad), -np.sin(rad), 0],
             		 [0, np.sin(rad),  np.cos(rad), 0],
             		 [0,		   0,	      0,    0]])
    return roll 


def Rotation_Y(rad):
    pitch = np.array([[np.cos(rad), 0, np.sin(rad), 0],
              		  [0,		    1, 	         0, 0],
              		  [-np.sin(rad),0, np.cos(rad), 0],
              		  [0, 		    0, 	         0, 0]])
    return pitch


def Rotation_Z(rad):
    yaw = np.array([[np.cos(rad), -np.sin(rad),  0, 0],
         	        [np.sin(rad),  np.cos(rad),  0, 0],
              		[0, 			         0,  1, 0],
             		[0, 			         0,  0, 0]])
    return yaw 

def Translation(x , y, z):
    Position = np.array([[0, 0, 0, x],
                         [0, 0, 0, y],
                         [0, 0, 0, z],
                         [0, 0, 0, 1]])
    return Position


def HT_matrix(Rotation, Position):
    Homogeneous_Transform = Rotation + Position
    return Homogeneous_Transform


def pr2t(position, rotation): 
    position_4diag  = np.array([[0, 0, 0, position[0]],
                                [0, 0, 0, position[1]],
                                [0, 0, 0, position[2]], 
                                [0, 0, 0, 1]], dtype=object)
    rotation_4diag  = np.append(rotation,[[0],[0],[0]], axis=1)
    rotation_4diag_ = np.append(rotation_4diag, [[0, 0, 0, 1]], axis=0)
    ht_matrix = position_4diag + rotation_4diag_ 
    return ht_matrix


def t2p(ht_matrix):
    return ht_matrix[:-1, -1]


def t2r(ht_matrix):
    return ht_matrix[:-1, :-1]


def make_rotation(rad=0):
    for idx, rad_num in enumerate(rad.split()):
        if idx == 0 and float(rad_num) !=0:
            idx0 = rot_x(float(rad_num))
        elif idx==0 and float(rad_num) == 0: 
            idx0 = rot_e()
        if idx == 1 and float(rad_num) !=0:
            idx1 = rot_y(float(rad_num))
        elif idx==1 and float(rad_num) == 0: 
            idx1 = rot_e()
        if idx == 2 and float(rad_num) !=0:
            idx2 = rot_z(float(rad_num))
        elif idx==2 and float(rad_num)==0: 
            idx2 = rot_e()
    rot = idx2.dot(idx1).dot(idx0) 
    return rot

