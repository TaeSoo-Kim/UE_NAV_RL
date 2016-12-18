import sys
from unrealcv import client
import matplotlib.pyplot as plt
import pdb
import cv2
from math import cos,sin,radians
import numpy as np

## globals
reward = -0.01 # default to -0.01 because every step counts as penalty:-0.01
game_over = 0

def reward_listener(message):
  global reward
  # message should be : "Reward TYPE"
  print 'Got server message %s' % repr(message)
  
  
  if "COLLISION" in message:
    reward = -1.0
  elif "INSIGHT" in message:
    reward = 1.0
  elif "GAME_OVER" in message:
    game_over = 1
    reward = 10.0
  else: ## every step is -0.1,  try to find the shortest path
    reward = -0.1

  

## World object. This represents the state of the UnrealEngine game environment.
class World(object):
  def __init__(self):
    global reward, game_over
    reward = 0
    game_over = 0
    
    client.connect()
    client.message_handler = reward_listener
    if not client.isconnected():
      print 'UnrealCV server is not running. Gaming running?'
      sys.exit()
    ## System Status
    res = client.request('vget /unrealcv/status')
    print "Connection?? :",res
  

  def observe(self):
    ## get current frame
    #UE4_root = '/home/tk/dev/ThirdParty/UE4.13/UnrealEngine/'
    img_out_path = '/home/tk/dev/unrealcv/src/temp_imgs/img.png'
    res = client.request('vget /camera/0/lit ' + img_out_path )
    img = cv2.resize(cv2.imread(img_out_path), (224,224))
    return img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))

  def act(self,action):
    # 0:go_front, 1:go_back, 2:turn_left, 3:turn_right

    ## Try to move -> collision check -> undo move
    if action == 0:
      self._move_forward()
    elif action == 1:
      self._move_backward()
    elif action == 2:
      self._rotate_left()
    elif action == 3:
      self._rotate_right()
    
    return self.observe(), self._get_reward(), self._is_over()

  def reset(self):
    ## The game can reset itself once the trigger is set off.
    pass

  def _get_reward(self):
    global reward
    ###################
    # copy reward, set it back to -0.1, return the copy

    return_reward_val = reward
    reward = -0.01
    return return_reward_val

  

  def _is_over(self):
    global game_over
    return game_over


  ## Actor Manipulation Methods
  def _move_forward(self,id=0):
    ## This moves the camera forward by a fixed step size
    step_size = 50

    cur_pos = self._get_camera_pos(id)
    cur_rot = self._get_camera_rotation(id)
    R = self._toRotationMatrix(cur_rot[0],cur_rot[1],cur_rot[2])

    noise = np.random.rand() ## random noise  0~1
    new_pos = np.dot(R,np.transpose(np.array((step_size+noise,0,0)))) + np.array(cur_pos)
    #res = client.request('vset /camera/%d/location %f %f %f'%(id,new_pos[0],new_pos[1],new_pos[2]))
    res = client.request('vset /camera/%d/moveto %f %f %f'%(id,new_pos[0],new_pos[1],new_pos[2]))
    return True

  def _move_backward(self,id=0):
    ## This moves the camera forward by a fixed step size
    step_size = -50

    cur_pos = self._get_camera_pos(id)
    cur_rot = self._get_camera_rotation(id)
    R = self._toRotationMatrix(cur_rot[0],cur_rot[1],cur_rot[2])

    noise = np.random.rand() ## random noise  0~1
    new_pos = np.dot(R,np.transpose(np.array((step_size+noise,0,0)))) + np.array(cur_pos)
    res = client.request('vset /camera/%d/location %f %f %f'%(id,new_pos[0],new_pos[1],new_pos[2]))
    return True

  def _rotate_right(self,id=0):
    step_size = 45
    cur_rot = self._get_camera_rotation(id)

    noise = np.random.rand()*2 - 1 ## random noise  -1~1 degrees
    new_rot = (cur_rot[0],cur_rot[1]+step_size+noise,cur_rot[2])
    res = client.request('vset /camera/%d/rotation %f %f %f'%(id,new_rot[0],new_rot[1],new_rot[2]))

  def _rotate_left(self,id=0):
    step_size = -45
    cur_rot = self._get_camera_rotation(id)
    noise = np.random.rand()*2 - 1 ## random noise  -1~1 degrees
    new_rot = (cur_rot[0],cur_rot[1]+step_size+noise,cur_rot[2])
    res = client.request('vset /camera/%d/rotation %f %f %f'%(id,new_rot[0],new_rot[1],new_rot[2]))

  ## Camera Accessor Methods
  def _get_camera_pos(self,id=0):
    res = client.request('vget /camera/%d/location'%id)
    x,y,z = res.split(' ')
    return float(x), float(y), float(z)

  def _get_camera_rotation(self,id=0):
    res = client.request('vget /camera/%d/rotation'%id)
    pitch,yaw,roll = res.split(' ')
    return float(pitch), float(yaw), float(roll)


  def _toRotationMatrix(self,pitch,yaw,roll):
    p = radians(pitch)
    y = radians(yaw)
    r = radians(roll)
    Rz = np.array([[cos(y),-sin(y),0],[sin(y),cos(y),0],[0,0,1]])
    Ry = np.array([[cos(p),0,sin(p)],[0,1,0],[-sin(p),0,cos(p)]])
    Rx = np.array([[1,0,0],[0,cos(r),-sin(r)],[0,sin(r),cos(r)]])
    RzRy = np.dot(Rz,Ry)
    RzRyRx = np.dot(RzRy,Rx)
    return RzRyRx




