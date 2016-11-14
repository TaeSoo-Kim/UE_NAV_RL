import sys
from unrealcv import client
import matplotlib.pyplot as plt
import pdb
import cv2
from math import cos,sin,radians
import numpy as np

## World object. This represents the state of the UnrealEngine game environment.
class World(object):
  def __init__(self):
    client.connect()
    if not client.isconnected():
      print 'UnrealCV server is not running. Gaming running?'
      sys.exit()
    ## System Status
    res = client.request('vget /unrealcv/status')
    print res
  

  def observe(self):
    ## get current frame
    img_out_path = '/home/tk/dev/unrealcv/src/temp_imgs/img.png'
    res = client.request('vget /camera/0/lit ' + img_out_path )
    img = cv2.imread(img_out_path)
    return img

  def act(self,action):
    ## TODO: Write routine to move the actor
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
    ## TODO: Write routine to restart the game

    ## The game can reset itself once the trigger is set off.
    pass

  def _get_reward(self):
    ## TODO: design reward
    # +10: task_complete, -0.1: every step, -1: collision
    if self._is_over():
      return 10
    elif self._in_collision():
      return -1
    else:
      return -0.1

    
    return 0

  def _is_over(self):
    ## TODO: implement game over logic
    ## If in view and close enough

    ## Maybe listen to an event that is triggered by the trigger box in game.
    return False

  def _in_collision(self):
    ## TODO: Listen to collision nevents
    pass

  ## Actor Manipulation Methods
  def _move_forward(self,id=0):
    ## This moves the camera forward by a fixed step size
    step_size = 50

    cur_pos = self._get_camera_pos(id)
    cur_rot = self._get_camera_rotation(id)
    R = self._toRotationMatrix(cur_rot[0],cur_rot[1],cur_rot[2])
    new_pos = np.dot(R,np.transpose(np.array((step_size,0,0)))) + np.array(cur_pos)
    res = client.request('vset /camera/%d/location %f %f %f'%(id,new_pos[0],new_pos[1],new_pos[2]))
    return True

  def _move_backward(self,id=0):
    ## This moves the camera forward by a fixed step size
    step_size = -50

    cur_pos = self._get_camera_pos(id)
    cur_rot = self._get_camera_rotation(id)
    R = self._toRotationMatrix(cur_rot[0],cur_rot[1],cur_rot[2])
    new_pos = np.dot(R,np.transpose(np.array((step_size,0,0)))) + np.array(cur_pos)
    res = client.request('vset /camera/%d/location %f %f %f'%(id,new_pos[0],new_pos[1],new_pos[2]))
    return True

  ## Camera Accessor Methods
  def _get_camera_pos(self,id=0):
    res = client.request('vget /camera/%d/location'%id)
    x,y,z = res.split(' ')
    return float(x), float(y), float(z)

  def _get_camera_rotation(self,id=0):
    res = client.request('vget /camera/%d/rotation'%id)
    pitch,yaw,roll = res.split(' ')
    return float(pitch), float(yaw), float(roll)

  def _rotate_right(self,id=0):
    step_size = 10
    cur_rot = self._get_camera_rotation(id)
    new_rot = (cur_rot[0],cur_rot[1]+step_size,cur_rot[2])
    res = client.request('vset /camera/%d/rotation %f %f %f'%(id,new_rot[0],new_rot[1],new_rot[2]))

  def _rotate_left(self,id=0):
    step_size = -10
    cur_rot = self._get_camera_rotation(id)
    new_rot = (cur_rot[0],cur_rot[1]+step_size,cur_rot[2])
    res = client.request('vset /camera/%d/rotation %f %f %f'%(id,new_rot[0],new_rot[1],new_rot[2]))

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