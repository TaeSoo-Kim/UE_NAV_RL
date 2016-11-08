from unrealcv import client
import numpy as np 
import matplotlib.pyplot as plt 
import os,sys,pdb
from math import cos,sin,radians

world_objects = {}

def navigate():
  for i in range(100):
    move_forward()

## Camera Accessor Methods
def get_camera_pos(id=0):
  res = client.request('vget /camera/%d/location'%id)
  x,y,z = res.split(' ')
  return float(x), float(y), float(z)

def get_camera_rotation(id=0):
  res = client.request('vget /camera/%d/rotation'%id)
  pitch,yaw,roll = res.split(' ')
  return float(pitch), float(yaw), float(roll)

## World Accessor Methods
def get_all_object_names():
  res = client.request('vget /objects')
  obj_names = res.split(' ')
  return obj_names


## Camera Manipulation Methods
def move_forward(id=0):
  ## This moves the camera forward by a fixed step size
  step_size = 10

  cur_pos = get_camera_pos(id)
  cur_rot = get_camera_rotation(id)
  R = toRotationMatrix(cur_rot[0],cur_rot[1],cur_rot[2])
  
  
  res = client.request('vset /camera/%d/location %f %f %f'%(id,new_pos[0],new_pos[1],new_pos[2]))
  return True

def rotate_right(id=0):
  step_size = 10
  cur_rot = get_camera_rotation(id)
  new_rot = (cur_rot[0],cur_rot[1]+step_size,cur_rot[2])
  res = client.request('vset /camera/%d/rotation %f %f %f'%(id,new_rot[0],new_rot[1],new_rot[2]))

def rotate_left(id=0):
  step_size = 10
  cur_rot = get_camera_rotation(id)
  new_rot = (cur_rot[0],cur_rot[1]-step_size,cur_rot[2])
  res = client.request('vset /camera/%d/rotation %f %f %f'%(id,new_rot[0],new_rot[1],new_rot[2]))

## Utility Functions
def toRotationMatrix(pitch,yaw,roll):
  p = radians(pitch)
  y = radians(yaw)
  r = radians(roll)
  Rz = np.array([[cos(y),-sin(y),0],[sin(y),cos(y),0],[0,0,1]])
  Ry = np.array([[cos(p),0,sin(p)],[0,1,0],[-sin(p),0,cos(p)]])
  Rx = np.array([[1,0,0],[0,cos(r),-sin(r)],[0,sin(r),cos(r)]])
  RzRy = np.dot(Rz,Ry)
  RzRyRx = np.dot(RzRy,Rx)
  return RzRyRx

def set_up_world():
  global world_objects
  obj_names = get_all_object_names()
  for obj in obj_names:
    loc = client.request('vget /object/%s/location'%obj).split()
    world_objects[obj] = (float(loc[0]),float(loc[1]),float(loc[2]))
  pdb.set_trace()
  
def main():
  ## Connect to game
  client.connect()
  if not client.isconnected():
    print 'UnrealCV server is not running. Gaming running?'
    sys.exit()
  
  ## System status
  res = client.request('vget /unrealcv/status')
  print res
  ## Quick vget /objects bug fix
  client.request('vset /viewmode object_mask')
  client.request('vset /viewmode lit')

  #print 'Setting up the world'
  #set_up_world() ## this sets up a dictionary of  object to location pairs
  #print 'Done'

  pdb.set_trace()
  move_forward()
  


if __name__ == "__main__":
  main()
