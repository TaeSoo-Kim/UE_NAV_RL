import keras
from rl.core import Env, Processor
from unrealcv import client
import numpy as np
import cv2
from math import cos,sin,radians
import pdb
import matplotlib.pyplot as plt

## globals
REWARD = -0.01 # default to -0.01 because every step counts as penalty:-0.01
default_step_penalty = -0.01
game_over = 0
INPUT_SHAPE = (224,224)

def reward_listener(message):
  global REWARD, default_step_penalty
  # message should be : "Reward TYPE"
  print 'Got server message %s' % repr(message)
  
  
  if "COLLISION" in message:
    REWARD = -1.0
  elif "INSIGHT" in message:
    REWARD = 1.0
  elif "GAME_OVER" in message:
    game_over = 1
    REWARD = 10.0
  else: ## every step is -0.1,  try to find the shortest path
    REWARD = default_step_penalty

class UEEnv(Env):
  """The abstract environment class that is used by all agents. This class has the exact
  same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
  OpenAI Gym implementation, this class only defines the abstract methods without any actual
  implementation.
  """
  

  def __init__(self):
    global REWARD, game_over
    REWARD = 0
    game_over = 0

    self.reward_range = (-np.inf, np.inf)
    self.action_space = 4
    self.observation_space = None
    
    client.connect()
    client.message_handler = reward_listener
    if not client.isconnected():
      print 'UnrealCV server is not running. Gaming running?'
      sys.exit()
    ## System Status
    res = client.request('vget /unrealcv/status')
    print "Connection?? :",res

  def step(self, action):
    """Run one timestep of the environment's dynamics.
    Accepts an action and returns a tuple (observation, reward, done, info).
    Args:
        action (object): an action provided by the environment
    Returns:
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    """
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
    
    info = {}
    return self.observe(), self._get_reward(), self._is_over(), info

  def reset(self):
    """
    Resets the state of the environment and returns an initial observation.
    Returns:
        observation (object): the initial observation of the space. (Initial reward is assumed to be 0.)
    """
    return self.observe()

  def render(self, mode='human', close=False):
    """Renders the environment.
    The set of supported modes varies per environment. (And some
    environments do not support rendering at all.) By convention,
    if mode is:
    - human: render to the current display or terminal and
      return nothing. Usually for human consumption.
    - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
      representing RGB values for an x-by-y pixel image, suitable
      for turning into a video.
    - ansi: Return a string (str) or StringIO.StringIO containing a
      terminal-style text representation. The text can include newlines
      and ANSI escape sequences (e.g. for colors).
    Note:
        Make sure that your class's metadata 'render.modes' key includes
          the list of supported modes. It's recommended to call super()
          in implementations to use the functionality of this method.
    Args:
        mode (str): the mode to render with
        close (bool): close all open renderings
    """
    raise NotImplementedError()

  def close(self):
    """Override in your subclass to perform any necessary cleanup.
    Environments will automatically close() themselves when
    garbage collected or when the program exits.
    """
    raise NotImplementedError()

  def seed(self, seed=None):
    """Sets the seed for this env's random number generator(s).
    Note:
        Some environments use multiple pseudorandom number generators.
        We want to capture all such seeds used in order to ensure that
        there aren't accidental correlations between multiple generators.
    Returns:
        list<bigint>: Returns the list of seeds used in this env's random
          number generators. The first value in the list should be the
          "main" seed, or the value which a reproducer should pass to
          'seed'. Often, the main seed equals the provided 'seed', but
          this won't be true if seed=None, for example.
    """
    np.random.seed(seed)
    return [seed]

  def configure(self, *args, **kwargs):
    """Provides runtime configuration to the environment.
    This configuration should consist of data that tells your
    environment how to run (such as an address of a remote server,
    or path to your ImageNet data). It should not affect the
    semantics of the environment.
    """
    raise NotImplementedError()


  ########################## ABOVE THIS LINE: ALL keras-rl Env() methods

  def observe(self):
    ## get current frame
    #UE4_root = '/home/tk/dev/ThirdParty/UE4.13/UnrealEngine/'
    img_out_path = '/home/tk/dev/unrealcv/src/temp_imgs/img.png'
    res = client.request('vget /camera/0/lit ' + img_out_path )
    return cv2.imread(img_out_path)
    
    

  def _get_reward(self):
    global REWARD, default_step_penalty
    ###################
    # copy reward, set it back to -0.1, return the copy

    return_reward_val = REWARD
    REWARD = default_step_penalty
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


class UEProcessor(Processor):
  def process_observation(self, observation):
    global INPUT_SHAPE
    assert observation.ndim == 3  # (height, width, rgb)
    observation = cv2.resize(observation,INPUT_SHAPE)  # resize 
    return observation.astype('uint8')  # saves storage in experience memory

#    return img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))

  def process_state_batch(self, batch):
      # We could perform this processing step in `process_observation`. In this case, however,
      # we would need to store a `float32` array instead, which is 4x more memory intensive than
      # an `uint8` array. This matters if we store 1M observations.
      batch = batch.reshape((batch.shape[1],batch.shape[2],batch.shape[3],batch.shape[4]))
      processed_batch = batch.astype('float32') / 255.
      return processed_batch

if __name__ == "__main__":
    env = UEEnv()
    pdb.set_trace()