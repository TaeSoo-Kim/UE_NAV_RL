import sys
from unrealcv import client
import numpy as np
import pdb

## World object. This represents the state of the UnrealEngine game environment.
class ExperienceReplay(object):
  def __init__(self,max_memory=100,discount=0.9):
    self.max_memory = max_memory
    self.memory = list()
    self.discount = discount

  def remember(self, state, game_over):
    # memory is a list where each element is: [state, game_over]
    # state is: [state_t, action_t, reward_t, state_t+1]
    self.memory.append([state,game_over])
    if len(self.memory) > self.max_memory:
      del self.memory[0] # forget the old memories first

  def get_batch(self, model, batch_size = 10):
    len_memory = len(self.memory)
    num_actions = model.output_shape[-1]
    env_dim = self.memory[0][0][0].shape
    #pdb.set_trace()
    inputs = np.zeros((min(len_memory,batch_size),env_dim[1],env_dim[2],env_dim[3]))
    targets = np.zeros((inputs.shape[0], num_actions))

    for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
      state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
      game_over = self.memory[idx][1]

      inputs[i:i+1] = state_t
      # There should be no target values for actions not taken.
      # Thou shalt not correct actions not taken #deep

      targets[i] = model.predict(state_t)[0]
      Q_sa = np.max(model.predict(state_tp1)[0])
      if game_over:
        targets[i, action_t] = reward_t
      else:
        targets[i,action_t] = reward_t + self.discount*Q_sa

    return inputs, targets