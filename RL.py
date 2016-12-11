import pdb
from World import *
from ExperienceReplay import *
#from Models import *
import numpy as np 

from keras.optimizers import Adam
from keras.optimizers import sgd
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
#from rl.agents.dqn import DQNAgent
#from rl.policy import BoltzmannQPolicy
#from rl.memory import SequentialMemory


def train(model,env,exp_replay,epoch,action_space,epsilon,batch_size):
  model_path_out_path = '/home/tk/dev/unrealcv/src/model_weights/'
  for ep in range(0,epoch):
    loss = 0.0
    env.reset()
    objective_complete = False
    win_count = 0
    
    frame = env.observe() 
    epoch_reward = 0.0
    current_best_reward = -30000
    while not objective_complete:
      # get state observation
      frame_prev = frame

      if np.random.rand() <= epsilon:
        # explore
        action = np.random.randint(0,action_space)
      else:
        # sample action
        q = model.predict(frame_prev)
        action = np.argmax(q[0])

      # Act
      frame, reward, objective_complete = env.act(action)
      epoch_reward += reward
      print "Current reward:", epoch_reward

      if objective_complete:
        # book keeping
        if current_best_reward < epoch_reward:
          current_best_reward = epoch_reward
          model.save_weights('model_weights/ResNet50_RL_epoch_%d.hdf5'%ep, overwrite=False)
        epoch_reward = 0 
        win_count += 1

      # store experience
      exp_replay.remember([frame_prev,action,reward,frame],objective_complete)

      # learn
      inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
      loss += model.train_on_batch(inputs,targets)[0]

    print "Epoch {%03d}/{%d} | Loss {%.6f} | Win count {}".format(ep, loss, win_count)


if __name__ == "__main__":
  ## TRAINING PARAMETERS
  EPOCH = 150
  ACTION_SPACE = 4 # 0:go_front, 1:go_back, 2:turn_left, 3:turn_right
  EPSILON = 0.1 # Exploration, 1.0 = always random
  BATCH_SIZE = 128
  max_memory = 100

  OUTPUT_DIM = 4
  INPUT_SHAPE = (240, 320, 3)
  
  
  # get model
  #model = ResNetBuilder.build_resnet_50(INPUT_SHAPE, OUTPUT_DIM)
  
  env = World()
  #exp_replay = None
  exp_replay = ExperienceReplay(max_memory=max_memory)
 
  #memory = SequentialMemory(limit=50000, window_length=1)
  #policy = BoltzmannQPolicy()
  #dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               #target_model_update=1e-2, policy=policy)
  #model.compile(Adam(lr=1e-3), metrics=['mae'])
  model = ResNet50(include_top=True, weights='imagenet')
  model.compile(Adam(lr=1e-3), 'mae', metrics=['mae'])
#  pdb.set_trace()


  train(model=model,
        env=env,
        exp_replay=exp_replay,
        epoch=EPOCH,
        action_space=ACTION_SPACE,
        epsilon=EPSILON,
        batch_size=BATCH_SIZE)
  #model = UENet()

  """
  train(model=model,
        env=env,
        exp_replay=exp_replay,
        epoch=EPOCH,
        action_space=ACTION_SPACE,
        epsilon=EPSILON,
        batch_size=BATCH_SIZE)

  """
  
