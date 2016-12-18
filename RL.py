import pdb
from UEEnv import *
from Models import *
import numpy as np 

from keras.optimizers import Adam
from keras.optimizers import sgd

#from keras.callbacks import ModelCheckpoint

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

"""
def train(model,env,exp_replay,epoch,action_space,epsilon,batch_size):
  model_path_out_path = '/home/tk/dev/unrealcv/src/model_weights/'
  MAX_STEP_COUNT = 1000
  for ep in range(0,epoch):
    loss = 0.0
    env.reset()
    objective_complete = False
    win_count = 0
    
    frame = env.observe() 
    epoch_reward = 0.0
    current_best_reward = -30000
    step_count = 0
    while not objective_complete:
      # get state observation
      frame_prev = frame

      
      if np.random.rand() <= epsilon:
        # explore
        action = np.random.randint(0,action_space)
        print 'explore'
      else:
        # sample action
        q = model.predict(frame_prev)
        action = np.argmax(q[0])

      #action = 0

      # Act
      frame, reward, objective_complete = env.act(action)
      epoch_reward += reward
      print "step: ",step_count, ", action: ", action, ", current reward:", epoch_reward
      print 'q distribution: ',q

      if objective_complete:
        # book keeping
        if current_best_reward < epoch_reward:
          current_best_reward = epoch_reward
          model.save_weights('model_weights/ResNet_RL_epoch_%d.hdf5'%ep, overwrite=False)
        epoch_reward = 0 
        win_count += 1

      # store experience
      exp_replay.remember([frame_prev,action,reward,frame],objective_complete)

      # learn
      inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
      loss += model.train_on_batch(inputs,targets)[0]

      step_count += 1
      if step_count >= MAX_STEP_COUNT:
        model.save_weights('model_weights/ResNet_RL_epoch_%d.hdf5'%ep, overwrite=False)
        break
      if objective_complete:
        break

    print "Epoch {%d}/{%d} | Loss {%.6f} | Win count {%d}"%(ep,epoch, loss, win_count)
"""

if __name__ == "__main__":
  ## TRAINING PARAMETERS
  num_steps = 1000000
  max_memory = 1000
  eps_decay_schedule_max_step = 100000

  INPUT_SHAPE = (224, 224, 3)
  nb_actions = 4
  WINDOW_LENGTH = 1

  model = UE_Net(nb_actions)
  print 'Created UENET'
  memory = SequentialMemory(limit=max_memory, window_length=WINDOW_LENGTH)
  print 'Created SequentialMemory'
  policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=.1, value_test=.05,nb_steps=100000)
  print "Created Policy"
  processor = UEProcessor()
  print "Created processor"
  env = UEEnv()
  print 'Created Environment'
  dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=1000, gamma=.99, delta_range=(-1., 1.),
               target_model_update=10000, train_interval=4)
  dqn.compile(Adam(lr=.00025), metrics=['mae'])
  print 'Created agent'
  
  # Okay, now it's time to learn something! We capture the interrupt exception so that training
  # can be prematurely aborted. Notice that you can the use built-in Keras callbacks!
  weights_filename = 'model_weights/dqn_{}_weights.h5f'.format('UE')
  checkpoint_weights_filename = 'model_weights/dqn_' + 'UE' + '_weights_{step}.h5f'
  log_filename = 'model_weights/dqn_{}_log.json'.format('UE')
  callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000)]
  callbacks += [FileLogger(log_filename, interval=1000)]
  dqn.fit(env, callbacks=callbacks, nb_steps=num_steps, log_interval=1000)

  # After training is done, we save the final weights one more time.
  dqn.save_weights(weights_filename, overwrite=True)