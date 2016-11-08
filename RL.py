import pdb
from World import *
from Models import *
import numpy as np 

def train(model,env,exp_replay,epoch,action_space,epsilon,batch_size):
  model_path_out_path = '/home/tk/dev/unrealcv/src/model_weights/'
  for ep in epoch:
    loss = 0.0
    evn.reset()
    objective_complete = False
    win_count = 0
    
    frame = env.observe() 
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

      if objective_complete:
        # book keeping
        win_count += 1

      # store experience
      exp_replay.remember(frame_prev,action,reward,frame,objective_complete)

      # learn
      inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
      loss += model.train_on_batch(inputs,targets)[0]

    print "Epoch {%03d}/{%d} | Loss {%.6f} | Win count {}".format(ep, loss, win_count)


if __name__ == "__main__":
  ## TRAINING PARAMETERS
  EPOCH = 150
  ACTION_SPACE = 4 # go_front, go_back, turn_left, turn_right
  EPSILON = 0.1 # Exploration
  BATCH_SIZE = 128

  # get model
  env = World()
  exp_replay = ExperienceReplay(max_memory=max_memory)
  #model = UENet()

  train(model=model,
        env=env,
        exp_replay=exp_replay,
        epoch=EPOCH,
        action_space=ACTION_SPACE,
        epsilon=EPSILON,
        batch_size=BATCH_SIZE)


  
