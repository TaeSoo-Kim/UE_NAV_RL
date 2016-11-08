import sys
from unrealcv import client

## World object. This represents the state of the UnrealEngine game environment.
class ExperienceReplay(object):
  def __init__(self):
    client.connect()
    if not client.isconnected():
      print 'UnrealCV server is not running. Gaming running?'
      sys.exit()
    ## System Status
    res = client.request('vget /unrealcv/status')
    print res
  
    pass

  def observe(self):
    pass

  def act(self,action):
    pass

  def reset(self):
    pass