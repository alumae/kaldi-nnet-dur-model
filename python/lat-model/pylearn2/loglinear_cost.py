from pylearn2.models.mlp import Linear, Layer
from pylearn2.utils import wraps
import theano.tensor as T
import numpy as np
from pylearn2.space import VectorSpace
from theano.compat.python2x import OrderedDict

class CombinedLogLikelihood(Linear):

  def __init__(self, components, **kwargs):
        self.component_dists = [(s[0].strip(), int(s[2])) for s in [xx.partition(":") for xx in components]]
        super(CombinedLogLikelihood, self).__init__(dim=(sum([d[1] for d in self.component_dists]) * 2), **kwargs)
        self.__dict__.update(locals())
        del self.self
        del self.kwargs


  def logprob_lognormal(self, y_target, mean, sigma):
    return (((T.log(y_target) - mean)**2 / (2 * sigma**2) + T.log(y_target * sigma * T.sqrt(2 * np.pi)))).sum(axis=1)

  def logprob_normal(self, y_target, mean, sigma):
    sigma_square = sigma ** 2
    return (0.5 * (T.log(2 * np.pi * sigma_square) + 0.5 * ((y_target - mean) ** 2) / sigma_square)).sum(axis=1)
  
  
  def logprob(self, y_targets, means, sigmas):
    sum_logprob = T.zeros_like(y_targets[:,0])
    i = 0
    for (dist_name, dimension) in self.component_dists:
      y_target = y_targets[:, i:(i+dimension)]
      mean = means[:, i:(i+dimension)]
      sigma = sigmas[:, i:(i+dimension)]
      if dist_name == 'lognormal':
        sum_logprob += self.logprob_lognormal(y_target, mean, sigma)
      elif dist_name == 'normal':
        sum_logprob += self.logprob_normal(y_target, mean, sigma)
      i += dimension
    return sum_logprob
    
  
  def cost(self, Y, Y_hat):    
    y_targets = Y[:, 0::2]
    means = Y_hat[:, 0::2]
    sigmas = T.exp(Y_hat[:, 1::2])
    return self.logprob(y_targets, means, sigmas).mean()
      
  def get_monitoring_channels_from_state(self, state, target=None):
    rval = super(Linear, self).get_monitoring_channels()
    if target:
      pass
    return rval
  

class LogNormalLogLikelihood(Linear):
  
  def __init__(self, ** kwargs):
        super(LogNormalLogLikelihood, self).__init__(dim=2, **kwargs)
        self.__dict__.update(locals())
        del self.self
        del self.kwargs
        #self.output_space = VectorSpace(2)

  
  def logprob(self, y_target, mean, sigma):
    return (((T.log(y_target) - mean)**2 / (2 * sigma**2) + T.log(y_target * sigma * T.sqrt(2 * np.pi))))
  
  def cost(self, Y, Y_hat):    
    #sigma = 0.26165911509618789
    #mean = 1.6091597151048114
    mean = Y_hat[:, 0] #+ 1.6091597151048114
    sigma = T.exp(Y_hat[:, 1]) #+ 0.26165911509618789
    y_target =  Y[:, 0]
    #return (-T.log((T.exp(-(T.log(y_target) - mean)**2 / (2 * sigma**2)) / (y_target * sigma * T.sqrt(2 * np.pi))))).mean()
    #return ((((T.log(y_target) - mean)**2 / (2 * sigma**2) + T.log(y_target * sigma * T.sqrt(2 * np.pi))))).mean()
    return self.logprob(y_target, mean, sigma).mean()
    
  @wraps(Layer.get_layer_monitoring_channels)
  def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):

    rval = super(LogNormalLogLikelihood, self).get_layer_monitoring_channels(state_below, state, targets)
    if targets:
        y_target =  targets[:, 0]
        mean =  state[:, 0]
        sigma = T.exp(state[:, 1])
        prob_vector = T.exp(- self.logprob(y_target, mean, sigma))
        rval['prob'] = prob_vector.mean()
        rval['ppl'] = T.exp(-T.log(prob_vector).mean())
    return rval

  
  #def get_monitoring_channels_from_state(self, state, target=None):
    #rval =  OrderedDict([])
    #return rval

  #def set_input_space(self, space):
    #super(LogLinearDist, self).set_input_space(space)
    #self.output_space = VectorSpace(1 + self.copy_input * self.input_dim)


class LinearDist(Linear):
  
  def __init__(self, ** kwargs):
        super(LinearDist, self).__init__(dim=2, **kwargs)
        self.__dict__.update(locals())
        del self.self
        del self.kwargs
        #self.output_space = VectorSpace(2)

  
  def cost(self, Y, Y_hat):    
    #sigma = 0.26165911509618789
    #mean = 1.6091597151048114
    mean =  Y_hat[:, 0]
    sigma = T.exp(Y_hat[:, 1])
    
    
    y_target =  Y[:, 0]
    return (-T.log(T.exp(-((y_target - mean) ** 2)/ ( 2 * sigma **2)) / (sigma * T.sqrt(2 * np.pi)))).mean()


  def get_monitoring_channels_from_state(self, state, target=None):
    rval = super(Linear, self).get_monitoring_channels()
    if target:
        y_target =  target[:, 0]
        mean =  state[:, 0]
        sigma = T.exp(state[:, 1])
        prob_vector = T.exp(-((y_target - mean) ** 2)/ ( 2 * sigma **2)) / (sigma * T.sqrt(2 * np.pi))
        rval['prob'] = prob_vector.mean()
        rval['ppl'] = T.exp(-T.log(prob_vector).mean())
    return rval
