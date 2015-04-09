from pylearn2.models.mlp import Linear, Layer
from pylearn2.utils import wraps
from pylearn2.expr.basic import log_sum_exp
import theano
import theano.tensor as T
import numpy as np
from pylearn2.space import VectorSpace
from theano.compat.python2x import OrderedDict

import sys
import warnings

from pylearn2.utils import sharedX
from pylearn2.space import Space, CompositeSpace
from pylearn2.datasets.preprocessing import Preprocessor

class CombinedLogLikelihood(Linear):
    def __init__(self, components, **kwargs):
        self.component_dists = [(s[0].strip(), int(s[2])) for s in [xx.partition(":") for xx in components]]
        super(CombinedLogLikelihood, self).__init__(dim=(sum([d[1] for d in self.component_dists]) * 2), **kwargs)
        self.__dict__.update(locals())
        del self.self
        del self.kwargs

    def logprob_lognormal(self, y_target, mean, sigma):
        return (((T.log(y_target) - mean) ** 2 / (2 * sigma ** 2) + T.log(y_target * sigma * T.sqrt(2 * np.pi)))).sum(axis=1)

    def logprob_normal(self, y_target, mean, sigma):
        sigma_square = sigma ** 2
        return (0.5 * (T.log(2 * np.pi * sigma_square) + 0.5 * ((y_target - mean) ** 2) / sigma_square)).sum(axis=1)

    def logprob(self, y_targets, means, sigmas):
        sum_logprob = T.zeros_like(y_targets[:, 0])
        i = 0
        for (dist_name, dimension) in self.component_dists:
            y_target = y_targets[:, i:(i + dimension)]
            mean = means[:, i:(i + dimension)]
            sigma = sigmas[:, i:(i + dimension)]
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
    def __init__(self, **kwargs):
        super(LogNormalLogLikelihood, self).__init__(dim=2, **kwargs)
        self.__dict__.update(locals())
        del self.self
        del self.kwargs
        #self.output_space = VectorSpace(2)

    def logprob(self, y_target, mean, sigma):
        return (((T.log(y_target) - mean) ** 2 / (2 * sigma ** 2) + T.log(y_target * sigma * T.sqrt(2 * np.pi))))

    def cost(self, Y, Y_hat):
        #sigma = 0.26165911509618789
        #mean = 1.6091597151048114
        mean = Y_hat[:, 0] #+ 1.6091597151048114
        sigma = T.exp(Y_hat[:, 1]) #+ 0.26165911509618789
        y_target = Y[:, 0]
        #return (-T.log((T.exp(-(T.log(y_target) - mean)**2 / (2 * sigma**2)) / (y_target * sigma * T.sqrt(2 * np.pi))))).mean()
        #return ((((T.log(y_target) - mean)**2 / (2 * sigma**2) + T.log(y_target * sigma * T.sqrt(2 * np.pi))))).mean()
        return self.logprob(y_target, mean, sigma).mean()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        rval = super(LogNormalLogLikelihood, self).get_layer_monitoring_channels(state_below, state, targets)
        if targets:
            y_target = targets[:, 0]
            mean = state[:, 0]
            sigma = T.exp(state[:, 1])
            nll = self.logprob(y_target, mean, sigma)
            prob_vector = T.exp(-nll)
            rval['prob'] = prob_vector.mean()
            rval['ppl'] = T.exp(nll.mean())
        return rval


class LogNormalMixtureLogLikelihood(Linear):
    def __init__(self, num_mixtures = 2, **kwargs):
        super(LogNormalMixtureLogLikelihood, self).__init__(dim=3*num_mixtures, **kwargs)
        self.__dict__.update(locals())
        del self.self
        del self.kwargs
        #self.output_space = VectorSpace(self.num_mixtures)

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        super(LogNormalMixtureLogLikelihood, self).set_input_space(space)
        self.output_space = VectorSpace(1)

    def logprob(self, y_target, pdf_params):
        weights = T.nnet.softmax(pdf_params[:, 0::3])
        means = pdf_params[:, 1::3]
        sigmas = T.exp(pdf_params[:, 2::3])
        return self.calculate_logprob(y_target, weights, means, sigmas)

    def calculate_logprob(self, y_target, weights, means, sigmas):
        return log_sum_exp([T.log(weights[:, i]) + ((T.log(y_target[:, 0]) - means[:, i]) ** 2 / (2 * sigmas[:, i] ** 2) + T.log(y_target[:, 0] * sigmas[:, i] * T.sqrt(2 * np.pi))) for i in range(self.num_mixtures)], axis=0)

    def cost(self, Y, Y_hat):
        return self.logprob(Y, Y_hat).mean()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        rval = super(LogNormalMixtureLogLikelihood, self).get_layer_monitoring_channels(state_below, state, targets)
        if targets:
            nll = self.logprob(targets, state)
            prob_vector = T.exp(-nll)
            rval['prob'] = prob_vector.mean()
            rval['ppl'] = T.exp(nll.mean())
        return rval


class NegativeBinomialLogLikelihood(Linear):
    """
    Negative binomial distribution can be regarded as a discrete equivalent of the Gamma distribution
    """

    def __init__(self, **kwargs):
        super(NegativeBinomialLogLikelihood, self).__init__(dim=2, **kwargs)
        self.__dict__.update(locals())
        del self.self
        del self.kwargs
        #self.output_space = VectorSpace(2)

    def logprob(self, y_target, n, p):
        coeff = T.gammaln(n + y_target) - T.gammaln(y_target + 1) - T.gammaln(n)
        return - (coeff + n * T.log(p) + y_target * T.log(1-p))


    def cost(self, Y, Y_hat):
        n = T.exp(Y_hat[:, 0]) # n > 0
        p = T.nnet.sigmoid(Y_hat[:, 1]) # p in (0,1)
        y_target = Y[:, 0]
        return self.logprob(y_target, n, p).mean()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        rval = super(NegativeBinomialLogLikelihood, self).get_layer_monitoring_channels(state_below, state, targets)
        if targets:
            y_target = targets[:, 0]
            n = T.exp(state[:, 0]) # n > 0
            p = T.nnet.sigmoid(state[:, 1]) # p in (0,1)
            prob_vector = T.exp(- self.logprob(y_target, n, p))
            rval['prob'] = prob_vector.mean()
            rval['ppl'] = T.exp(-T.log(prob_vector).mean())
        return rval




class LinearDist(Linear):
    def __init__(self, **kwargs):
        super(LinearDist, self).__init__(dim=2, **kwargs)
        self.__dict__.update(locals())
        del self.self
        del self.kwargs
        #self.output_space = VectorSpace(2)


    def cost(self, Y, Y_hat):
        #sigma = 0.26165911509618789
        #mean = 1.6091597151048114
        mean = Y_hat[:, 0]
        sigma = T.exp(Y_hat[:, 1])

        y_target = Y[:, 0]
        return (-T.log(T.exp(-((y_target - mean) ** 2) / ( 2 * sigma ** 2)) / (sigma * T.sqrt(2 * np.pi)))).mean()


    def get_monitoring_channels_from_state(self, state, target=None):
        rval = super(Linear, self).get_monitoring_channels()
        if target:
            y_target = target[:, 0]
            mean = state[:, 0]
            sigma = T.exp(state[:, 1])
            prob_vector = T.exp(-((y_target - mean) ** 2) / ( 2 * sigma ** 2)) / (sigma * T.sqrt(2 * np.pi))
            rval['prob'] = prob_vector.mean()
            rval['ppl'] = T.exp(-T.log(prob_vector).mean())
        return rval




   
    
class ConnectionScaler(Layer):
  def __init__(self, layer_name, use_bias=True, init_bias=1.0):
    super(ConnectionScaler, self).__init__()
    
    self.__dict__.update(locals())
    del self.self
      
  @wraps(Layer.set_input_space)
  def set_input_space(self, space):
    self.input_space = space

    assert isinstance(space, CompositeSpace)
   
    assert isinstance(space.components[0], VectorSpace)
    assert isinstance(space.components[1], VectorSpace)
    if space.components[0].dim != space.components[1].dim:
      raise ValueError("Input space dimensions differ: %d != %d" % (space.components[0].dim, space.components[1].dim))
    self.dim = space.components[0].dim
    self.output_space = VectorSpace(self.dim)
    if self.use_bias:
      self.b = sharedX(np.zeros((self.dim,)) + self.init_bias, name=(self.layer_name + '_b'))
    self._params = [self.b]
    	
  @wraps(Layer.get_lr_scalers)
  def get_lr_scalers(self):

    rval = OrderedDict()

    if not hasattr(self, 'b_lr_scale'):
        self.b_lr_scale = None

    if self.b_lr_scale is not None:
        assert isinstance(self.b_lr_scale, float)
        rval[self.b] = self.b_lr_scale

    return rval    
    
  @wraps(Layer.fprop)
  def fprop(self, state_below):
    if self.use_bias:
      return state_below[0] * (state_below[1] + self.b)
    else:
      return state_below[0] * state_below[1]
      
  @wraps(Layer.get_params)
  def get_params(self):
    return [self.b]


  @wraps(Layer.get_layer_monitoring_channels)
  def get_layer_monitoring_channels(self, state_below=None, state=None,
                                    target=None):
    b = self.b
    rval = OrderedDict([('bias_min', b.min()),
                        ('bias_mean', b.mean()),
                        ('bias_max', b.max()),])
    return rval

  @wraps(Layer.set_biases)
  def set_biases(self, biases):
    self.b.set_value(biases)

  @wraps(Layer.get_weight_decay)
  def get_weight_decay(self, coeff):
    return 0
