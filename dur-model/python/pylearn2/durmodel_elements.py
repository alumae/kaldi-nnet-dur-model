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

from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.space import IndexSpace, CompositeSpace
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.sandbox.rnn.utils.iteration import SequenceDatasetIterator
from pylearn2.sandbox.rnn.models.mlp_hook import RNNWrapper
from pylearn2.utils.rng import make_np_rng
from pylearn2.linear.matrixmul import MatrixMul

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


class DurationSequencesDataset(VectorSpacesDataset):
    def _create_subset_iterator(self, mode, batch_size=None, num_batches=None,
                                rng=None):
        subset_iterator = resolve_iterator_class(mode)
        if rng is None and subset_iterator.stochastic:
            rng = make_np_rng()
        return subset_iterator(self.get_num_examples(), batch_size,
                               num_batches, rng)

    @wraps(VectorSpacesDataset.iterator)
    def iterator(self, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False, mode=None):
        subset_iterator = self._create_subset_iterator(
            mode=mode, batch_size=batch_size, num_batches=num_batches, rng=rng
        )
        # This should be fixed to allow iteration with default data_specs
        # i.e. add a mask automatically maybe?
        return SequenceDatasetIterator(self, data_specs, subset_iterator,
                                       return_tuple=return_tuple)

class WeightedLogNormalLogLikelihood(Layer):

    __metaclass__ = RNNWrapper

    def __init__(self, layer_name, irange=0.0, init_bias=0.):
        super(WeightedLogNormalLogLikelihood, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.dim = 2

        self.b = sharedX(np.zeros((self.dim,)) + init_bias,
                             name=(layer_name + '_b'))

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim)

        rng = self.mlp.rng

        W = rng.uniform(-self.irange,
                        self.irange,
                        (self.input_dim, self.dim))

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None


    @wraps(Layer.get_params)
    def get_params(self):

        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b.name is not None
        assert self.b not in rval
        rval.append(self.b)
        return rval


    @wraps(Layer.get_weights)
    def get_weights(self):

        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W, = self.transformer.get_params()

        W = W.get_value()

        return W

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        W, = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):
        """
        .. todo::
            WRITEME
        """
        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        W, = self.transformer.get_params()

        W = W.T

        W = W.reshape((self.dim, self.input_space.shape[0],
                       self.input_space.shape[1],
                       self.input_space.num_channels))

        W = Conv2DSpace.convert(W, self.input_space.axes, ('b', 0, 1, 'c'))

        return function([], W)()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        rval = OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

        if (state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)

            mx = state.max(axis=0)
            mean = state.mean(axis=0)
            mn = state.min(axis=0)
            rg = mx - mn

            rval['range_x_max_u'] = rg.max()
            rval['range_x_mean_u'] = rg.mean()
            rval['range_x_min_u'] = rg.min()

            rval['max_x_max_u'] = mx.max()
            rval['max_x_mean_u'] = mx.mean()
            rval['max_x_min_u'] = mx.min()

            rval['mean_x_max_u'] = mean.max()
            rval['mean_x_mean_u'] = mean.mean()
            rval['mean_x_min_u'] = mean.min()

            rval['min_x_max_u'] = mn.max()
            rval['min_x_mean_u'] = mn.mean()
            rval['min_x_min_u'] = mn.min()

        if targets:
            y_target = targets[:, 0]
            cost_multiplier = targets[:, 1]
            mean = state[:, 0]
            sigma = T.exp(state[:, 1])
            nll = self.logprob(y_target, mean, sigma)
            prob_vector = T.exp(-nll)
            rval['prob'] = (prob_vector * cost_multiplier).sum() / (1.0 * cost_multiplier.sum())
            rval['ppl'] = T.exp((nll* cost_multiplier).sum() / (1.0 * cost_multiplier.sum()))
        return rval

    def _linear_part(self, state_below):
        """
        Parameters
        ----------
        state_below : member of input_space
        Returns
        -------
        output : theano matrix
            Affine transformation of state_below
        """
        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        z = self.transformer.lmul(state_below)
        z += self.b

        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        return z

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        p = self._linear_part(state_below)
        return p


    def logprob(self, y_target, mean, sigma):
        return (((T.log(y_target) - mean) ** 2 / (2 * sigma ** 2) + T.log(y_target * sigma * T.sqrt(2 * np.pi))))

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        mean = Y_hat[:, 0] #+ 1.6091597151048114
        sigma = T.exp(Y_hat[:, 1]) #+ 0.26165911509618789
        y_target = Y[:, 0]
        cost_multiplier = Y[:, 1]
        return (self.logprob(y_target, mean, sigma) * cost_multiplier).sum() / (1.0 * cost_multiplier.sum())

    #@wraps(Layer.cost)
    #def cost(self, Y, Y_hat):
    #
    #    return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))
    #
    #@wraps(Layer.cost_from_cost_matrix)
    #def cost_from_cost_matrix(self, cost_matrix):
    #
    #    return cost_matrix.sum(axis=1).mean()
    #
    #@wraps(Layer.cost_matrix)
    #def cost_matrix(self, Y, Y_hat):
    #
    #    return T.sqr(Y - Y_hat)
