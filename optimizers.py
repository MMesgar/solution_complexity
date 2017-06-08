#!/usr/local/bin python
#! -*- coding: utf-8 -*-

import theano
from collections import OrderedDict
from utils import as_floatX
import numpy as np
import theano.tensor as T
#%%
"""
references: 
https://gist.github.com/SnippyHolloW/67effa81dd1cd5a488b4
https://gist.github.com/skaae/ae7225263ca8806868cb
http://chainer.readthedocs.org/en/stable/reference/optimizers.html?highlight=optimizers
http://qiita.com/skitaoka/items/e6afbe238cd69c899b2a
"""
class Optimizer(object):
	def __init__(self, params=None):
		if params is None:
			return NotImplementedError()
		self.params = params

	def updates(self, loss=None):
		if loss is None:
			return NotImplementedError()

		self.updates = OrderedDict()
		self.gparams = [T.grad(loss, param) for param in self.params]

class SGD(Optimizer):
	def __init__(self, learning_rate=0.01, params=None):
		super(SGD, self).__init__(params=params)
		self.learning_rate = 0.01

	def updates(self, loss=None):
		super(SGD, self).updates(loss=loss)

		for param, gparam in zip(self.params, self.gparams):
			self.updates[param] = param - self.learning_rate * gparam

		return self.updates

class MomentumSGD(Optimizer):
	def __init__(self, learning_rate=0.01, momentum=0.9, params=None):
		super(MomentumSGD, self).__init__(params=params)
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.vs = [build_shared_zeros(t.shape.eval(), 'v') for t in self.params]

	def updates(self, loss=None):
		super(MomentumSGD, self).updates(loss=loss)

		for v, param, gparam in zip(self.vs, self.params, self.gparams):
			_v = v * self.momentum
			_v = _v - self.learning_rate * gparam
			self.updates[param] = param + _v
			self.updates[v] = _v

		return self.updates

class AdaGrad(Optimizer):
	def __init__(self, learning_rate=0.01, eps=1e-6, params=None):
		super(AdaGrad, self).__init__(params=params)

		self.learning_rate = learning_rate
		self.eps = eps
		self.accugrads = [build_shared_zeros(t.shape.eval(),'accugrad') for t in self.params]

	def updates(self, loss=None):
		super(AdaGrad, self).updates(loss=loss)

		for accugrad, param, gparam\
		in zip(self.accugrads, self.params, self.gparams):
			agrad = accugrad + gparam * gparam
			dx = - (self.learning_rate / T.sqrt(agrad + self.eps)) * gparam
			self.updates[param] = param + dx
			self.updates[accugrad] = agrad

		return self.updates

class RMSprop(Optimizer):
	def __init__(self, learning_rate=0.001, alpha=0.99, eps=1e-8, params=None):
		super(RMSprop, self).__init__(params=params)

		self.learning_rate = learning_rate
		self.alpha = alpha
		self.eps = eps

		self.mss = [build_shared_zeros(t.shape.eval(),'ms') for t in self.params]

	def updates(self, loss=None):
		super(RMSprop, self).updates(loss=loss)

		for ms, param, gparam in zip(self.mss, self.params, self.gparams):
			_ms = ms*self.alpha
			_ms += (1 - self.alpha) * gparam * gparam
			self.updates[ms] = _ms
			self.updates[param] = param - self.learning_rate * gparam / T.sqrt(_ms + self.eps)

		return self.updates


class AdaDelta(Optimizer):
	def __init__(self, rho=0.95, eps=1e-6, params=None):
		super(AdaDelta, self).__init__(params=params)

		self.rho = rho
		self.eps = eps
		self.accugrads = [build_shared_zeros(t.shape.eval(),'accugrad') for t in self.params]
		self.accudeltas = [build_shared_zeros(t.shape.eval(),'accudelta') for t in self.params]

	def updates(self, loss=None):
		super(AdaDelta, self).updates(loss=loss)

		for accugrad, accudelta, param, gparam\
		in zip(self.accugrads, self.accudeltas, self.params, self.gparams):
			agrad = self.rho * accugrad + (1 - self.rho) * gparam * gparam
			dx = - T.sqrt((accudelta + self.eps)/(agrad + self.eps)) * gparam
			self.updates[accudelta] = (self.rho*accudelta + (1 - self.rho) * dx * dx)
			self.updates[param] = param + dx
			self.updates[accugrad] = agrad

		return self.updates


class Adam(Optimizer):
	def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, gamma=1-1e-8, params=None):
		super(Adam, self).__init__(params=params)

		self.alpha = alpha
		self.b1 = beta1
		self.b2 = beta2
		self.gamma = gamma
		self.t = theano.shared(np.float32(1))
		self.eps = eps

		self.ms = [build_shared_zeros(t.shape.eval(), 'm') for t in self.params]
		self.vs = [build_shared_zeros(t.shape.eval(), 'v') for t in self.params]

	def updates(self, loss=None):
		super(Adam, self).updates(loss=loss)
		self.b1_t = self.b1 * self.gamma ** (self.t - 1)

		for m, v, param, gparam \
		in zip(self.ms, self.vs, self.params, self.gparams):
			_m = self.b1_t * m + (1 - self.b1_t) * gparam
			_v = self.b2 * v + (1 - self.b2) * gparam ** 2

			m_hat = _m / (1 - self.b1 ** self.t)
			v_hat = _v / (1 - self.b2 ** self.t)

			self.updates[param] = param - self.alpha*m_hat / (T.sqrt(v_hat) + self.eps)
			self.updates[m] = _m
			self.updates[v] = _v
		self.updates[self.t] = self.t + 1.0

		return self.updates


def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 





