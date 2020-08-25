import time
import numpy as onp
from jax import numpy as np, grad, random, jit
from jax.ops import index, index_add, index_update
from autograd.misc.optimizers import adam, rmsprop, sgd
from scipy.optimize import minimize

# simplification of the generic 2d oscillator
def dfun(state, w, theta):
    x, y = state
    tau, a, c = theta
    dx = tau * (x - x**3.0 / 3.0 + y)
    dy = (1.0 / tau) * (a - x + c * np.dot(w, x))
    return np.array([dx, dy])

# eeg spectrum forward model
def fwd(gain, state, win):
    eeg_win = np.dot(gain, state[0]).reshape((gain.shape[0], -1, win.size))
    eeg_spec = np.abs(np.fft.fft(eeg_win * win, axis=-1)).mean(axis=1) # (n_eeg, win.size)
    return eeg_spec

# prediction error of model
neval = [0] # yuck but whatev
def make_loss(dt, w, eeg, gain, win):
    n_node = len(w)
    n_eeg, _ = eeg.shape
    def loss(params):
        neval[0] += 1
        # state.shape = (2, n_node, n_time)
        theta = params[-3:]
        state = np.reshape(params[:-3], (2, n_node, -1))
        # predict latent state evolution
        next_state = state + dt * dfun(state, w, theta)
        loss_state = np.sum((next_state[:, :, :-1] - state[:, :, 1:])**2)
        # predict observed data
        loss_eeg = np.sum((eeg - fwd(gain, state, win))**2)
        return loss_eeg + loss_state
    return loss

# create test data
n_node, n_time, n_eeg = 84, 4800, 64
dt = 0.01
theta = tau, a, c = 3.0, 1.04, 0.1
state = np.zeros((2, n_node, n_time))
key = random.PRNGKey(42)
key, r1, r2, r3 = random.split(key, 4)
# state[..., 0] = random.normal(r1, shape=(2, n_node)) / n_node
state = index_update(
    state,
    index[..., 0],
    random.normal(r1, shape=(2, n_node)) / n_node)
gain = random.uniform(r2, shape=(n_eeg, n_node)) / n_node
w = random.normal(r3, shape=(n_node, n_node)) / n_node
eeg = np.zeros((n_eeg, n_time))
eeg = index_update(eeg, index[:, 0], gain.dot(state[0, :, 0]))
for t in range(n_time - 1):
    next_t = state[..., t] + dt * dfun(state[..., t], w, theta)
    state = index_update(state, index[..., t + 1], next_t)
    eeg = index_update(eeg, index[:, 0], gain.dot(state[0, :, t + 1]))

# spectral analysis of eeg data
n_win = 10
win = np.blackman(eeg.shape[-1])
eeg = fwd(gain, state, win)

# make loss & grad, note starting loss
loss = make_loss(dt, w, eeg, gain, win)
gl = jit(grad(loss))
print('ll truth %0.3f' % (np.log(loss(np.concatenate([state.reshape((-1, )), np.array(theta)])))))

# perturb known states for initial guess on optimizers
key, r1, r2 = random.split(key, 3)
state_ = state + random.normal(r1, shape=state.shape)/5
theta_ = np.array(theta) + random.normal(r2, shape=(3,))/5
x0_ = np.concatenate([state_.reshape((-1, )), theta_])

# run different optimizers for certain number of iterations
# and compare performance (in reduction of log loss (rrl) per loss eval)
max_iter = 100
for opt in 'adam rmsprop bfgs tnc'.split():
    tic = time.time()
    print(opt.rjust(8), end=': ')
    x0 = x0_.copy()
    neval[0] = 0
    ll0 = np.log(loss(x0))
    print('ll %0.3f' % ll0, end=' -> ')
    if opt in ('bfgs', 'tnc'):
        method = {'bfgs': 'L-BFGS-B', 'tnc': 'TNC'}[opt]
        def loss_(x): return loss(x).item()
        def jac_(x): return onp.asarray(gl(x)).astype('d')
        for i in range(3):
            x0 = minimize(loss_, x0, method=method, jac=jac_, options={'maxiter': max_iter//3}).x
    elif opt in ('adam', 'rmsprop'):
        cb = lambda x, i: gl(x)
        opt = eval(opt)
        for h in [0.1, 0.01, 0.001]:
            x0 = opt(cb, x0, step_size=h, num_iters=max_iter//3)
    else:
        raise ValueError(opt)
    toc = time.time() - tic
    ll1 = np.log(loss(x0))
    rll_eval = neval[0] / (ll0 - ll1)
    print('%0.3f, %d feval, %0.3fs, %0.3f evals/rll' % (ll1, neval[0], toc, rll_eval))