
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dopamine.jax import losses
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.metrics import statistics_instance
from dopamine.replay_memory import prioritized_replay_buffer
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf


    coeff_vector = jnp.arange(lap_dim, 0, -1)
    coeff_vector = np.concatenate((coeff_vector, np.zeros(1)))
    def neg_loss_fn(phi_u, phi_v):
      loss = 0
      for dim in range(lap_dim, 0, -1):
        coeff = coeff_vector[dim-1] - coeff_vector[dim]
        x_norm = jnp.sqrt(jnp.dot(phi_u[:dim], phi_u[:dim]))
        y_norm = jnp.sqrt(jnp.dot(phi_v[:dim], phi_v[:dim]))
        dot_product = jnp.dot(phi_u[:dim], phi_v[:dim])
        loss += coeff * (
          dot_product ** 2 - jnp.log(1 + x_norm)  - jnp.log(1 + y_norm)  )
      return loss
    neg_loss_vmap = jax.vmap(neg_loss_fn)

    def _update_lap(
      rng_key, opt_state, params, transitions):#, transitions_u, transitions_v):
      """Computes learning update from batch of replay transitions."""
      rng_key, update_key = jax.random.split(rng_key)

      def lap_loss_fn(params, update_key):
        """Calculates loss given network parameters and transitions."""
        phis = lap_network.apply(params, update_key,
                              transitions).q_values
        phis = jnp.split(phis, 4, axis=0)
        phi_tm1 = phis[0]
        phi_t = phis[1]
        phi_u = phis[2]
        phi_v = phis[3]
        pos_loss = ((phi_tm1 - phi_t)**2).dot(coeff_vector[:lap_dim])
        neg_loss  = neg_loss_vmap(phi_u, phi_v)
        loss = pos_loss + neg_loss
        loss = rlax.clip_gradient(loss, -grad_error_bound, grad_error_bound)
        chex.assert_shape(loss, (self._batch_size,))
        loss = jnp.mean(loss)
        return loss, (jnp.mean(pos_loss), jnp.mean(neg_loss))

      grads, (pos_loss, neg_loss) = jax.grad(
          lap_loss_fn, has_aux=True)(params, update_key)
      updates, new_opt_state = rep_optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return rng_key, new_opt_state, new_params, pos_loss, neg_loss

@functools.partial(jax.jit, static_argnums=(0, 3, 12))
def train_rep(network_def, rep_params, optimizer, optimizer_state,
          states, next_states):
  """Run a training step."""
  def loss_fn(params, target, loss_multipliers):
    def q_online(state):
      return network_def.apply(params, state, support)

    logits = jax.vmap(q_online)(states).logits
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
    loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
        target,
        chosen_action_logits)
    mean_loss = jnp.mean(loss_multipliers * loss)
    return mean_loss, loss

  def q_target(state):
    return network_def.apply(target_params, state, support)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  target = target_distribution(q_target,
                               next_states,
                               rewards,
                               terminals,
                               support,
                               cumulative_gamma)

  # Get the unweighted loss without taking its mean for updating priorities.
  (mean_loss, loss), grad = grad_fn(rep_params, target, loss_weights)
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=rep_params)
  rep_params = optax.apply_updates(rep_params, updates)
  return optimizer_state, rep_params, loss, mean_loss


@functools.partial(jax.jit, static_argnums=(0, 3, 12))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, loss_weights,
          support, cumulative_gamma):
  """Run a training step."""
  def loss_fn(params, target, loss_multipliers):
    def q_online(state):
      return network_def.apply(params, state, support)

    logits = jax.vmap(q_online)(states).logits
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
    loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
        target,
        chosen_action_logits)
    mean_loss = jnp.mean(loss_multipliers * loss)
    return mean_loss, loss

  def q_target(state):
    return network_def.apply(target_params, state, support)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  target = target_distribution(q_target,
                               next_states,
                               rewards,
                               terminals,
                               support,
                               cumulative_gamma)

  # Get the unweighted loss without taking its mean for updating priorities.
  (mean_loss, loss), grad = grad_fn(online_params, target, loss_weights)
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, mean_loss


@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None))
def target_distribution(target_network, next_states, rewards, terminals,
                        support, cumulative_gamma):
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  target_support = rewards + gamma_with_terminal * support
  next_state_target_outputs = target_network(next_states)
  q_values = jnp.squeeze(next_state_target_outputs.q_values)
  next_qt_argmax = jnp.argmax(q_values)
  probabilities = jnp.squeeze(next_state_target_outputs.probabilities)
  next_probabilities = probabilities[next_qt_argmax]
  return jax.lax.stop_gradient(
      project_distribution(target_support, next_probabilities, support))


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11))
def select_action(network_def, params, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn, support):
  epsilon = jnp.where(eval_mode,
                      epsilon_eval,
                      epsilon_fn(epsilon_decay_period,
                                 training_steps,
                                 min_replay_history,
                                 epsilon_train))

  rng, rng1, rng2 = jax.random.split(rng, num=3)
  p = jax.random.uniform(rng1)
  return rng, jnp.where(
      p <= epsilon,
      jax.random.randint(rng2, (), 0, num_actions),
      jnp.argmax(network_def.apply(params, state, support).q_values))


@gin.configurable
class JaxRainbowAgent(dqn_agent.JaxDQNAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=networks.RainbowNetwork,
               rep_network=networks.NatureDQNNetwork,
               num_atoms=51,
               vmin=None,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               optimizer='adam',
               seed=None,
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False,
               num_options=0,
               option_prob=0.0,
               rep_dim=10,):
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self._num_atoms = num_atoms
    # If vmin is not specified, set it to -vmax similar to C51.
    vmin = vmin if vmin else -vmax
    self._support = jnp.linspace(vmin, vmax, num_atoms)
    self._replay_scheme = replay_scheme

    self.num_options = num_options
    self.option_prob = option_prob
    self.rep_dim = rep_dim

    if preprocess_fn is None:
      self.rep_network_def = rep_network(num_actions=rep_dim)
      self.rep_preprocess_fn = networks.identity_preprocess_fn
    else:
      self.rep_network_def = rep_network(num_actions=rep_dim,
                                 inputs_preprocessed=True)
      self.rep_preprocess_fn = preprocess_fn

    super(JaxRainbowAgent, self).__init__(
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=functools.partial(network,
                                  num_atoms=num_atoms),
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        optimizer=optimizer,
        seed=seed,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency,
        allow_partial_reload=allow_partial_reload)

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    self.online_params = self.network_def.init(rng, x=self.state,
                                               support=self._support)
    self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = self.online_params

    self.options = []
    for o in range(self.num_options):
      self._rng, rng = jax.random.split(self._rng)
      online_params = self.network_def.init(rng, x=self.state,
                                              support=self._support)
      optimizer_state = self.optimizer.init(self.online_params)
      target_network_params = online_params
      self.options.append(Option(
          online_params=online_params,
          target_network_params=target_network_params,
          optimizer_state=optimizer_state))

      self._rng, rng = jax.random.split(self._rng)
      self.rep_params = self.rep_network_def.init(rng, x=self.state,)
      self.rep_optimizer_state = self.optimizer.init(self.rep_params)

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    # Both replay schemes use the same data structure, but the 'uniform' scheme
    # sets all priorities to the same value (which yields uniform sampling).
    return prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)

  # TODO(psc): Refactor this so we have a class _select_action that calls
  # select_action with the right parameters. This will allow us to avoid
  # overriding begin_episode.
  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    (   self._rng, 
        self.action 
    ) = select_action(
        self.network_def,
        self.online_params,
        self.preprocess_fn(self.state),
        self._rng,
        self.num_actions,
        self.eval_mode,
        self.epsilon_eval,
        self.epsilon_train,
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_fn,
        self._support
    )
    # TODO(psc): Why a numpy array? Why not an int?
    self.action = onp.asarray(self.action)
    return self.action

  def step(self, reward, observation):
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    (   self._rng, 
        self.action
    ) = select_action(
        self.network_def,
        self.online_params,
        self.preprocess_fn(self.state),
        self._rng,
        self.num_actions,
        self.eval_mode,
        self.epsilon_eval,
        self.epsilon_train,
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_fn,
        self._support
    )
    self.action = onp.asarray(self.action)
    return self.action

  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_params to target_network_params if training
    steps is a multiple of target update period.
    """
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:

        self._sample_from_replay_buffer()
        states = self.preprocess_fn(self.replay_elements['state'])
        next_states = self.preprocess_fn(self.replay_elements['next_state'])
        self.rep_optimizer_state, self.rep_params, loss = train_rep(
            self.rep_network_def,
            self.rep_params,
            self.optimizer,
            self.rep_optimizer_state,
            states,
            next_states,)

        for o in np.random.choice(self._num_options, 3, replace=False):
          option = self.options[o]

          self._sample_from_replay_buffer()

          if self._replay_scheme == 'prioritized':
            probs = self.replay_elements['sampling_probabilities']
            # Weight the loss by the inverse priorities.
            loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
            loss_weights /= jnp.max(loss_weights)
          else:
            loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

          option.optimizer_state, self.online_params, loss, mean_loss = train(
              self.network_def,
              option.online_params,
              option.target_network_params,
              self.optimizer,
              option.optimizer_state,
              self.preprocess_fn(self.replay_elements['state']),
              self.replay_elements['action'],
              self.preprocess_fn(self.replay_elements['next_state']),
              self.replay_elements['reward'],
              self.replay_elements['terminal'],
              loss_weights,
              self._support,
              self.cumulative_gamma)

        self._sample_from_replay_buffer()

        if self._replay_scheme == 'prioritized':
          probs = self.replay_elements['sampling_probabilities']
          # Weight the loss by the inverse priorities.
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
        else:
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

        self.optimizer_state, self.online_params, loss, mean_loss = train(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            self.preprocess_fn(self.replay_elements['state']),
            self.replay_elements['action'],
            self.preprocess_fn(self.replay_elements['next_state']),
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            loss_weights,
            self._support,
            self.cumulative_gamma)

        if self._replay_scheme == 'prioritized':
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          with self.summary_writer.as_default():
            tf.summary.scalar('CrossEntropyLoss', mean_loss,
                              step=self.training_steps)
          self.summary_writer.flush()
          if hasattr(self, 'collector_dispatcher'):
            self.collector_dispatcher.write(
                [statistics_instance.StatisticsInstance(
                    'Loss', onp.asarray(mean_loss), step=self.training_steps),
                 ],
                collector_allowlist=self._collector_allowlist)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1


def project_distribution(supports, weights, target_support):
  """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  Args:
    supports: Jax array of shape (num_dims) defining supports for
      the distribution.
    weights: Jax array of shape (num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Jax array of shape (num_dims) defining support of the
      projected distribution. The values must be monotonically increasing. Vmin
      and Vmax will be inferred from the first and last elements of this Jax
      array, respectively. The values in this Jax array must be equally spaced.

  Returns:
    A Jax array of shape (num_dims) with the projection of a batch
    of (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
  v_min, v_max = target_support[0], target_support[-1]
  # `N` in Eq7.
  num_dims = target_support.shape[0]
  # delta_z = `\Delta z` in Eq7.
  delta_z = (v_max - v_min) / (num_dims - 1)
  # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
  clipped_support = jnp.clip(supports, v_min, v_max)
  # numerator = `|clipped_support - z_i|` in Eq7.
  numerator = jnp.abs(clipped_support - target_support[:, None])
  quotient = 1 - (numerator / delta_z)
  # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
  clipped_quotient = jnp.clip(quotient, 0, 1)
  # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))` in Eq7.
  inner_prod = clipped_quotient * weights
  return jnp.squeeze(jnp.sum(inner_prod, -1))
