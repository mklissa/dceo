# Overview
This repository contains JAX code for the implementation of the Deep Covering Eigenoptions (DCEO) algorithm which learns a diverse set of task-agnostic options to improve exploration. We additionally provide a general guide to the integration of Lplacian-based options in a different codebase.

**[Deep Laplacian-based Options for Temporally-Extended Exploration](https://proceedings.mlr.press/v202/klissarov23a/klissarov23a.pdf)**

by [Martin Klissarov](https://mklissa.github.io) and [Marlos C. Machado](https://webdocs.cs.ualberta.ca/~machado/). 

DCEO is based on the idea of the [Representation Driven Option Disovery](https://medium.com/@marlos.cholodovskis/the-representation-driven-option-discovery-cycle-e3f5877696c2) cycle where options and representations are iteratively refined and bootstrapped from eachother. In this work, we argue that the Laplacian representation (also referred to as [Proto-Value Functions](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PVF.pdf)) is very well-suited scaffold for option discovery as it naturally encodes the topology of the environment at various timescales.

![dceo](https://github.com/mklissa/deco_dopamine/assets/22938475/285c7ed1-f1a3-499f-8655-5802ee4738c9)

# Atari Results

![mr_repo](https://github.com/mklissa/dceo/assets/22938475/4d028716-49ee-41f6-bb4e-38bf653e0697)


In this repository, we can only share the code with respect to the Montezuma's Revenge experiments, which are built on the [Dopamine](https://github.com/google/dopamine) codebase. To replicate the results on Montezuma's Revenge, only two files need to be added with respect to the original repository: `full_rainbow_dceo.gin` and `full_rainbow_dceo.py` which are both located in `dopamine/jax/agents/full_rainbow`. 

For the sake of simplicity we include the complete Dopamine code source. To replicate results on Montezuma's Revenge, simply run the following

```
python -um dopamine.discrete_domains.train --base_dir results_folder \
     --gin_files "dopamine/jax/agents/full_rainbow/configs/full_rainbow_dceo.gin" \
     --gin_bindings "atari_lib.create_atari_environment.game_name='MontezumaRevenge'" \
     --gin_bindings "JaxFullRainbowAgentDCEO.seed=1337"
```
The experiments should take from 5 to 7 days in order to run the complete 200M timesteps of training

**Installation**
1. To run experiments on Atari, you will need to get the ROMS as described [here](https://github.com/Farama-Foundation/Arcade-Learning-Environment). For example, by using `` on directory containing the ROMS to import them, or by setting `ALE_PY_ROM_DIR` in you `bashrc` to point to the ROM directory.

2. To install the necessary requirements, with a `virutalenv` or conda environment, simply do

```
pip install -r requirements.txt
```

3. Finally, install Dopamine using the editable mode

```
pip install -e .
```

# How do I use Laplacian-based options in my codebase?

As DCEO performs remarkably well compared to several HRL baseline (such as DIAYN, CIC and DCO) as well as exploration baselines (such as RND and Counts), we believe it is important to facilitate the usage of Laplacian-based options. Therefore we must answer, what is the minimum amount of code needed to launch some experiments using such options? We answer this question by point at snippets of code in this repository.

**Learning the Laplacian representation**
The first step is to integrate the loss for learning the Laplacian representation on which the options are based. This can be done with the following

```
def neg_loss_fn(phi_u, phi_v):
  loss = 0
  for dim in range(rep_dim, 0, -1):
    coeff = coeff_vector[dim-1] - coeff_vector[dim]
    x_norm = jnp.sqrt(jnp.dot(phi_u[:dim], phi_u[:dim]))
    y_norm = jnp.sqrt(jnp.dot(phi_v[:dim], phi_v[:dim]))
    dot_product = jnp.dot(phi_u[:dim], phi_v[:dim])
    loss += coeff * (
      dot_product ** 2 - jnp.log(1 + x_norm)  - jnp.log(1 + y_norm)  )
  return loss

neg_loss_vmap = jax.vmap(neg_loss_fn)

def train_rep(rep_params, optimizer, optimizer_state, states):

  def loss_fn(params):
    """Calculates loss given network parameters and transitions."""
    def rep_online(state):
      return rep_network_def.apply(params, state)
    phis = jax.vmap(rep_online)(states).q_values

    phis = jnp.split(phis, 4, axis=0)
    phi_tm1, phi_t, phi_u, phi_v = phis[0], phis[1], phis[2], phis[3]

    pos_loss = ((phi_tm1 - phi_t)**2).dot(coeff_vector[:rep_dim])
    neg_loss = neg_loss_vmap(phi_u, phi_v)

    loss = pos_loss + neg_loss
    loss = jnp.mean(loss)

    return loss, (jnp.mean(pos_loss), jnp.mean(neg_loss))

  grad_fn = jax.grad(loss_fn, has_aux=True)
  grad, (pos_loss, neg_loss) = grad_fn(rep_params)
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=rep_params)
  rep_params = optax.apply_updates(rep_params, updates)
  return optimizer_state, rep_params, pos_loss, neg_loss

self._train_rep = jax.jit(train_rep, static_argnames=('optimizer'))
```

The optimization can then be done by sampling state from a buffer and calling `_train_rep`

```
### Train the representation module ###
phi_tm1, phi_t, phi_u, phi_v = self._rep_sample_from_replay_buffer()
all_phis = (phi_tm1, phi_t, phi_u, phi_v)

(self.rep_optimizer_state, self.rep_params,
 pos_loss, neg_loss) = self._train_rep(
    self.rep_params, self.optimizer, 
    self.rep_optimizer_state, all_phis)
```
Here `phi_tm1` and `phi_t` are consecutive states whereas `phi_u` and `phi_v` are random samples.

**Option Learning**
To learn each of the options, we need to define the intrinsic reward from Section 3 in the paper. We then iterate over all the options and update each of them using this intrinsic reward signal.

```
for o in onp.random.choice(self.num_options, 1, replace=False):
  option = self.options[o]

  self._sample_from_replay_buffer()
  states = self.preprocess_fn(self.replay_elements['state'])
  next_states = self.preprocess_fn(self.replay_elements['next_state'])

  all_states = onp.vstack((states, next_states))
  rep = jax.vmap(self._get_rep, in_axes=(None, 0))(
      self.rep_params, all_states).q_values
  rep = onp.asarray(rep)
  intr_rew = rep[len(rep) // 2:, o] - rep[:len(rep) // 2, o]

  # Usual Q-Learning, or your preferred RL method
```

**Option Execution**
Our option execution algorithm defined in Algorithm 1 in the paper is straightforward as it does not rely on meta-policies or parametrized terminations functions. The following code implements it.

```
option_term = option_term or onp.random.rand() < (1 / dur)
epsilon = jnp.where(
    eval_mode, epsilon_eval,
    epsilon_fn(epsilon_decay_period, training_steps, min_replay_history,
               epsilon_train))

if option_term:
  cur_opt = None
  if onp.random.rand() < epsilon:
    if onp.random.rand() < option_prob:
      cur_opt = onp.random.randint(num_options)
      option_term = False

params = main_policy_params if cur_opt is None else options[cur_opt].online_params
rng, action = act(
  network_def, params, state, rng, num_actions, eval_mode, support, epsilon)
```

Hopefully this should make it easier to implement Laplacian-based options.

# Citation
If you found this repository useful, we invite you to cite our work. 

```
@inproceedings{klissarov2023deep,
  title = {Deep Laplacian-based Options for Temporally-Extended Exploration},
  author = {Martin Klissarov and Marlos C. Machado},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2023}
}
```
