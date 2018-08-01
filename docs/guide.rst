================
Quickstart Guide
================

The basic idea of reinforcement learning is to find a behavior for an `agent` inside an `environment` that leads to a
maximal `reward`. Such a behavior is called a `policy` and it decides what `action` to take based on the current
`observation` (also called `state`).

For example, the environment can be an Atari game. In this case the reward is the score, the actions are the controller
actions, and the current frame/image of the game is an observation.

The `gym library <https://gym.openai.com/>`_ (`GitHub <https://github.com/openai/gym>`_) by OpenAI provides several
types of environments. A basic reinforcement learning setup to learn a policy for the `Breakout` environment could look
like this:

.. code::

    import gym

    # create the environment
    env = gym.make('BreakoutNoFrameskip-v4')

    # receive an initial observation (frame) to select the first action
    observation = env.reset()

    while True:
        # let the current policy select an action
        action = policy(observation)

        # execute the action and take one step in the environment (go to next frame)
        next_observation, reward, terminal, info = env.step(action)

        # improve the policy based on this experience
        improve_policy(observation, action, reward, terminal, next_observation)

        observation = next_observation

        if terminal:
            observation = env.reset()

:code:`terminal` indicates whether the game ended, so the game has to be reset. :code:`reward` is just a number that
represents the points that were achieved in this step. :code:`info` contains debug information (the current number of
lives).

`A2C` and `ACKTR` actually use `multiple environments` at once by running them in multiple subprocesses. This means that
we can improve the policy faster, since we simply have more observations and rewards available.
For that reason there is :obj:`~actorcritic.multi_env.MultiEnv`:

.. code::

    from actorcritic.multi_env import MultiEnv

    envs = create_environments()  # create multiple environments
    multi_env = MultiEnv(envs)

Yet the crucial parts are :code:`policy(observation)` and
:code:`improve_policy(observation, action, reward, next_observation)`. We need to know how to define a policy and
especially how to improve it.

`Actor-critic` methods define the policy as a probability distribution, such that it computes the probability of
every action based on the current observation. Then these probabilities are used to sample one of the actions.
For example, if the ball approaches the bottom in `Breakout`, the probability to move the paddle towards the ball should
be high.

We typically use a neural network to compute these probabilities. Then the observations (frames) are sent into the
network, which produces a score for every action. These scores can be passed in the softmax function to obtain
probabilities. :obj:`~actorcritic.envs.atari.model.AtariModel` provides a neural network and a policy made for Atari
environments:

.. code::

    from actorcritic.envs.atari.model import AtariModel

    # observation_space and action_space define the type and shape of the observations and actions
    # e.g. the size of the frames
    model = AtariModel(multi_env.observation_space, multi_env.action_space)

Additionally `A2C` and `ACKTR` do not take one step only and improve the policy immediately. Instead they take multiple
steps and use all the experienced observations and rewards to improve the policy.
A :obj:`~actorcritic.agents.MultiEnvAgent` simplifies this process. It takes the neural network and the policy
(the `'model'`), and the environments. Then we just have to call :meth:`~actorcritic.agents.MultiEnvAgent.interact` and
it uses the policy to take multiple steps:

.. code::

    from actorcritic.agents import MultiEnvAgent

    agent = MultiEnvAgent(multi_env, model, num_steps=5)

    while True:
        # take 5 steps in all environments
        # session is a tf.Session used to compute the values of the neural network
        observations, actions, rewards, terminals, next_observations, infos = agent.interact(session)

        # improve the policy based on this experience
        improve_policy(observations, actions, rewards, terminals, next_observations)

In `actor-critic` methods we do not define a loss function directly, but a `policy objective` function to optimize the
neural network. It needs the observations, the actions, and the rewards that the agent experienced. Then we can learn
through the policy objective, which looks at the rewards in order to decide whether the actions were good or not.

Furthermore we need a `baseline` function that enhances the policy objective. It should express how much reward
we can expect if we would follow our policy proceeding from the observations we just have seen. This helps the policy to
decide whether the actions it has taken actually were better or worse than expected. This `baseline` function is the
`'critic'` of `actor-critic` (the policy is the `'actor'`). It distinguishes actor-critic methods from `policy gradient`
methods which just have an `'actor'`.

Unfortunately we do not have such a `baseline` function. That is why we will learn the `baseline`, too, at the same time
as the policy. Therefore an :obj:`~actorcritic.model.ActorCriticModel` like the
:obj:`~actorcritic.envs.atari.model.AtariModel` has to provide a baseline. `A2C` and `ACKTR` use the
`state-value function` which indeed tells us how much reward we can expect from a given observation.

It can be beneficial to use the same neural network as the policy for the baseline.
:obj:`~actorcritic.envs.atari.model.AtariModel` does exactly this.

In summary we need a :obj:`~actorcritic.objectives.ActorCriticObjective`. The policy objective of `A2C` and `ACKTR` is
implemented in :obj:`~actorcritic.objectives.A2CObjective`. It `discounts` the rewards and uses
`entropy regularization` (see :obj:`~actorcritic.objectives.A2CObjective`).

.. code::

    from actorcritic.objectives import A2CObjective

    objective = A2CObjective(model, discount_factor=0.99, entropy_regularization_strength=0.01)

Next we need an optimizer for our neural network:

.. code::

    import tensorflow as tf

    # A2C uses the RMSProp optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0007)

    # create an 'optimize' operation that we can call
    # use optimize_shared() since we share the network between the policy and the baseline
    optimize_op = objective.optimize_shared(optimizer)

That is all. We can use all variables defined above to run the A2C algorithm:

.. code::

    while True:
        # take multiple steps in all environments
        observations, actions, rewards, terminals, next_observations, infos = agent.interact(session)

        # improve the policy and the baseline
        session.run(optimize_op, feed_dict={
            model.observations_placeholder: observations,
            model.bootstrap_observations_placeholder: next_observations,
            model.actions_placeholder: actions,
            model.rewards_placeholder: rewards,
            model.terminals_placeholder: terminals
        })

:obj:`~actorcritic.model.ActorCriticModel.bootstrap_observations_placeholder` is needed to compute the
:obj:`~actorcritic.model.ActorCriticModel.bootstrap_values`, which are used in the policy objective.

In order to use `ACKTR` we just have to change the optimizer to a :obj:`kfac.KfacOptimizer`.

See `a2c_acktr.py <https://github.com/jrobine/actor-critic/blob/master/actorcritic/examples/atari/a2c_acktr.py>`_ for a
full implementation, especially how to implement :code:`create_environments()` and how to use the K-FAC optimizer.
