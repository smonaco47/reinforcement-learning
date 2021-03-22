from tensorforce.agents import DoubleDQN, TensorforceAgent


class AgentFactory(object):
    @classmethod
    def create_tensorforce_agent(cls, environment, params, seed=0):
        '''
        https://tensorforce.readthedocs.io/en/latest/agents/tensorforce.html
        '''
        agent = TensorforceAgent.create(
            environment=environment,
            memory=params.memory,
            update=dict(unit='timesteps', batch_size=params.batch_size),
            optimizer=dict(type='adam', learning_rate=cls._build_learning_rate_dict(params)),
            exploration=cls._build_explore_dict(params),
            policy=dict(network='auto'),
            objective='policy_gradient',
            reward_estimation=dict(horizon=params.horizon, discount=params.discount),
            config={'seed': seed}
        )
        return agent

    @classmethod
    def create_ddqn_agent(cls, environment, params, seed=0):
        '''
        https://tensorforce.readthedocs.io/en/latest/agents/double_dqn.html
        '''
        agent = DoubleDQN(
            states=environment.states(),
            actions=environment.actions(),
            memory=params.memory,
            batch_size=params.batch_size,  # Timesteps per update batch
            update_frequency=params.batch_size,
            start_updating=params.batch_size,  # number of steps prior to first update
            network='auto',
            learning_rate=cls._build_learning_rate_dict(params),
            horizon=params.horizon,
            discount=params.discount,  # Default 0.99
            exploration=cls._build_explore_dict(params),
            config={'seed': seed}
        )
        agent.initialize()
        return agent

    @classmethod
    def _build_explore_dict(cls, params):
        explore_dict = dict(
            type=params.explore_type, unit='episodes', num_steps=params.explore_steps,
            initial_value=params.explore_initial, final_value=params.explore_final
        )
        if params.explore_type == 'linear':
            explore_dict["final_value"] = params.explore_final
        else:
            explore_dict["decay_rate"] = params.explore_decay
        return explore_dict

    @classmethod
    def _build_learning_rate_dict(cls, params):
        lr_dict = dict(
            type=params.lr_type, unit='episodes', num_steps=params.lr_steps,
            initial_value=params.lr_initial
        )
        if params.lr_type == 'linear':
            lr_dict["final_value"] = params.lr_final
        else:
            lr_dict["decay_rate"] = params.lr_decay
        return lr_dict
