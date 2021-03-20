from tensorforce import Agent


class AgentFactory(object):
    @classmethod
    def create_tensorforce_agent(cls, environment, params, seed=0):
        '''
        https://tensorforce.readthedocs.io/en/latest/agents/tensorforce.html
        '''
        agent = Agent.create(
            agent='tensorforce',
            environment=environment,
            memory=params.memory,
            update=dict(unit='timesteps', batch_size=params.batch_size),
            optimizer=dict(type='adam', learning_rate=cls.build_learning_rate_dict(params)),
            exploration=cls.build_explore_dict(params),
            policy=dict(network='auto'),
            objective='policy_gradient',
            reward_estimation=dict(horizon=params.horizon, discount=params.discount),
            config={'seed': seed}
        )
        return agent

    @classmethod
    def build_explore_dict(cls, params):
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
    def build_learning_rate_dict(cls, params):
        lr_dict = dict(
            type=params.lr_type, unit='episodes', num_steps=params.lr_steps,
            initial_value=params.lr_initial
        )
        if params.lr_type == 'linear':
            lr_dict["final_value"] = params.lr_final
        else:
            lr_dict["decay_rate"] = params.lr_decay
        return lr_dict
