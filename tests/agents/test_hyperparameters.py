from agents.hyperparameters import Hyperparameters


def test_randomization():
    hyperparameters = Hyperparameters()
    hyperparameters.randomize()

    assert hyperparameters.lr_initial >= hyperparameters.lr_final
    assert hyperparameters.lr_type in ['linear', 'exponential']

    assert hyperparameters.explore_initial >= hyperparameters.explore_final
    assert hyperparameters.explore_type in ['linear', 'exponential']


def test_randomization_loop():
    hp = Hyperparameters()
    for i in range(10):
        hp.randomize()
        print(hp.lr_initial, hp.lr_final, hp.lr_type, hp.lr_steps, hp.explore_initial, hp.explore_final,
              hp.explore_type, hp.explore_steps)
    assert False