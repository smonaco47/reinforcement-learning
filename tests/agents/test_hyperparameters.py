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

def test_exponent_range():
    range = Hyperparameters.random_range_exponential(-6, -3)

    assert range[1] >= range[0]
    assert 1e-6 < range[0]
    assert range[1] < 9e-3