import cellulus


def test_train_api(experiment_config) -> None:
    # limit number of iterations for this test
    experiment_config.train_config.max_iterations = 1

    cellulus.train(experiment_config)
