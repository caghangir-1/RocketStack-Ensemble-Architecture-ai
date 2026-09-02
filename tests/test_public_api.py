"""Smoke tests for the installed RocketStack public interface."""

import rocketstack_ensemble_ai


def test_public_exports() -> None:
    assert rocketstack_ensemble_ai.RocketStack.__name__ == "RocketStack"
    assert rocketstack_ensemble_ai.__version__ == "0.1.0"
    assert set(rocketstack_ensemble_ai.__all__) == {
        "RocketStack",
        "__version__",
        "ascentTheRocket_binary_model",
        "ascentTheRocket_multiclass_model",
    }


def test_binary_factory() -> None:
    model = rocketstack_ensemble_ai.ascentTheRocket_binary_model(
        level=1,
        iffeatselection=False,
        n_jobs=1,
        verbose=False,
    )
    assert model.problem_type == "binary"
    assert model.num_of_level == 1
    assert hasattr(rocketstack_ensemble_ai.RocketStack(), "ascentTheRocket_binary_model")
    names, estimators = model._make_model_pool()
    assert len(names) == 20
    assert set(names) == set(estimators)


def test_multiclass_factory() -> None:
    model = rocketstack_ensemble_ai.ascentTheRocket_multiclass_model(
        level=1,
        iffeatselection_or_not=False,
        n_jobs=1,
        verbose=False,
    )
    assert model.problem_type == "multiclass"
    assert model.num_of_level == 1
    assert hasattr(rocketstack_ensemble_ai.RocketStack(), "ascentTheRocket_multiclass_model")
    names, estimators = model._make_model_pool()
    assert len(names) == 14
    assert set(names) == set(estimators)
