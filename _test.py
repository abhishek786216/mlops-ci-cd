import pytest
from app import predict_species

def test_predict_species():
    # Typical Iris setosa measurements
    features_setosa = [5.1, 3.5, 1.4, 0.2]
    assert predict_species(features_setosa) == "setosa"

    # Typical Iris versicolor measurements
    features_versicolor = [6.0, 2.8, 4.5, 1.3]
    assert predict_species(features_versicolor) == "versicolor"

    # Typical Iris virginica measurements
    features_virginica = [6.9, 3.1, 5.4, 2.1]
    assert predict_species(features_virginica) == "virginica"

def test_invalid_input():
    with pytest.raises(ValueError):
        # Passing strings instead of floats
        predict_species(["a", "b", "c", "d"])
