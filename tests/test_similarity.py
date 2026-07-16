import math
import pytest

from batch_processor.similarity import cosine_similarity


def test_cosine_similarity_geometry():
    assert math.isclose(
        cosine_similarity([1.0, 0.0], [10.0, 0.0]),
        1.0,
    )

    assert math.isclose(
        cosine_similarity([1.0, 0.0], [0.0, 2.0]),
        0.0,
    )

    assert math.isclose(
        cosine_similarity([1.0, 0.0], [-3.0, 0.0]),
        -1.0,
    )


def test_cosine_similarity_rejects_invalid_vectors():
    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarity([1.0], [2.0, 3.0])

    with pytest.raises(ValueError, match="zero"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])
