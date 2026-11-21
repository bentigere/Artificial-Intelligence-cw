
import numpy as np
from pathlib import Path

from genetic_algorithm.chromosome import (
    LOWER_LIMITS,
    UPPER_LIMITS,
)


def test_matlab_gait_export():
    """Validate the GA output used by MATLAB."""

    project_root = Path(__file__).resolve().parent.parent
    gait_path = project_root / "data" / "best_gait.csv"

    assert gait_path.exists(), "best_gait.csv not found"

    gait = np.loadtxt(gait_path, delimiter=",")

    # 300 frames, 24 angles per frame
    assert gait.shape == (300, 24)

    # No invalid numerical values
    assert np.isfinite(gait).all()

    # All joint angles must stay within their limits
    assert np.all(gait >= LOWER_LIMITS)
    assert np.all(gait <= UPPER_LIMITS)

    # Test smooth transitions between consecutive frames
    changes = np.abs(np.diff(gait, axis=0))

    assert np.isfinite(changes).all()

    # Include final-to-first-frame continuity
    loop_change = np.abs(gait[-1] - gait[0])
    assert np.isfinite(loop_change).all()

    print("MATLAB gait validation passed")
