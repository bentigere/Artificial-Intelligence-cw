# Part 1 — Genetic Algorithm Integration Testing

**Module:** M33174 Artificial Intelligence  
**Date:** 26 November 2026   
**Branch:** `main`

## Purpose

To check that the integrated Genetic Algorithm runs successfully, generates valid joint-angle data and produces output that can be visualised in MATLAB.

## Test Results

| Test | Result | Status |
|---|---|---|
| GA execution | Completed 100 generations | PASS |
| Fitness progression | Best fitness improved from -150.0052 to -35.0511 | PASS |
| Invalid individuals | 0 reported in the final generation | PASS |
| Gait dimensions | 300 frames × 24 joint angles | PASS |
| Numerical validity | All generated joint angles were finite | PASS |
| MATLAB visualisation | Generated gait loaded and spider model displayed | PASS |
| Ground plane | Ground-reference plane visible in the animation | PASS |

## Testing Procedure

**1. Run the Genetic Algorithm**

```bash
git switch main
git pull origin main
python -m genetic_algorithm.main
```

**2. Validate the generated gait**

```bash
python -c "import numpy as np; a=np.loadtxt('data/best_gait_matlab.csv',delimiter=','); print(a.shape); print(np.isfinite(a).all())"
```

Output:

```text
(300, 24)
True
```

**3. Visualise the gait in MATLAB**

Run `ga_main` from the project directory in MATLAB.

The generated spider gait was displayed with all eight legs and the ground-reference plane visible.

## Evidence

- GA fitness progression graph
- Terminal output showing generation 100 and the final fitness
- CSV dimension and numerical validity check
- MATLAB animation screenshot showing the ground plane

## Conclusion

The integrated GA completed its optimisation run and generated a valid 300-frame joint-angle dataset. MATLAB successfully loaded and visualised the output.

These tests confirm that the GA and MATLAB visualisation work together. They do not verify physical walking stability.

