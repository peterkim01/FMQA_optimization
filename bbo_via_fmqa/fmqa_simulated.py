from datetime import datetime
import sys, os
import time
import csv
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fmqa")))

import numpy as np
import matplotlib.pyplot as plt

# Helper and module imports
import read_grid
from FM_surrogate import train_surrogate_model, print_final_equation
from ising_machine import solve_surrogate_SA
from dimod import SimulatedAnnealingSampler

# --- Load Full Dataset ---
# path = path/to/your/dataset.csv
path = os.environ.get("FMQA_DATASET", "./qhd_2D_graphs/alpine1_30x30.csv")

graphtype = os.path.splitext(os.path.basename(path))[0]
grid, obj_min, obj_max, x_bound, y_bound = read_grid.load_grid(filename=path)
print(f"Grid loaded: {len(grid)} points, x in [0,{x_bound}], y in [0,{y_bound}]")
print(f"Objective range: [{obj_min}, {obj_max}]")

# --- Parameters ---
# Each cycle trains the surrogate once and evaluates at most one new point.
sampler = SimulatedAnnealingSampler()
start_time = time.perf_counter()

# --- Helper Functions ---
def evaluate(x, y, grid):
    """Evaluate objective from grid."""
    return read_grid.obj_funct([x, y], grid)

def scale_value(y_val):
    """Normalize objective values to [0,1] range."""
    if y_val == np.inf:
        return np.inf
    return (y_val - obj_min) / (obj_max - obj_min)

def run_fmqa(grid, x_bound, y_bound, max_cycles, convergence_patience, sampler, all_feasible_points):
    """Run the FMQA optimization loop."""
    # --- Initial dataset (blackbox start) ---
    start_point = random.choice(all_feasible_points)
    start_val_raw = evaluate(*start_point, grid)

    evaluated_points = {start_point}
    xs = [read_grid.coord_bits(start_point[0], start_point[1], x_bound, y_bound)]
    x_vectors = [[int(bit) for bit in xs[0]]]
    ys = [scale_value(start_val_raw)]

    # --- Tracking and Convergence ---
    best_val_raw = start_val_raw
    best_coord = start_point
    history = [best_val_raw]
    no_improvement_count = 0

    loop_records = [
        {
            "cycle": 0,
            "px": start_point[0],
            "py": start_point[1],
            "objective": start_val_raw,
            "best_val": best_val_raw,
            "evaluated_count": len(evaluated_points),
            "no_improvement_count": no_improvement_count,
            "status": "initial_random",
        }
    ]

    # --- FMQA Loop ---
    final_model = None

    for t in range(max_cycles):
        print(f"\n=== Cycle {t+1}/{max_cycles} ===")
        print(f"Cycle {t+1}: best so far = {best_val_raw} at {best_coord}")

        # Train surrogate model
        fm = train_surrogate_model(xs, ys, x_vectors=x_vectors)
        final_model = fm

        # Solve surrogate to get candidate
        px, py = solve_surrogate_SA(
            fm,
            x_bound,
            y_bound,
            evaluated_points,
            sampler,
            grid
        )

        # Debug info
        print((px, py), "in grid?", (px, py) in grid)
        if (px, py) in grid:
            print("grid value:", grid[(px, py)])

        # --- Skip invalid or duplicate proposals ---
        if px is None or py is None:
            print("All proposed samples were already evaluated or invalid.")
            no_improvement_count += 1
            history.append(best_val_raw)

            loop_records.append({
                "cycle": t + 1,
                "px": None,
                "py": None,
                "objective": None,
                "best_val": best_val_raw,
                "evaluated_count": len(evaluated_points),
                "no_improvement_count": no_improvement_count,
                "status": "invalid_or_duplicate",
            })

            print(f"No improvement for {no_improvement_count} consecutive cycles.")

            if no_improvement_count >= convergence_patience:
                print(f"\nConvergence reached after {t+1} cycles. Stopping.")
                break

            continue

        # --- Evaluate candidate ---
        obj_val_raw = evaluate(px, py, grid)
        evaluated_points.add((px, py))
        
        loop_records.append({
            "cycle": t + 1,
            "px": px,
            "py": py,
            "objective": obj_val_raw,
            "best_val": best_val_raw,
            "evaluated_count": len(evaluated_points),
            "no_improvement_count": no_improvement_count,
            "status": "evaluated",
        })
        # --- Print objective every iteration ---
        if np.isfinite(obj_val_raw):
            print(f"Cycle {t+1}: objective = {obj_val_raw:.6f}, current best = {best_val_raw:.6f}")
        else:
            print(f"Cycle {t+1}: objective = inf (invalid or out of bounds)")

        # --- Update best found if improved ---
        if obj_val_raw < best_val_raw:
            print(f"New best found: {obj_val_raw} at ({px}, {py})")
            best_val_raw = obj_val_raw
            best_coord = (px, py)
            no_improvement_count = 0
        else:
            print("No improvement this iteration.")
            no_improvement_count += 1
            if no_improvement_count >= convergence_patience:
                print(f"\nConvergence reached after {t+1} cycles. Stopping.")
                break

        # --- Scale and append data for next iteration ---
        obj_val_scaled = scale_value(obj_val_raw)
        if obj_val_scaled == np.inf:
            obj_val_scaled = 1.1

        new_bits = read_grid.coord_bits(px, py, x_bound, y_bound)
        xs.append(new_bits)
        x_vectors.append([int(bit) for bit in new_bits])
        ys.append(obj_val_scaled)

        history.append(best_val_raw)
        print(f"No improvement for {no_improvement_count} consecutive cycles.")
        print(f"Evaluated so far: {len(evaluated_points)} / total {len(grid)}")

    if t == max_cycles - 1:
        print(f"\nMax cycles reached without convergence.")

    return final_model, best_val_raw, best_coord, loop_records, evaluated_points, history, no_improvement_count, t

def output_results(graphtype, final_model, best_val_raw, best_coord, obj_min, grid, loop_records, start_time, evaluated_points, no_improvement_count, x_bound, y_bound):
    """Output the results: final report, CSV, and visualization."""
    # --- Final Report ---
    print(f"\n--- Final Results ---")
    print(f"Best objective found by FMQA = {best_val_raw} at {best_coord}")
    print(f"The known global minimum for the entire dataset is = {obj_min}")
    true_best_coord = min(grid, key=grid.get)
    print(f"Global minimum is at {true_best_coord} with objective {grid[true_best_coord]}")

    if final_model:
        print_final_equation(final_model)

    found_optimal = best_val_raw <= obj_min

    # WRITE CSV LOG (UNIQUE FILE)
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join("output", f"{graphtype}_SA_log_{timestamp}.csv")

    if loop_records:
        fieldnames = [
            "cycle",
            "px",
            "py",
            "objective",
            "best_val",
            "evaluated_count",
            "no_improvement_count",
            "status",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in loop_records:
                writer.writerow(rec)

            runtime_seconds = time.perf_counter() - start_time
            writer.writerow({
                "cycle": "runtime_sec",
                "px": "",
                "py": "",
                "objective": runtime_seconds,
                "best_val": "",
                "evaluated_count": "",
                "no_improvement_count": "",
                "status": "runtime_seconds",
            })
            writer.writerow({
                "cycle": "found_optimal",
                "px": "",
                "py": "",
                "objective": found_optimal,
                "best_val": best_val_raw,
                "evaluated_count": len(evaluated_points),
                "no_improvement_count": no_improvement_count,
                "status": "summary",
            })

        print(f"\nFMQA loop log written to: {csv_path}")
    else:
        print("\nNo loop records to write.")
        
    # --- Visualization ---
    x_vals = np.arange(x_bound + 1)
    y_vals = np.arange(y_bound + 1)
    Z = np.full((y_bound + 1, x_bound + 1), np.nan)

    for (x, y), v in grid.items():
        Z[y, x] = v

    plt.figure(figsize=(8, 6))
    plt.title("FMQA Optimization")

    # Heatmap of the objective values
    im = plt.imshow(Z, origin="lower", cmap="cividis", aspect="auto")
    plt.colorbar(im, label="Objective Value")

    # Plot evaluated points as numbered labels (1 = first evaluated)
    eval_sequence = [
        (rec["px"], rec["py"]) for rec in loop_records
        if isinstance(rec.get("cycle"), int) and rec.get("status") in {"initial_random", "evaluated"}
    ]
    for idx, (x, y) in enumerate(eval_sequence, start=1):
        plt.text(x, y, str(idx), color="white", fontsize=7, ha="center", va="center")

    # Plot best found and true global minimum
    if best_coord is not None:
        plt.scatter(*best_coord, color="red", s=120, marker="*", label=f"FMQA Best")
    true_best_coord = min(grid, key=grid.get)
    plt.scatter(*true_best_coord, color="cyan", s=120, marker="*", label=f"Global Optimal")

    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Save the generated plot to ./figures_output
    os.makedirs("figures_output", exist_ok=True)
    fig_save_path = os.path.join("figures_output", f"{graphtype}_plot_{timestamp}.png")
    plt.gcf().savefig(fig_save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {fig_save_path}")

def main():
    # --- Blackbox data (no train/test split) ---
    all_feasible_points = [p for p, v in grid.items() if not np.isnan(v)]
    # random.seed(0)
    random.shuffle(all_feasible_points)
    convergence_patience = max(1, int(len(all_feasible_points) * 0.03)) # 3% by default
    max_cycles = 150    # 150 max cycles by default

    print(f"\nTotal feasible points: {len(all_feasible_points)}")
    print(f"\nConvergence patience set to {convergence_patience} ")

    # Run FMQA
    final_model, best_val_raw, best_coord, loop_records, evaluated_points, history, no_improvement_count, t = run_fmqa(
        grid, x_bound, y_bound, max_cycles, convergence_patience, sampler, all_feasible_points
    )

    # Output results
    output_results(graphtype, final_model, best_val_raw, best_coord, obj_min, grid, loop_records, start_time, evaluated_points, no_improvement_count, x_bound, y_bound)

if __name__ == "__main__":
    main()


