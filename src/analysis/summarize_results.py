from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS_DIR = ROOT / "outputs" / "experiments"

OUTPUT_DIR = EXPERIMENTS_DIR / "final_summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        print(f"WARNING: missing {path}")
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path, **kwargs):
    if not path.exists():
        print(f"WARNING: missing {path}")
        return None

    return pd.read_csv(path, **kwargs)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
        )

def main():

    print()
    print("================================")
    print("BUILDING FINAL EXPERIMENT SUMMARY")
    print("================================")
    print()

    comparison_path = (
        EXPERIMENTS_DIR
        / "ae_vs_vae"
        / "comparison.json"
    )

    comparison = load_json(
        comparison_path
    )

    if comparison is not None:

        save_json(
            comparison,
            OUTPUT_DIR / "model_comparison.json",
        )

        print(
            "Loaded AE/VAE reconstruction comparison."
        )

    probe_path = (
        EXPERIMENTS_DIR
        / "latent_probe_results.csv"
    )

    probe_df = load_csv(
        probe_path
    )

    latent_probe_summary = None

    if probe_df is not None:

        if "r2" in probe_df.columns:
            probe_df = probe_df.sort_values(
                "r2",
                ascending=False,
            )

        probe_df.to_csv(
            OUTPUT_DIR
            / "latent_probe_summary.csv",
            index=False,
        )

        latent_probe_summary = (
            probe_df.to_dict(
                orient="records"
            )
        )

        print(
            "Loaded latent probe results."
        )

    disentanglement_path = (
        EXPERIMENTS_DIR
        / "disentanglement.csv"
    )

    disentanglement_matrix_path = (
        EXPERIMENTS_DIR
        / "disentanglement_matrix.csv"
    )

    disentanglement_ratios_path = (
        EXPERIMENTS_DIR
        / "disentanglement_ratios.csv"
    )

    disentanglement_df = load_csv(
        disentanglement_path
    )

    disentanglement_matrix_df = load_csv(
        disentanglement_matrix_path,
        index_col=0,
    )

    disentanglement_ratios_df = load_csv(
        disentanglement_ratios_path
    )

    disentanglement_summary = None
    disentanglement_matrix_summary = None
    disentanglement_ratios_summary = None

    if disentanglement_df is not None:

        disentanglement_df.to_csv(
            OUTPUT_DIR
            / "disentanglement_summary.csv",
            index=False,
        )

        disentanglement_summary = (
            disentanglement_df.to_dict(
                orient="records"
            )
        )

        print(
            "Loaded disentanglement results."
        )

    if disentanglement_matrix_df is not None:

        disentanglement_matrix_df.to_csv(
            OUTPUT_DIR
            / "disentanglement_matrix.csv"
        )

        disentanglement_matrix_summary = (
            disentanglement_matrix_df.to_dict()
        )

        print(
            "Loaded disentanglement matrix."
        )

    if disentanglement_ratios_df is not None:

        disentanglement_ratios_df.to_csv(
            OUTPUT_DIR
            / "disentanglement_ratios.csv",
            index=False,
        )

        disentanglement_ratios_summary = (
            disentanglement_ratios_df.to_dict(
                orient="records"
            )
        )

        print(
            "Loaded disentanglement ratios."
        )
    sampling_runs = [
        {
            "latent_std": 1.0,
            "path": (
                EXPERIMENTS_DIR
                / "vae_samples"
                / "summary.json"
            ),
        },
        {
            "latent_std": 0.7,
            "path": (
                EXPERIMENTS_DIR
                / "vae_samples_std_0.7"
                / "summary.json"
            ),
        },
        {
            "latent_std": 0.5,
            "path": (
                EXPERIMENTS_DIR
                / "vae_samples_std_0.5"
                / "summary.json"
            ),
        },
    ]

    sampling_rows = []

    for run in sampling_runs:

        latent_std = run["latent_std"]
        path = run["path"]

        summary = load_json(
            path
        )

        if summary is None:
            continue

        row = {
            "latent_std": latent_std,
        }

        row.update(summary)

        sampling_rows.append(row)

    if sampling_rows:

        sampling_df = pd.DataFrame(
            sampling_rows
        )

        columns = list(
            sampling_df.columns
        )

        if "latent_std" in columns:
            columns.remove(
                "latent_std"
            )

            sampling_df = sampling_df[
                ["latent_std"] + columns
            ]

        sampling_df.to_csv(
            OUTPUT_DIR
            / "vae_sampling_comparison.csv",
            index=False,
        )

        print(
            "Loaded VAE sampling summaries."
        )

    optimization_path = (
        EXPERIMENTS_DIR
        / "shape_optimization"
        / "history.csv"
    )

    optimization_history = load_csv(
        optimization_path
    )

    optimization_summary = None

    if optimization_history is not None:

        optimization_history.to_csv(
            OUTPUT_DIR
            / "shape_optimization_history.csv",
            index=False,
        )

        first = optimization_history.iloc[0]
        last = optimization_history.iloc[-1]

        optimization_summary = {
            "num_history_rows":
                int(len(optimization_history)),
        }

        if "step" in optimization_history.columns:

            optimization_summary[
                "start_step"
            ] = int(first["step"])

            optimization_summary[
                "end_step"
            ] = int(last["step"])

        if (
            "predicted_base_radius"
            in optimization_history.columns
        ):

            start_base = float(
                first[
                    "predicted_base_radius"
                ]
            )

            end_base = float(
                last[
                    "predicted_base_radius"
                ]
            )

            optimization_summary[
                "start_predicted_base_radius"
            ] = start_base

            optimization_summary[
                "end_predicted_base_radius"
            ] = end_base

            optimization_summary[
                "base_radius_change"
            ] = (
                end_base
                - start_base
            )

        if (
            "predicted_ball_radius"
            in optimization_history.columns
        ):

            start_ball = float(
                first[
                    "predicted_ball_radius"
                ]
            )

            end_ball = float(
                last[
                    "predicted_ball_radius"
                ]
            )

            optimization_summary[
                "start_predicted_ball_radius"
            ] = start_ball

            optimization_summary[
                "end_predicted_ball_radius"
            ] = end_ball

            optimization_summary[
                "ball_radius_change"
            ] = (
                end_ball
                - start_ball
            )

        if (
            "normalized_base"
            in optimization_history.columns
        ):

            optimization_summary[
                "start_normalized_base"
            ] = float(
                first[
                    "normalized_base"
                ]
            )

            optimization_summary[
                "end_normalized_base"
            ] = float(
                last[
                    "normalized_base"
                ]
            )

        if (
            "normalized_ball"
            in optimization_history.columns
        ):

            optimization_summary[
                "start_normalized_ball"
            ] = float(
                first[
                    "normalized_ball"
                ]
            )

            optimization_summary[
                "end_normalized_ball"
            ] = float(
                last[
                    "normalized_ball"
                ]
            )

        if (
            "latent_distance"
            in optimization_history.columns
        ):

            optimization_summary[
                "start_latent_distance"
            ] = float(
                first[
                    "latent_distance"
                ]
            )

            optimization_summary[
                "final_latent_distance"
            ] = float(
                last[
                    "latent_distance"
                ]
            )

        if (
            "objective"
            in optimization_history.columns
        ):

            start_objective = float(
                first["objective"]
            )

            end_objective = float(
                last["objective"]
            )

            optimization_summary[
                "start_objective"
            ] = start_objective

            optimization_summary[
                "end_objective"
            ] = end_objective

            optimization_summary[
                "objective_change"
            ] = (
                end_objective
                - start_objective
            )

        if (
            "latent_bound_hits"
            in optimization_history.columns
        ):

            optimization_summary[
                "max_latent_bound_hits"
            ] = int(
                optimization_history[
                    "latent_bound_hits"
                ].max()
            )

            optimization_summary[
                "final_latent_bound_hits"
            ] = int(
                last[
                    "latent_bound_hits"
                ]
            )

        save_json(
            optimization_summary,
            OUTPUT_DIR
            / "shape_optimization_summary.json",
        )

        print(
            "Loaded shape optimization history."
        )

    reconstruction_csv_path = (
        EXPERIMENTS_DIR
        / "ae_vs_vae"
        / "reconstruction_metrics.csv"
    )

    reconstruction_df = load_csv(
        reconstruction_csv_path
    )

    reconstruction_metrics = None

    if reconstruction_df is not None:

        reconstruction_df.to_csv(
            OUTPUT_DIR
            / "model_reconstruction_metrics.csv",
            index=False,
        )

        reconstruction_metrics = (
            reconstruction_df.to_dict(
                orient="records"
            )
        )

        print(
            "Loaded reconstruction metrics CSV."
        )

    interpolation_dir = (
        EXPERIMENTS_DIR
        / "interpolation"
    )

    interpolation_summary = None

    if interpolation_dir.exists():

        interpolation_files = sorted(
            interpolation_dir.glob(
                "*.npy"
            )
        )

        interpolation_summary = {
            "num_interpolation_steps":
                len(interpolation_files),

            "files": [
                path.name
                for path
                in interpolation_files
            ],
        }

        print(
            f"Found "
            f"{len(interpolation_files)} "
            f"AE interpolation samples."
        )

    latent_edits_dir = (
        EXPERIMENTS_DIR
        / "latent_edits"
    )

    latent_edit_summary = None

    if latent_edits_dir.exists():

        parameter_directories = sorted(
            [
                path
                for path
                in latent_edits_dir.iterdir()
                if path.is_dir()
            ]
        )

        parameter_info = {}

        for parameter_dir in (
            parameter_directories
        ):

            voxel_files = sorted(
                parameter_dir.glob(
                    "*_voxels.npy"
                )
            )

            latent_files = sorted(
                parameter_dir.glob(
                    "*_latent.npy"
                )
            )

            direction_path = (
                parameter_dir
                / "direction.npy"
            )

            parameter_info[
                parameter_dir.name
            ] = {
                "num_voxel_edits":
                    len(voxel_files),

                "num_latent_edits":
                    len(latent_files),

                "has_direction":
                    direction_path.exists(),
            }

        manifest_path = (
            latent_edits_dir
            / "manifest.csv"
        )

        latent_edit_summary = {
            "parameters":
                parameter_info,

            "manifest_exists":
                manifest_path.exists(),
        }

        print(
            f"Found latent edits for "
            f"{len(parameter_info)} parameters."
        )

    final_summary = {
        "ae_vs_vae": comparison,

        "reconstruction_metrics":
            reconstruction_metrics,

        "latent_probe":
            latent_probe_summary,

        "disentanglement": {
            "raw_results":
                disentanglement_summary,

            "effect_matrix":
                disentanglement_matrix_summary,

            "ratios":
                disentanglement_ratios_summary,
        },

        "latent_edits":
            latent_edit_summary,

        "interpolation":
            interpolation_summary,

        "shape_optimization":
            optimization_summary,

        "vae_sampling":
            sampling_rows,
    }

    save_json(
        final_summary,
        OUTPUT_DIR
        / "results_summary.json",
    )

    overview_lines = []

    overview_lines.append(
        "3D TROPHY GENERATION - FINAL EXPERIMENT SUMMARY"
    )

    overview_lines.append(
        "=" * 52
    )

    overview_lines.append("")

    if comparison is not None:

        reconstruction = comparison.get(
            "reconstruction",
            {},
        )

        ae = reconstruction.get(
            "ae",
            {},
        )

        vae = reconstruction.get(
            "vae",
            {},
        )

        overview_lines.append(
            "AE VS VAE RECONSTRUCTION"
        )

        if ae:

            overview_lines.append(
                "AE:"
            )

            overview_lines.append(
                f"  BCE:  "
                f"{ae.get('bce', 'N/A')}"
            )

            overview_lines.append(
                f"  IoU:  "
                f"{ae.get('iou', 'N/A')}"
            )

            overview_lines.append(
                f"  Dice: "
                f"{ae.get('dice', 'N/A')}"
            )

        if vae:

            overview_lines.append(
                "VAE:"
            )

            overview_lines.append(
                f"  BCE:  "
                f"{vae.get('bce', 'N/A')}"
            )

            overview_lines.append(
                f"  IoU:  "
                f"{vae.get('iou', 'N/A')}"
            )

            overview_lines.append(
                f"  Dice: "
                f"{vae.get('dice', 'N/A')}"
            )

        overview_lines.append("")

    if probe_df is not None:

        overview_lines.append(
            "LATENT PROBE"
        )

        if (
            "parameter" in probe_df.columns
            and "r2" in probe_df.columns
        ):

            for _, row in (
                probe_df.iterrows()
            ):

                parameter = row[
                    "parameter"
                ]

                r2 = row[
                    "r2"
                ]

                if "mae" in probe_df.columns:
                    mae = row["mae"]

                    overview_lines.append(
                        f"  {parameter}: "
                        f"R2={r2:.4f}, "
                        f"MAE={mae:.4f}"
                    )

                else:

                    overview_lines.append(
                        f"  {parameter}: "
                        f"R2={r2:.4f}"
                    )

        overview_lines.append("")

    if sampling_rows:

        overview_lines.append(
            "VAE SAMPLING"
        )

        for row in sampling_rows:

            latent_std = row.get(
                "latent_std"
            )

            mean_fraction = row.get(
                "mean_largest_component_fraction"
            )

            fully_connected = row.get(
                "fully_connected_fraction"
            )

            over_95 = row.get(
                "over_95_percent_fraction"
            )

            overview_lines.append(
                f"  std={latent_std}:"
            )

            if mean_fraction is not None:

                overview_lines.append(
                    f"    mean largest component: "
                    f"{mean_fraction:.4f}"
                )

            if fully_connected is not None:

                overview_lines.append(
                    f"    fully connected: "
                    f"{100 * fully_connected:.1f}%"
                )

            if over_95 is not None:

                overview_lines.append(
                    f"    >95% largest component: "
                    f"{100 * over_95:.1f}%"
                )

        overview_lines.append("")

    if optimization_summary is not None:

        overview_lines.append(
            "SHAPE OPTIMIZATION"
        )

        if (
            "base_radius_change"
            in optimization_summary
        ):

            overview_lines.append(
                "  predicted base radius change: "
                f"{optimization_summary['base_radius_change']:.6f}"
            )

        if (
            "ball_radius_change"
            in optimization_summary
        ):

            overview_lines.append(
                "  predicted ball radius change: "
                f"{optimization_summary['ball_radius_change']:.6f}"
            )

        if (
            "start_objective"
            in optimization_summary
            and
            "end_objective"
            in optimization_summary
        ):

            overview_lines.append(
                "  objective: "
                f"{optimization_summary['start_objective']:.6f}"
                " -> "
                f"{optimization_summary['end_objective']:.6f}"
            )

        if (
            "final_latent_distance"
            in optimization_summary
        ):

            overview_lines.append(
                "  final latent distance: "
                f"{optimization_summary['final_latent_distance']:.6f}"
            )

        if (
            "max_latent_bound_hits"
            in optimization_summary
        ):

            overview_lines.append(
                "  max latent bound hits: "
                f"{optimization_summary['max_latent_bound_hits']}"
            )

        overview_lines.append("")

    overview_path = (
        OUTPUT_DIR
        / "results_overview.txt"
    )

    with open(
        overview_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                overview_lines
            )
        )

    print()
    print("================================")
    print("FINAL EXPERIMENT SUMMARY")
    print("================================")
    print()

    print(
        f"Saved to:\n{OUTPUT_DIR}"
    )

    print()

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):
        print(
            f"  {path.name}"
        )


if __name__ == "__main__":
    main()