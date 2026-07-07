import argparse

from src.evaluation.image.comparison import compare_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render reflected/non-reflected cached sample comparisons."
    )
    parser.add_argument("nonref_run_dir", type=str)
    parser.add_argument("reflected_run_dir", type=str)
    parser.add_argument("--plot-count", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--comparison-images-per-row", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.plot_count <= 0:
        raise ValueError("--plot-count must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.comparison_images_per_row <= 0:
        raise ValueError("--comparison-images-per-row must be positive.")
    group_size = args.comparison_images_per_row**2
    if args.plot_count % group_size != 0:
        raise ValueError(
            "--plot-count must be divisible by --comparison-images-per-row squared."
        )
    if args.top_k % group_size != 0:
        raise ValueError(
            "--top-k must be divisible by --comparison-images-per-row squared."
        )
    output_dir = compare_runs(
        nonref_run_dir=args.nonref_run_dir,
        reflected_run_dir=args.reflected_run_dir,
        plot_count=args.plot_count,
        top_k=args.top_k,
        comparison_images_per_row=args.comparison_images_per_row,
    )
    print(f"Wrote comparison output to {output_dir}")


if __name__ == "__main__":
    main()
