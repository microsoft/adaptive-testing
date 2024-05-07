import argparse
import logging
import pathlib
import subprocess
import sys

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def build_argument_parser():
    desc = "Install Adaptive Testing from a wheel file"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--wheel-dir",
        help="Directory containing the AdaTest wheel",
        required=True,
    )

    return parser


def main(argv):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    _logger.info("Finding wheel file")
    target_dir = pathlib.Path(args.wheel_dir)
    # Globbing works from Python, but not in Windows builds
    wheel_list = list(target_dir.glob("adaptivetesting*.whl"))
    assert len(wheel_list) == 1, f"Bad wheel_list: {wheel_list}"
    wheel_path = wheel_list[0].resolve()
    msg = f"Path to wheel: {wheel_path}"
    _logger.info(msg)

    _logger.info("Installing wheel")
    # Use this approach so that extras can be added
    adatest_spec = f"adaptivetesting[dev] @ {wheel_path.as_uri()}"
    subprocess.run(["pip", "install", f"{adatest_spec}"], check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
