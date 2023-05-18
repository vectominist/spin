import argparse
import logging
import os
import sys

from src.data.librispeech import find_all_librispeech, save_data_info

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("prepare_data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="Root directory of LibriSpeech")
    parser.add_argument("json_dir", type=str, help="Directory to save .json files")
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        nargs="+",
        default=[
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
        ],
        help="LibriSpeech partitions to be processed",
    )
    parser.add_argument(
        "--sort-by-len", "-l", action="store_true", help="Sort audio files by length"
    )
    args = parser.parse_args()

    logger.info(f"Preparing data from LibriSpeech at {args.root}")
    logger.info(f"Splits: {args.split}")
    logger.info(f"Sort audio files by length = {args.sort_by_len}")
    for s in args.split:
        logger.info(f"Processing {s} split...")
        data = find_all_librispeech(os.path.join(args.root, s), args.sort_by_len)
        save_data_info(data, os.path.join(args.json_dir, s + ".json"))


if __name__ == "__main__":
    main()
