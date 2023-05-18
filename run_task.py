import argparse

from src import task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    args, _ = parser.parse_known_args()

    runner = getattr(task, args.task)()
    runner.run()


if __name__ == "__main__":
    main()
