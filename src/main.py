import argparse
import logging
import sys

from autoda import RoutineExecutor

EXPERIMENT_NAME = 'Baseline'
DATASET_CONFIG = 'config_autoda.json'
CONFIG_FILE = './confs/my_conf.yaml'


def setup_logging() -> None:
    # Set the threshold logging level of the logger to INFO
    logging.root.setLevel(logging.INFO)
    # Create a stream-based handler that writes the log entries    #into the standard output stream
    handler = logging.StreamHandler(sys.stdout)
    # Create a formatter for the logs
    formatter = logging.Formatter('[%(levelname)s] (%(asctime)s %(name)s) %(message)s')
    # '%(created)f:%(levelname)s:%(name)s:%(module)s:%(message)s'
    #
    # Set the created formatter as the formatter of the handler
    handler.setFormatter(formatter)
    # Add the created handler to this logger
    logging.root.addHandler(handler)


if __name__ == '__main__':
    setup_logging()

    # Create argument parser
    parser = argparse.ArgumentParser(description='Parse augmentation operations from a file.')
    parser.add_argument('-n', '--name', type=str, default='Experiment',
                        help='Name of the experiment.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file.')

    # Parse command-line arguments
    args = parser.parse_args()

    RoutineExecutor(name=args.name, config_file=args.config).exec()
