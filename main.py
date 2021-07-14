"""Main module for running train and prediction pipelines"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

import torch

from src.train_pipeline import train_pipeline
from src.predict_pipeline import predict_pipeline

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CONFIG_PATH = "./configs/train_params.yml"
DEFAULT_PREDICT_CONFIG_PATH = "./configs/predict_params.yml"

def callback_train(arguments):
    """callback for train model"""
    train_pipeline(arguments.config_path)

def callback_predict(arguments):
    """callback for make prediction"""
    predict_pipeline(arguments.config_path)

def setup_parser(parser):
    """Setup CLI-parser"""
    subparsers = parser.add_subparsers(help="choose mode: train or predict")
    train_parser = subparsers.add_parser(
        "train",
        help="train model for facial landmarks recognition",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    predict_parser = subparsers.add_parser(
        "predict",
        help="predict landmarks on given dataset",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    train_parser.set_defaults(callback=callback_train)
    predict_parser.set_defaults(callback=callback_predict)

    train_parser.add_argument(
        "--config-path",
        default=DEFAULT_TRAIN_CONFIG_PATH,
        help="path to train config, default path is %(default)s"
    )

    predict_parser.add_argument(
        "--config-path",
        default=DEFAULT_PREDICT_CONFIG_PATH,
        help="path to predict config, default path is %(default)s"
    )


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = ArgumentParser(
        prog="facial-landmarks-recognition",
        description="tool for train CNN for facial landmarks recognition",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)

if __name__ == "__main__":
    main()