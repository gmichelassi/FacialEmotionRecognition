import sys

from typing import List
from video_capture.webcam import capture
from neural_network.data_analysis import data_analysis
from neural_network.training import train_network

commands = {
    'capture': capture,
    'train': train_network,
    'analysis': data_analysis
}


def main(arguments: List[str]):
    if len(arguments) == 0:
        return

    try:
        commands[arguments[1]]()
    except KeyError:
        print(f'Command not valid: {arguments[1]}')


if __name__ == '__main__':
    main(sys.argv)
