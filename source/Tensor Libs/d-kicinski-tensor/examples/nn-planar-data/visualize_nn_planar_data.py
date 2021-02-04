#!/usr/bin/env python3

import argparse
import colorsys
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


@dataclass
class Colors:
    color1 = "#4c86a8"
    color2 = "#e0777d"
    color3 = "#e1dd8f"


@dataclass
class TestDataPath:
    input_data: str
    labels: str


def color(hex_color: str, scale_l: float = 1.0) -> Tuple[float, float, float]:
    rgb = matplotlib.colors.ColorConverter.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def visualize(train_data_path: str, test_data_path: Optional[TestDataPath] = None):
    color_map_background = ListedColormap([color(Colors.color1),
                                           color(Colors.color2),
                                           color(Colors.color3)])
    color_map = ListedColormap([color(Colors.color1, 0.7),
                                color(Colors.color2, 0.7),
                                color(Colors.color3, 0.7)])

    data_train = np.genfromtxt(train_data_path, delimiter="\t")
    inputs_train = data_train[:, :2]
    labels_train = data_train[:, 2:]

    if test_data_path:
        inputs_test = np.genfromtxt(test_data_path.input_data, delimiter="\t")[:, :2]
        labels_test = np.genfromtxt(test_data_path.labels, delimiter="\t")

        plt.scatter(inputs_test[:, 0], inputs_test[:, 1], c=labels_test, s=10,
                    cmap=color_map_background)

    plt.scatter(inputs_train[:, 0], inputs_train[:, 1], c=labels_train, s=90, cmap=color_map,
                edgecolors="k", linewidths=0.5)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training and test data.')
    parser.add_argument("--train_data", type=str, default="resources/train_planar_data.tsv")
    parser.add_argument("--test_data", type=str, default="resources/test_planar_data.tsv")
    parser.add_argument("--labels", type=str, default="resources/labels_planar_data.tsv")
    args = parser.parse_args()

    visualize(args.train_data,
              TestDataPath(args.test_data, args.labels)
              if args.test_data and os.path.exists(args.labels) else None)
