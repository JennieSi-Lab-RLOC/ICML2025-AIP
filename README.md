# Reinforcement Learning Control of a Physical Robot Device for Assisted Human Level Ground Walking without a Simulator (ICML2025-AIP)
Offline to online dHDP for soft exosuit personalized control

**[Reinforcement Learning Control of a Physical Robot Device for Assisted Human Level Ground Walking without a Simulator](https://icml.cc/virtual/2025/poster/43549)**

ICML | 2025

[Junmin Zhong](https://scholar.google.com/citations?user=uVv_eWQAAAAJ&hl=en&oi=ao)<sup>1</sup>, Emiliano Quinones Yumbla<sup>2</sup>, Seyed Yousef Soltanian<sup>2</sup>,   Ruofan Wu<sup>1</sup>, Wenlong Zhang<sup>2</sup><sup>3</sup>, Jennie Si<sup>1</sup><sup>3</sup>

<sup>1</sup>School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, Arizona. 

<sup>2</sup>School of Manufacturing Systems and Networks, Arizona State University, Mesa, Arizona.

<sup>3</sup>Corresponding Authors. Correspondence to: Jennie Si <si@asu.edu>, Wenlong Zhang <wenlong.zhang@asu.edu>.

## Installation

The code has been tested on Python 3.10.6 and PyTorch 2.0.1. 

See other required packages:
  1. Numpy


## Offline training


```Deploy_RL = 0``` for offline training, ```Deploy_RL = 1``` for validation

```data_csv``` is the path to the sensor state data, ```data_mocap_csv``` is the path to the MoCap ground truth data

## Online training

Interaction with bottom control files and sensor reading files.
