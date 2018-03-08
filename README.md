# dagnn
## DAGNN - Directed Acyclic Graph Neural Network

The project focuses on trying to design an artificial neural network architecture capable of learning and replicating a noisy 1D signal. The goal is to be able to replicate the behavior of the retina's neural layers using an artificial neural network.  

The motivation behind the project stems from the retina's neural layers, which have proven to be extremely effecient at processing visual input. As explained in [1], "in only three layers
of cells, the retina compresses the entire visual scene into the sparse responses of only a million output cells." Notwithstanding its deceptive simplicity, "the retina performs a wide range of nonlinear computations, including object motion detection, adaptation to complex spatiotemporal patterns, encoding spatial structure as spike latency."[1]   

## Prerequisites

Before starting the installation program, 

* For macOS, make sure you have Xcode installed. A copy can be obtained for free from the Mac App Store. If mex -setup returns errors, see troubleshooting.
* For Linux, make sure you have GCC 4.8 and LibJPEG are installed. To install LibJPEG in and Ubuntu/Debian-like distributions use: sudo apt-get install build-essential libjpeg-turbo8-dev For Fedora/Centos/RedHat-like distributions use instead: sudo yum install gcc gcc-c++ libjpeg-turbo-devel Older versions of GCC (e.g. 4.7) are not compatible with the C++ code in MatConvNet.
* For Windows, make sure you have Visual Studio 2015 or greater installed.

## Installation

To install, run 'setup.m' in the directory where you want to install the program. The output will let you know whether you will need any additional toolboxes.

## Execution

To execute the code, first add the path matconvnet/matlab using the command

	addpath matconvnet/matlab

To run the program, run the file DagNNNoisy.m using the following command

	DagNNNoisy.main()

## Parameters and Data

The paths for the parameters and data are specified as variables atop DagNNNoisy.m. The parameters can be changed by modifying the json files to be read, while the data can be modified by creating new data .mat files.

## Contributors

Yasin Zamani
Hong Xu
Neda Nategh

## Sources

1. McIntosh, Lane, and Niru Maheswaranathan. "A Deep Learning Model of the Retina." Trial 10 (2015): 20.