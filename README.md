# depth_segmentation
This package provides geometric segmentation of depth images and an interface to semantic instance segmentation, where the output of a semantic instance segmentation of RGB images gets combined with the geometric instance segmentation. For the later case we assign each geometric segment a semantic label as well.
**TODO** Add image(s)

If you are interested in a global segmentation map, please also take a look at [voxblox-plusplus](https://github.com/ethz-asl/voxblox-plusplus).

## Installation
In your terminal, define the installed ROS version and name of the catkin workspace to use:
```bash
export ROS_VERSION=kinetic # (Ubuntu 16.04: kinetic, Ubuntu 18.04: melodic)
export CATKIN_WS=~/catkin_ws
```

If you don't have a [catkin](http://wiki.ros.org/catkin) workspace yet, create a new one:
```bash
mkdir -p $CATKIN_WS/src && cd $CATKIN_WS
catkin init
catkin config --extend /opt/ros/$ROS_VERSION --merge-devel 
catkin config --cmake-args -DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_TYPE=Release
wstool init src
```

**Note:** If you already have a catkin workspace, ensure that its devel space layout is merged. If you do `catkin config` in your workspace the output should include:
```
Devel Space Layout:          merged
```

Clone the `depth_segmentation` repository over HTTPS (no Github account required):
```bash
cd $CATKIN_WS/src
git clone --recurse-submodules https://github.com/ethz-asl/depth_segmentation.git
```

Alternatively, clone over SSH (Github account required):
```bash
cd $CATKIN_WS/src
git clone --recurse-submodules git@github.com:ethz-asl/depth_segmentation.git
```

Automatically fetch dependencies:
```bash
wstool merge -t . depth_segmentation/dependencies.rosinstall
wstool update
```

Build and source the packages:
```bash
catkin build depth_segmentation
source ../devel/setup.bash # (bash shell: ../devel/setup.bash,  zsh shell: ../devel/setup.zsh)
```

To compile it with Mask R-CNN support you'll need to set the `WITH_MASKRCNNROS` to `ON` in the `CMakeLists.txt` file:
```cmake
set(WITH_MASKRCNNROS ON)
```

## Usage
The two use cases can be started as described below.

### Geometric Segmentation
If you only want geometric segmentation, use:
```bash
roslaunch depth_segmentation depth_segmenatation.launch
```
You'll need to adjust the ros topic names in the `sensor_topics_file` (by default this is in `depth_segmentation//cfg/primesense_topics.yaml`). The depth segmentation parameters can be adjusted via dynamic reconfigure or in the `depth_segmentation_params_file` directly.

### Combined Geometric Segmentation with Semantics
To additionally run the semantic segmentation you can use this command:
```bash
roslaunch depth_segmentation semantic_instance_segmentation.launch
```
**NOTE** This only works if you have compiled `depth_segmentation` with Mask R-CNN enabled (`WITH_MASKRCNNROS=ON`).

## Citing
If you use this, please cite:
- Fadri Furrer, Tonci Novkovic, Marius Fehr, Abel Gawel, Margarita Grinvald, Torsten Sattler, Roland Siegwart, Juan Nieto, **Incremental Object Database: Building 3D Models from Multiple Partial Observations**, _IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_, 2018. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594391)] [[Video](https://www.youtube.com/watch?v=9_xg92qqw70)]

```bibtex
@INPROCEEDINGS{8594391, 
author={F. {Furrer} and T. {Novkovic} and M. {Fehr} and A. {Gawel} and M. {Grinvald} and T. {Sattler} and R. {Siegwart} and J. {Nieto}}, 
booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
title={Incremental Object Database: Building 3D Models from Multiple Partial Observations}, 
year={2018},
pages={6835-6842}, 
keywords={feature extraction;image colour analysis;image reconstruction;image representation;image segmentation;mobile agents;object detection;solid modelling;multiple partial observations;incremental object database;indoor scenes;merged models;object model;observed instances;segmented RGB-D images;global segmentation map;3D models;mobile agent;Image segmentation;Databases;Three-dimensional displays;GSM;Shape;Image reconstruction;Solid modeling}, 
doi={10.1109/IROS.2018.8594391}, 
ISSN={2153-0866}, 
month={Oct},}
```
If you also use the semantic segmentation, additionally cite:

- Margarita Grinvald, Fadri Furrer, Tonci Novkovic, Jen Jen Chung, Cesar Cadena, Roland Siegwart, Juan Nieto, **Volumetric Instance-Aware Semantic Mapping and 3D Object Discovery**, _IEEE Robotics and Automation Letters_, 2019. [[PDF](https://arxiv.org/abs/1903.00268)] [[Video](https://www.youtube.com/watch?v=Jvl42VJmYxg)]


```bibtex
@article{grinvald2019volumetric,
  title={{Volumetric Instance-Aware Semantic Mapping and 3D Object Discovery}},
  author={Grinvald, Margarita and Furrer, Fadri and Novkovic, Tonci and Chung, Jen Jen and Cadena, Cesar and Siegwart, Roland and Nieto, Juan},
  journal={IEEE Robotics and Automation Letters},
  year={2019},
  note={Accepted}
}
```
## License
The code is available under the [BSD-3-Clause license](https://github.com/ethz-asl/depth_segmentation/blob/master/LICENSE).
