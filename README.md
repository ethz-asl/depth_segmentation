# depth_segmentation
This package provides geometric segmentation of depth images and an interface to semantic instance segmentation, where the output of a semantic instance segmentation of RGB images gets combined with the geometric instance segmentation. For the later case we assign each geometric segment a semantic label as well.
**TODO** Add image(s)

## Installation
**TODO**

## Usage
The two use cases can be started as described below.

### Geometric Segmentation
**TODO**

### Combined Geometric Segmentation with Semantics
**TODO**

## Citing
If you use this, please cite:
- Fadri Furrer, Tonci Novkovic, Marius Fehr, Abel Gawel, Margarita Grinvald, Torsten Sattler, Roland Siegwart, Juan Nieto, **Incremental Object Database: Building 3D Models from Multiple Partial Observations**, _IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_, 2018. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594391)] [[Video](https://www.youtube.com/watch?v=9_xg92qqw70)]

```bibtex
@INPROCEEDINGS{8594391, 
author={F. {Furrer} and T. {Novkovic} and M. {Fehr} and A. {Gawel} and M. {Grinvald} and T. {Sattler} and R. {Siegwart} and J. {Nieto}}, 
booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
title={Incremental Object Database: Building 3D Models from Multiple Partial Observations}, 
year={2018}, 
volume={}, 
number={}, 
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
The code is available under the [BSD-3-Clause license](https://github.com/ethz-asl/voxblox-plusplus/blob/master/LICENSE).
