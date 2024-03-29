# Half-Half

Half & Half is a set of recently released tasks and benchmarks to analyze a visual system’s capability of performing common knowledge inferences. This entails predicting the part of the scene which is not observed directly. It is a very practical and interesting domain with many-many possible applications in areas like robotics, smart navigation, etc. The authors who propose these tasks did an amazing job of creating the datasets and setting up the benchmarks. We believe learning context between objects on a global level should improve the performances of all the three subtasks mentioned in the half&half paper which is challenging for commonly used neural network architectures such as Convolutional Neural Networks (CNNs). We intend to improve the performance on these tasks by critically examining various parts of the visual common sense pipeline, for example, incorporating state-of-the-art architectures and training methodologies that claim to learn visual context and relation between objects in images.

## Approach
Incorporated visual context using Glove-based Graph Convolutional Networks (GCN) and Graph-Based Global Reasoning Networks (GloRe) models in the mulit-label image classification pipeline to achieve significant gains.

Dataset (subset of MSCOCO) and baselines taken from: http://openaccess.thecvf.com/content_CVPRW_2019/papers/Vision_Meets_Cognition_Camera_Ready/Singh_HalfHalf_New_Tasks_and_Benchmarks_for_Studying_Visual_Common_Sense_CVPRW_2019_paper.pdf
