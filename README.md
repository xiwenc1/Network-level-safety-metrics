# Network-level-safety-metrics
Some codes to extract network-level metrics for the paper: "Network-level safety metrics for overall traffic safety assessment: A case study"
The paper is ready to submit.

if you find the paper useful, please kindly cite
```
@article{chen2022network,
  title={Network-level Safety Metrics for Overall Traffic Safety Assessment: A Case Study},
  author={Chen, Xiwen and Wang, Hao and Razi, Abolfazl and Russo, Brendan and Pacheco, Jason and Roberts, John and Wishart, Jeffrey and Head, Larry},
  journal={arXiv preprint arXiv:2201.13229},
  year={2022}
}
```


## Abstract
Driving safety analysis has recently witnesses unprecedented performance due to advances in computation frameworks, connected vehicle technology, new generation sensors, and artificial intelligence (AI). Particularly, the astonishing performance of deep learning (DL) methods realized higher levels of safety for autonomous vehicles and empowered volume imagery processing for driving safety analysis. 
An important application of DL methods is extracting driving safety metrics from traffic imagery. However, the majority of current methods use safety metrics for micro-scale analysis of individual crash incidents, which does not provide insightful guidelines for the overall network-level traffic management. On the other hand, large-scale safety assessment efforts mainly emphasize spatial and temporal distributions of crashes, while not always revealing the safety violations that cause accidents. To bridge these two perspectives, we define a new set of network-level safety metrics for the overall safety assessment of traffic flow by processing volume imagery taken by roadside infrastructures. An integrative analysis of the safety metrics and crash data reveals insightful temporal and spatial correlation between the representative network-level safety metrics and the crash frequency. The analysis is performed using two video cameras in the state of Arizona along with a 5-year crash report obtained from the Arizona Department of Transportation. The results confirm that network-level safety metrics can be used by the traffic management teams to equip traffic monitoring systems with advanced AI-based risk analysis, and timely traffic flow control decisions.

## Pipeline

![iamge](https://github.com/XiwenChen-Clemson/Network-level-safety-metrics/blob/main/images/pipeline.png)



## Perspective transformation
The segments and the keypoints are shown below.
![image](https://github.com/XiwenChen-Clemson/Network-level-safety-metrics/blob/main/images/segment_123456.png)


## Proposed Metrics
![iamge](https://github.com/XiwenChen-Clemson/Network-level-safety-metrics/blob/main/images/metrics_table.PNG)







## Some refs
YOLOv5+DeepSort is used to extracted the trajectories.
Our code is modified from this [repo](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
```
@misc{yolov5deepsort2020,
    title={Real-time multi-object tracker using YOLOv5 and deep sort},
    author={Mikel Broström},
    howpublished = {\url{https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch}},
    year={2020}
}
```
