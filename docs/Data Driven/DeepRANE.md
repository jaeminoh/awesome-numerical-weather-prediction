---
title: DeepRANE
---

- Paper title: The deep learning model for heavy rainfall nowcasting in South Korea[@oh2024deep]
- Journal: Weather and Climate Extremes

## Data

- [Radar reflectivity and AWS](https://data.kma.go.kr/cmmn/main.do)
- [KLAPS prediction](https://www.data.go.kr/)
- [Model GitHub](https://github.com/jihoonko/DeepRaNE)

## Motivation

Prediction of precipitation is a challenging problem, as it may happen locally.
For example, precipitation from late morning to early evening is mainly due to the convective process from surface heating.
Locality means computer simulation must have the capability to capture small-scale behaviors.

“Classical” physics-based computer simulations are often computationally infeasible to capture such local behavior, due to the curse of dimensionality.
If we introduce $N$ points per coordinate, the 3d domain requires $N^3$ grid points.
Solving a linear system $Ax = b$ requires approximately $(N^3)^3 = N^9$ floating point operations.


## Model

This article presents a simple but effective deep-learning model to predict the probability distribution over three precipitation classes.
The model has a U-Net architecture.
The model takes seven radar reflectivity images, maybe a combination of radar images and ground measurements, (t, …, t-60mins) as inputs.
During the pre-training process, the model outputs are radar reflectivity images for the near future.
During the fine-tuning process, the model outputs are probability distribution over three precipitation classes (less than 1mm/h, more than 10mm/h, and between), and the model is optimized on a specially designed loss function for CSI score.

The authors compare the model to KLAPS, a numerical prediction model for precipitation.
(Of course) The performance metrics, the F1 and CSI scores, are better for the proposed model.
The authors explain that the improved performance comes from more accurate prediction of precipitation time and the capability of capturing the convective process during the daytime.


## Conclusion

The proposed model showed impressive performance in precipitation prediction.
However, it is impossible to validate the model with physical laws (the primitive equation), as the model cannot predict humidity.