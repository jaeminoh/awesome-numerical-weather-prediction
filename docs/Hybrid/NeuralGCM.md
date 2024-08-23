---
title: Neural general circulation models (NeuralGCMs)
---

Google Research에서 개발한 numeric + ML hybrid 모델이다.
2024년 7월 22일 Nature[@kochkov2024neural]에 출판되었다.

들어가기 전에, [YouTube video](#resources)를 보고 전체적인 윤곽을 잡을 수 있다.


## Motivation

(Global) Numerical weather prediction은 지구의 날씨를 기술하는 편미분방정식을 풀어서 얻는다.
수치적으로 편미분방정식을 풀 때는, domain을 작은 cell들로 나누게 된다.
Cell의 크기가 크면 계산 비용이 저렴하지만 small scale process (강수, 구름)를 반영할 수 없고,
Cell의 크기가 작으면 계산 비용이 비싸다(1).
따라서 과학자들은 [parametrization](https://en.wikipedia.org/wiki/Parametrization_(climate_modeling))을 통해 계산 비용을 적절하게 유지하면서 작은 스케일 현상들을 반영하려고 노력해왔다.
하지만 이 때문에 numerical simulation에는 내재적인 정확성 한계가 있다고도 볼 수 있다.
{ .annotate }

1. Cell의 크기를 $O(1/N)$이라고 할 때, $O(N^3)$.

잘 맞는 physical parametrization을 찾는 것은 거의 수작업과 같다.
그렇다면 neural network와 data를 통해 더 좋은 parametrization을 찾는다면 어떨까?


## Model Description
모델에 대한 설명 전에 [Google Research Blog](#resources)를 읽어보면 좋다.

Neural GCM은 parametrization를 climate data로 학습된 neural network로 대체한 모델이다.
저자들은 neural network를 training하기 위해서 numerical solver를 JAX[@jax2018github]로 다시 작성했다.
덕분에 TPU, GPU에서도 빠르게 동작하는 numerical solver를 얻었다고 한다.

Neural GCM은 이 physical parametrization을 대체하는 learned physics 파트와, 편미분방정식을 푸는 dynamical core 파트, 그리고 좌표계를 변경해주는 encoder & decoder 파트로 이루어져 있다.

### Learned Physics
Learned physics를 도입해서 편미분방정식으로 잡을 수 없는 dynamics를 잡고자 했다.
Global circulation model은 spatial domain (여기서는 지구, $\Omega$)를 discretization 하고 나면 다음과 같은 system of ODEs로 쓸 수 있다.
$$
\frac{dX}{dt} = \Phi(X).
$$

여기서 neural network $\Psi_\theta$를 도입해 다음과 같은 neural global circulation model을 고려할 수 있다.
$$
\frac{d\tilde{X}}{dt} = \Phi(\tilde{X}) + \Psi_\theta(\tilde{X}).
$$
이제 우리에게 남은 일은 실제 기상 현상을 잘 맞추는 $\tilde{X}, \Psi_\theta$를 찾는 일이다.

NeuralGCM에서는 horizontal point마다 수직 방향으로 column을 하나씩 고려했다.
즉 $\Psi_\theta(\tilde{X})$가 cell 내부의 vertical한 상호작용을 담당했다는 뜻이다.
$\tilde{X}$는 prognostic variables $X$뿐 아니라 horizontal gradient와 추가적인 features가 포함되어 있다.

자세한 구조는 arXiv paper Appendix C에 설명되어 있다.


### Dynamical Core
Dynamical Core는 [primitive equation](https://en.wikipedia.org/wiki/Primitive_equations)을 수치적으로 푸는 부분이다.
arXiv paper Appendix B에 잘 설명되어 있는데, 여기서는 두가지 ($\sigma$ coordinate, 그리고 spherical harmonics)만 짚고 넘어가겠다.

**$\sigma$ coordinate.**
산과 같은 지형 때문에 수직 방향으로 computational domain을 쪼개는 것에는 어려움이 있다.
Horizontal location $x$에 따라 표면의 위치가 달라지기 때문이다.
Finite difference 혹은 finite element 방법은 적용가능하지만 spectral method는 사용하기 어렵다.
$\sigma$ coordinate은 수직 방향의 domain을 $[0, 1]$로 만드는 방법이다.
이를 적용하면 spectral method를 사용해 편미분방정식을 풀 수 있게 된다.

압력 $p$를 통해 수직 방향의 domain을 기술하는 것을 pressure coordinate(1)이라고 한다. 
하지만, $x$에 따라 압력이 다르기에 추가적인 작업이 필요하다.
(Terrain-following) $\sigma$ coordinate은 $\sigma = p / p_\mathrm{surf}$를 이용해 수직 방향의 domain을 기술한다.
표면은 1, 그리고 하늘 저 높은 곳 어디는 0이 된다.
{ .annotate }

1. 고도가 높아지면 일반적으로 pressure가 떨어진다.



**Spherical harmonics.**
Fourier basis $\{e^{ikx}\}$는 periodic domain에서 정의된 함수들로 이루어진 vector space의 orthogonal basis이다.
지구의 표면은 sphere인데, 이 위에서 편미분방정식을 spectral method로 풀 때는 spherical harmonics라는 orthogonal basis를 주로 사용한다.
Longitude 방향으로 equiangular points ($\because$ periodic),
latitude 방향으로 Gauss-Lobatto points를 사용했다고 한다.

모든 loss function들을 spherical harmonics로 표현된 coefficient에 대해서 계산했다고 한다.


### Encoder & Decoder
[ERA5 데이터](#resources)는 수직 방향으로 pressure coordinate을 사용한다.
하지만 GCM을 spectral method로 풀 때는 $\sigma$ coordinate을 사용한다.
Encoder와 Decoder는 이 두 좌표계 사이의 변환을 "부드럽게" 해 주기 위해 고안되었다.
Encoding은 pressure coordinate에서 $\sigma$ coordinate으로 변환을 의미하고,
Decoding은 그 반대의 과정을 의미한다.

사실 두 좌표계 사이 변환은 linear interpolation을 통해 손쉽게 할 수 있지만,
interpolation 이후에 learned correction을 더해 주는 것이 성능적으로 좋았다고 한다.
Learned correction이 없으면 initialization에서 shock이 발생하는데,
이 때문에 prediction이 high oscillation으로 오염된다고 한다.

arXiv paper Appendix D에 더 자세한 내용이 설명되어 있다.


## Resources

- [YouTube video](https://www.youtube.com/watch?v=KUAWw32FjIo)
- [Google Research Blog](https://research.google/blog/fast-accurate-climate-modeling-with-neuralgcm/)
- [arXiv paper](https://arxiv.org/abs/2311.07222)
- [GitHub repository](https://github.com/google-research/neuralgcm)
- [API documentation](https://neuralgcm.readthedocs.io/en/latest/index.html)
- [ERA5 data](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)