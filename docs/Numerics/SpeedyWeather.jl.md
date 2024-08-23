# SpeedyWeather.jl

Global circulation model을 시뮬레이션 할 수 있는 Julia package[@Klower2024SpeedyWeather.jl].

새로운 methods는 아쉽게도 없다.
기존에 존재하던 computer simulation 프로그램의 monolithic 디자인을 피하고,
Oceananigans.jl처럼 library 스타일의 interface를 갖췄다고 한다.
덕분에 빠른 prototyping이 가능할 듯 하다.

이미 널리 쓰이는 numerical methods를 차용했다.

- spherical harmonic transform (Sphere 위에서의 spectral method)
- Leapfrog based semi-implicit time stepping (Numerical stability + 계산 속도)
- Robert-Asselin-Williams filter (Gibbs effect 잡기)

## Resourcecs

- [GitHub](https://github.com/SpeedyWeather/SpeedyWeather.jl)
- [YouTube video | Atmospheric General Circulation Modelling with SpeedyWeather.jl | Milan Klöwer | JuliaEO24 ](https://www.youtube.com/watch?v=drZlYByYZ4g)
- [YouTube video | Atmospheric Modelling With SpeedyWeather.jl | Milan Klöwer | JuliaCon 2023](https://www.youtube.com/watch?v=qgmgg_Bzgyg)
- [YouTube video | A 16-bit Weather Model with Machine Learning | Milan Klöwer | JuliaCon 2022](https://www.youtube.com/watch?v=qba-7kdXiJg)
