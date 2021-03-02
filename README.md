# Abstract

- adversarial process(적대적인 과정)을 통해 Generative Model을 추정하는 프레임워크 제안
- Generative model G와 Discriminative model D 2개의 모델을 학습한다.
- G는 D가 구별하지 못하도록 Train Data의 분포를 근사한다.
- D는 실제 Train Data인지 G가 생성한 데이터인지 확률을 추정한다.

# Introduction

- Deep generative model은 최대 가능도 추정과 관련된 전략들에서 발생하는 많은 확률 연산들을 근사할 때의 어려움과 generative context에서는 기존 deep learning model의 큰 성공을 이끌었던 선형 piecewise linear units의 이점들을 가져오는 것의 어려움이 있어 큰 임팩트가 없었다. 이러한 어려움을 해결하기 위해 이 논문에서는 새로운 모델 추정 과정을 제안하였다.
- 이 논문에서 소개하는 adversarial nets framework에서 생성 모델은 sample이 생성 모델이 생성한 분포인지 진짜 데이터의 분포인지를 판별하는 discriminative model과 이에 맞서 싸우는 생성 모델로 이루어져있다.

# Adversarial nets

- Discriminator D는 실제 데이터가 들어왔을 경우 1(진짜), Generator로부터 생성된 데이터가 들어왔을 경우 0(가짜)라고 판별 해야한다. 즉 D는 진짜 데이터 x가 들어올 때만 1이 되도록 학습이 진행된다.(logD(x), log(1-D(G(z))를 maximize한다.) 이에 맞서 Generator G는 D가 1(진짜)로 판단하도록(D(G(z))가 1이 되도록) 학습이 진행된다(log(1 -D(G(z)))를 minimize 한다.).

![image/function.png](image/function.png)

- 학습을 진행하면서 D를 최적화하는것은 많은 계산이 필요하고 한정된 데이터셋에서 overfitting을 초래할 수 있다. 따라서 이 논문에서는 k step만큼 D를 최적화 하고 G는 1 step만큼 최적화 하도록 한다.
- 실제 위의 수식은 G가 학습하기에 충분한 기울기를 제공하지 않을 수 있다.학습 초기를 생각해보면 G는 말도안되는 데이터를 생성할것이기 때문에 D는 가짜 데이터를 쉽게 판별할 수 있다. 이 경우 log(1-D(G(z))의 gradient는 너무 작은 값이되어 학습이 잘 되지 않게된다. log(1-D(G(z))에서 G를 minimize하는것 보다 logD(G(z))를 maximize하는것으로 학습할 수 있다.

![image/traingraph](image/traingraph.png)

- 위의 그림은 논문에서 학습이 어떻게 진행 되는지를 보여주는 그림이다. 검은 점선은 real data distribution, 초록 실선은 generative distribution, 파란 점선은 discriminative distribution을 나타낸다. x와 z는 생성기가 noise z를 data space의 x로 mapping하는 것을 나타낸다.

    (a). 맨 처음 실제 데이터와 generator의 결과도 차이가나고 discriminator도 성능이 좋지않다.

    (b). G를 고정하고 D를 학습시켜 가짜 데이터를 구분할 수 있게한다.

     G를 고정한다면 D*(x)는 아래의 식으로 나타낼 수 있다.(미분해보면)

    ![image/fixG.png](image/fixG.png)

    (c). D를 고정하고 D의 gradient를 이용해 G를 학습시킨다.

    (d). (b),(c)를 반복해 generator는 discriminator가 구분할 수 없을정도의 데이터를 생성하고, discriminator는 둘을 구분할 수 없어 D(x) = 0.5가 된다
