# # Attention Mechanism

# 'Attention' 은
# 신경망들의 성능을 높이기 위한 메커니즘이자,
# 이제는 AI 분야에서 대세 모듈로서 사용되고 있는 '트랜스포머'의 기반이 되는 메커니즘입니다

# ## Seq2Seq 모델의 문제점
# + 문제점1 : 하나의 **고정 길이 벡터**에 모든 정보를 압축 → 정보 손실 발생
# + 문제점2 : RNN의 문제점인 기울기 소실 문제 (Vanishing Gradient)가 여전히 발생
#   + 기계번역 분야에서 입력 문장이 길면 번역 품질이 떨어지는 현상으로 나타남
# * seq2seq 문제 개선을 위해 **Attention** Mechanism 고안
#   - 입력 시퀀스가 길어지면 출력 시퀀스의 정확도가 떨어지는 것을 보정(개선)

# # Attention 의 아이디어
# * Attention Mechanism은 디코더가 출력단어를 예측하는 매 시점(time step)마다 인코더의 '전체 입력 문장'을 다시 한번 참조!
# * 이때 전체 입력 문장을 단순히 (동일 비율로) 참조하는 것이 아니라, **예측할 단어와 연관이 있는 단어를 집중(Attention)**해서 참조하게 된다

# # Attention Function
# 어텐션 함수
# 
# ![](https://wikidocs.net/images/page/22893/%EC%BF%BC%EB%A6%AC.PNG)
# 
# - 어텐션을 함수로 표현하면 주로 다음과 같이 표현됩니다
# 
#   - **Attention(Q, K, V) = Attention Value**
#   - Q: Query
#   - K: Key
#   - V: Value
# 
# - 어텐션 함수는 주어진 '쿼리(Query)'에 대해서 모든 '키(Key)'와의 **유사도**를 각각 구합니다.
# - 그리고 구해낸 이 **유사도**를 키와 맵핑되어있는 각각의 **'값(Value)'에 반영**해줍니다.
# - 그리고 유사도가 반영된 **'값(Value)'을 모두 더해서 리턴**합니다.
#   - 이를 (편의상) **어텐션 값(Attention Value)**이라고 하겠습니다.
# 
# - 지금부터 배우게 되는 Seq2Seq + 어텐션 모델에서 Q, K, V에 해당되는 각각의 Query, Keys, Values는 각각 다음과 같습니다.
# 
# ```
# Q = Query : t 시점(타임스텝)의 디코더 셀에서의 은닉 상태
# K = Keys : 모든 시점(타임스텝)의 인코더 셀의 은닉 상태들
# V = Values : 모든 시점(타임스텝)의 인코더 셀의 은닉 상태들
# ```

# # Attention Mechanism 과정
# * Attention Mechanism 중 가장 기초적이고 수식적으로 이해하기 쉬운 **dot-product attention** 을 통해 배워보겠습니다

# * Attention Mechanism 과정 (단계)
#   1. attention score 계산
#   1. 소프트맥스 함수를 통한 attention distribution 계산
#   1. 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값 계산
#   1. 어텐션 값과 디코더의 t 시점의 은닉 상태를 연결
#   1. 출력층 연산의 입력이 되는 $\tilde{s}_t$ 계산
#   1. $\tilde{s}_t$ 를 출력층의 입력으로 사용합니다

# [전체적인 개요(예시)]
# 
# ![](https://wikidocs.net/images/page/22893/dotproductattention1_final.PNG)
# 
# - ↑ 위 그림은 디코더의 '세번째 LSTM 셀'에서 출력 단어를 '예측'할 때,
# 어텐션 메커니즘을 사용하는 모습을 보여줍니다.
# 
# - 디코더의 첫번째, 두번째 LSTM 셀은 이미 어텐션 메커니즘을 통해 je와 suis를 예측하는 과정을 거쳤다고 가정합니다.
# 
# - 디코더의 세번째 LSTM 셀은 출력 단어를 예측하기 위해서 **인코더의 모든 입력 단어들의 정보를 다시 한번 참고**하고자 합니다.
# 
# - 중간 과정에 대한 설명은 현재는 생략, 주목할 것은 인코더의 **소프트맥스 함수** !!.
# 
# - 소프트맥스 함수를 통해 나온 결과값은 I, am, a, student 단어 각각이 '출력 단어를 예측'할 때 '얼마나 도움'이 되는지의 정도를 수치화한 값.
# 
# - 빨간 직사각형의 크기로 소프트맥스 함수의 결과값의 크기를 표현. 직사각형의 크기가 클 수록 도움이 되는 정도의 크기가 큽니다.
# 
# - 각 입력 단어가 디코더의 예측에 도움이 되는 정도가 수치화하여 측정되면 이를 **'하나의 정보'**로 담아서 디코더로 전송됩니다. 그림에서는 초록색 삼각형이 이에 해당됩니다.
# 
# - 결과적으로, 디코더는 출력 단어를 더 정확하게 예측할 확률이 높아진다.

# ## 1) Attention Score 계산

# ![](https://wikidocs.net/images/page/22893/dotproductattention2_final.PNG)
# 
# *   인코더의 각 시점의 은닉 상태는 $h_n$ 라고 합시다
# 
# *   디코더의 현재 시점 (timestep) t의 디코더 은닉 상태를 $s_t$ 라고 합시다
# 
# *   위 예제에선 인코더의 은닉 상태와 디코더의 은닉상태의 차원이 같다고 가정합니다.  위 그림에선 인코더의 은닉 상태와 디코더의 은닉 상태가 동일하게 차원이 4입니다.
# 
# * 이전의 seq2seq 에서 배웠던 내용!
#   - 현재 시점(timestep) t 에서 출력 단어를 예측하기 위해서 디코더의 셀은 두 개의 입력값을 필요로 했다
#     - 이전 시점 t-1의 은닉 상태
#     - 이전 시점 t-1에 나온 출력 단어.
# 
# * 그런데 어텐션 메커니즘에서는 출력 단어 예측에 또 다른 값을 필요하다.
#   - 바로 **어텐션 값(Attention Value)**이라는 새로운 값입니다.
#   - t번째 타임스텝의 단어를 예측하기 위한 어텐션 값을 $a_t$
# 이라고 정의하자.
# 
# * **어텐션 값** 이 현재 시점 t 에서의 출력 예측에 구체적으로 어떻게 반영되는지는 나중에 설명하고.
# 
# * 지금부터 배우는 모든 과정은 $a_t$ 를 구하기 위한 여정입니다. 그리고 그 여정의 첫 걸음은 바로 **어텐션 스코어(Attention Score)**를 구하는 일입니다.
# 
# * **어텐션 스코어**란 현재 디코더의 타임스텝 t에서 단어를 예측하기 위해, 인코더의 모든 은닉 상태 각각이 디코더의 현 시점의 은닉 상태 $s_t$ 와 얼마나 유사한지를 판단하는 스코어값입니다
# 
# *   여기서는 dot-product attention 에선 이렇게 계산한다
#   - '$s_t$ 를 전치(transpose)' 하고(결과 $s_t^T$) '각 은닉상태 $h_n$'와 내적(dot product)를 수행합니다.  즉 모든 어텐션 스코어 값은 '스칼라' 입니다.
#   - 예를 들어 $s_t$ 와 인코더의 $i$번째 은닉상태와 어텐션 스코어의 계산 방법은 아래와 같다.
#   - ![](https://wikidocs.net/images/page/22893/i%EB%B2%88%EC%A7%B8%EC%96%B4%ED%85%90%EC%85%98%EC%8A%A4%EC%BD%94%EC%96%B4_final.PNG)
#   - 이 '어텐션 스코어 함수'를 정의해보면 다음과 같다<br>
#     $score(s_t, h_i) = s_t^Th_i$
#   - $s_t$와 인코더의 모든 은닉 상태의 어텐션 스코어의 모음값을 $e^t$ 라고 정의하면 → attention scores = $e^t = [s_t^Th_1, s_t^Th_2, ... , s_t^Th_n]$

# ## 2) Attention Distribution 계산
# 소프트맥스 함수를 통한 attention distribution 계산

# ![](https://wikidocs.net/images/page/22893/dotproductattention3_final.PNG)
# 
# 
# 
# *   $e^t = [s_t^Th_1, s_t^Th_2, ... , s_t^Th_n$] 에 softmax를 적용해 '각각의 확률값'들을 계산
#   - 이 값들은 합하면 1이 되는 확률분포 다.
#   - 이를 **어텐션 분포(Attention Distribution)** 라고 하고
#   - 각각의 값은 **어텐션 가중치(Attention Weight)**  라고 한다
#   - 예를 들어 소프트맥스 함수를 적용하여 얻은 출력값인 I, am, a, student의 어텐션 가중치를 각각 0.1, 0.4, 0.1, 0.4라고 합시다. 이들의 합은 1입니다. 위의 그림은 각 인코더의 은닉 상태에서의 어텐션 가중치의 크기를 직사각형의 크기를 통해 시각화하였습니다. 즉, 어텐션 가중치가 클수록 직사각형이 큽니다.
# 
# 
# *   내적이 크다면 확률값이 높아질 것이고, 작다면 확률값이 낮아짐
# *   결국 확률값을 구하는 것은 예측할 단어와 연관이 있는 단어를 찾는 것
# *   디코더의 시점 t 의 가중치 모음값인 어텐션 분포를 $α^t$이라고 하면, 다음과 같은 식이 성립함
# $$ α^t = softmax(e^t) $$

# ## 3) Attention Value 계산
# 각 인코더의 어텐션 가중치와 은닉상태를 가중합 하여 계산

# ![](https://wikidocs.net/images/page/22893/dotproductattention4_final.PNG)
# 
# * 이제 지금까지 준비해온 정보들을 하나로 합치는 단계다.
# * 어텐션의 최종 결과값을 얻기 위해서 각 인코더의 은닉 상태와 어텐션 가중치들을 곱하고, 최종적으로 모두 더합니다.
#   - 즉! 가중합(Weighted Sum) 을 진행합니다
# *   어텐션의 최종 결과. 즉, 어텐션 함수의 출력값인 어텐션 값(Attention Value) **$a_t$**
# 에 대한 식을 보여줍니다.<br>
#   $a_t = \sum^N_{i=1}{\alpha^t_ih_i}$
# 
# * 이러한 어텐션 값 $a_t$은 종종 인코더의 문맥을 포함하고 있다고하여, **컨텍스트 벡터(context vector)**라고도 불립니다.
#   * 앞서 배운 가장 기본적인 seq2seq에서는 인코더의 마지막 은닉 상태를 컨텍스트 벡터라고 부르는 것과 대조됩니다.

# ## 4) Attention Value 와 디코더의 t 시점의 은닉상태 연결
# - concatenate

# ![](https://wikidocs.net/images/page/22893/dotproductattention5_final_final.PNG)
# 
# 
# *   어텐션 함수의 최종값인 어텐션 값 $a_t$를 구했다면
# *  $a_t$ 와 $s_t$와 연결(concatenate)하여 '예측' 연산에 사용할 하나의 벡터 $v_t$를 만들어 낸다.
# *   이 $v_t$는 기존과는 다르게 '인코더의 정보'를 가지고 있고
# * $v_t$ 가 $\hat y$ 예측연산의 입력으로 사용하게되므로 좀 더 좋은 성능의 예측을 수행할 수 있게 된다.  
# * 이것이 어텐션 메커니즘의 핵심이다!

# ## 5) 출력층 연산의 입력이 되는 $\tilde{s}_t$ 계산

# ![](https://wikidocs.net/images/page/22893/st.PNG)
# * 다음은 논문에서의 내용입니다
# * $v_t$ 를 바로 출력층으로 보내기 전에 신경망 연산을 한 번 더 추가하였습니다.
# * 가중치 행렬과 곱한 후에 하이퍼볼릭탄젠트($tanh$) 함수를 지나도록 하여 출력층 연산을 위한 새로운 벡터인 $\tilde{s}_t$
# 를 얻습니다.
# * 어텐션 메커니즘을 사용하지 않는 seq2seq에서는 출력층의 입력이 t시점의 은닉 상태인 $s_t$
# 였던 반면, 어텐션 메커니즘에서는 출력층의 입력이 $\tilde{s}_t$ 가 되는 셈입니다.
# 
# * 이를 식으로 표현하면 다음과 같습니다.
#   - $\tilde{s}_t = tanh(W_c[a_t ; s_t] + b_c)$
#     - $W_c$는 학습 가능한 가중치 행렬,
#     - $b_c$는 bias(편향, 절편)  (위 그림에선 편향은 생략)

# ## 6) $\tilde{s}_t$ 를 출력층의 입력으로 사용합니다

# $\tilde{s}_t$ 를 출력층의 입력으로 사용하여 예측 벡터를 얻습니다.
# 
# $\hat{y}_t = Softmax(W_y\tilde{s}_t + b_y)$

# # Attention Mechanism 종류
# 다양한 종류의 어텐션
# * attention mechanism에는 어텐션 스코어 계산 방식(어센텬 스코어 함수)의 차이에 따라 다양한 종류가 존재
# 
# 
# 이 름                 | 스 코 어&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|Defined By
# ----------------------|-----------------------|--
# dot|$score(s_t, h_i) = s_t^Th_i$|Luong et al. (2015)
# scaled dot|$score(s_t, h_i) = \frac{s_t^Th_i}{\sqrt n}$|Vaswani et al. (2017)
# general|$score(s_t, h_i) = s_t^TW_ah_i$|Luong et al. (2015)
# concat|$score(s_t, h_i) = W_a^Ttanh(W_b[s_t;h_i])$|Bahdanau et al. (2015)
# location - base|$a_t = softmax(W_as_t)$|Luong et al. (2015)
#   + $s_t$ : querys(t 시점에서의 디코더 셀의 은닉 상태)
#   + $h_i$ : keys(모든 시점의 인코더 셀 은닉 상태)
#   + $W_a, W_b$ : 학습 가능한 가중치 행렬

# # 바나다우 어텐션 (Banadau Attention)

# ## 1. 바다나우 어텐션 함수(Bahdanau Attention Function)
# 
# 어텐션 메커니즘을 함수 Attention()으로 정의하였을 때, 바다나우 어텐션 함수의 입, 출력은 다음과 같의 정의할 수 있습니다.
# <br>
# **<center>Attention(Q,K,V) = Attention Value</center>**
# <br>
# 
# ```
# t = 어텐션 메커니즘이 수행되는 디코더 셀의 현재 시점을 의미.
# 
# Q = Query : t-1 시점의 디코더 셀에서의 은닉 상태
# K = Keys : 모든 시점의 인코더 셀의 은닉 상태들
# V = Values : 모든 시점의 인코더 셀의 은닉 상태들
# ```
# 
# 여기서는 어텐션 함수의 Query가 디코더 셀의 t 시점의 은닉 상태가 아니라 **t-1 시점의 은닉 상태**임을 주의하셔야 합니다

# ## 1) 어텐션 스코어 (Attention Score) 구하기

# ![](https://wikidocs.net/images/page/73161/%EB%B0%94%EB%8B%A4%EB%82%98%EC%9A%B0%EC%96%B4%ED%85%90%EC%85%981.PNG)
# - 인코더의 시점(time step)을 각각 1, 2, ... N이라고 하였을 때 **인코더의 은닉 상태(hidden state)**를 각각 $h_1, h_2, ... h_N$
# 라고 합시다.
# 
# - 디코더의 현재 시점(time step) t에서의 **디코더의 은닉 상태(hidden state)**를 $s_t$
# 라고 합시다.
# 
# - 또한 여기서는 인코더의 은닉 상태와 디코더의 은닉 상태의 차원이 같다고 가정합니다. 위의 그림의 경우에는 인코더의 은닉 상태와 디코더의 은닉 상태가 동일하게 차원이 4입니다.
# 앞서 루옹 어텐션에서는 Query로 디코더의 t 시점의 은닉 상태를 사용한 것과는 달리 이번에는 t-1 시점의 은닉 상태 $s_{t-1}$
# 를 사용합니다.
# <br>
# 
# 바다나우 어텐션의 어텐션 스코어 함수. 즉,
# $s_{t-1}$ 과 인코더의 i번째 은닉 상태의 어텐션 스코어 계산 방법은 아래와 같습니다.
# 
# 
# <h3><center>$score(s_{t-1}, h_i) = W_a^Ttanh(W_{bS_{t-1}} + W_ch_i)$</center></h3>
# 
# 
# 단, $W_a, W_b, W_c$ 는 학습 가능한 가중치 행렬입니다.
# $s_{t-1}$와 $h_1,h_2,h_3,h_4$
# 의 어텐션 스코어를 각각 구해야하므로 병렬 연산을 위해 $h_1,h_2,h_3,h_4$
# 를 하나의 행렬 $H$
# 로 두겠습니다. 수식은 다음과 같이 변경됩니다.
# 
# <h3><center>$score(s_{t-1}, H) = W_a^Ttanh(W_{bS_{t-1}} + W_cH)$</center></h3>

# ↓ $W_bs_{t-1}$ 와 $W_cH$ 를 각각 구하면 다음과 같다
# 
# ![](https://wikidocs.net/images/page/73161/%EB%B0%94%EB%8B%A4%EB%82%98%EC%9A%B0%EC%96%B4%ED%85%90%EC%85%982.PNG)
# 
# 이들을 더한 후, 하이퍼볼릭탄젠트(tanh) 함수를 지나도록 합니다.
# 
# ![](https://wikidocs.net/images/page/73161/%EB%B0%94%EB%8B%A4%EB%82%98%EC%9A%B0%EC%96%B4%ED%85%90%EC%85%983.PNG)
# 
# 지금까지 진행된 연산의 수식은 다음과 같습니다.
# 
# $tanh(W_bs_{t-1} + W_cH)$
# 
# 이제 $W_a^T$와 곱하여 $s_{t-1}$ 와 $h_1, h_2, h_3, h_4$ 의 유사도가 기록된 **어텐션 스코어 벡터 $e^t$**를 얻습니다.
# 
# 
# ![](https://wikidocs.net/images/page/73161/%EB%B0%94%EB%8B%A4%EB%82%98%EC%9A%B0%EC%96%B4%ED%85%90%EC%85%984.PNG)
# 
# $e^t = W_a^Ttanh(W_bs_{t-1} + W_cH)$

# ## 2) 어텐션 분포 (Attention Distribution) 구하기

# 소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution)를 구한다.
# 
# ![](https://wikidocs.net/images/page/73161/%EC%96%B4%ED%85%90%EC%85%98%EB%94%94%EC%8A%A4%ED%8A%B8%EB%A6%AC%EB%B7%B0%EC%85%98.PNG)
# 
# $e^t$에 소프트맥스 함수를 적용하여, 모든 값을 합하면 1이 되는 확률 분포를 얻어냅니다. 이를 **어텐션 분포(Attention Distribution)**라고 하며, 각각의 값은 **어텐션 가중치(Attention Weight)**라고 합니다.

# ## 3) 어텐션 값 (Attention Value) 구하기

# 
# 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value)을 구한다
# 
# ![](https://wikidocs.net/images/page/73161/%EC%BB%A8%ED%85%8D%EC%8A%A4%ED%8A%B8%EB%B2%A1%ED%84%B0.PNG)
# 
# 
# - 지금까지 준비해온 정보들을 하나로 합치는 단계입니다.
# 
# - 어텐션의 최종 결과값을 얻기 위해서 각 인코더의 은닉 상태와 어텐션 가중치값들을 곱하고, 최종적으로 모두 더합니다.
# 
# - 요약하면 가중합(Weighted Sum)을 한다고 말할 수도 있겠습니다.
# 
# - 이 벡터는 인코더의 문맥을 포함하고 있다고하여, **컨텍스트 벡터(context vector)** 라고 부릅니다.

# ## 4) 현재 시점의 예측값 구하기

# 컨텍스트 벡터로부터 $s_t$ 를 구합니다.
# - 기존의 LSTM이 $s_t$ 를 구할 때 (상기!)
# 
#   - 기존의 LSTM은 이전 시점의 셀로부터 전달받은 은닉 상태 $s_{t-1}$와 현재 시점의 입력 $x_t$
# 를 가지고 연산하였습니다.
# 
#   - 아래의 LSTM은 seq2seq의 디코더이며 현재 시점의 입력 $x_t$ 는 임베딩된 단어 벡터입니다
# 
# ![](https://wikidocs.net/images/page/73161/LSTM.PNG)

# - 어텐션 메커니즘에서 $s_t$ 구하기
# 
#   - 아래의 그림은 바다나우 어텐션 메커니즘에서는 컨텍스트 벡터와 현재 시점의 입력인 단어의 임베딩 벡터를 연결(concatenate)하고, 현재 시점의 새로운 입력으로 사용하는 모습을 보여줍니다. 그리고 이전 시점의 셀로부터 전달받은 은닉 상태 $s_{t-1}$
# 와 현재 시점의 새로운 입력으로부터
# $s_t$를 구합니다.
# 
#   - 기존의 LSTM이 임베딩된 단어 벡터를 입력으로 했던것에 반해, **컨텍스트 벡터**와 **임베딩된 단어 벡터**를 **연결(concatenate)**하여 입력으로 사용하는 것이 달라졌습니다.
# 
# ![](https://wikidocs.net/images/page/73161/%EB%B0%94%EB%8B%A4%EB%82%98%EC%9A%B0%EC%96%B4%ED%85%90%EC%85%985.PNG)
# 
# 이후에는 어텐션 메커니즘을 사용하지 않는 경우와 동일합니다.
# 는 출력층으로 전달되어 현재 시점의 예측값을 구하게 됩니다.

# # Attention 구현 (bi-LSTM, IMDB 사용)

# ## 기본 import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras

import random
def set_seed(seed = 42):
  tf.keras.utils.set_random_seed(seed)
  tf.config.experimental.enable_op_determinism()

from keras.datasets import imdb
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# 사전크기
vocab_size = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
# >>> 출력:
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
# [1m17464789/17464789[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step

X_train.shape, X_test.shape
# >>> 출력:
# ((25000,), (25000,))

print(X_train[0])
# >>> 출력:
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]

# 이미 정수 인코딩 되어 있는 상태,  남은 전처리는 패딩

print('리뷰의 최대 길이: ', max(len(l) for l in X_train))
print('리뷰의 평균 길이: ', sum(map(len, X_train))/len(X_train))
# >>> 출력:
# 리뷰의 최대 길이:  2494
# 리뷰의 평균 길기:  238.71364

max_len = 500  # 평균 길이보다 조금 크게 패딩해볼거다.
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

X_train.shape, X_test.shape
# >>> 출력:
# ((25000, 500), (25000, 500))

# ## 바다나우 어텐션

# 1. 닷 프로덕트 어텐션 스코어 함수
#   - $score(query, key) = query^Tkey$
# 
# 2. 바다나우 어텐션 스코어 함수
#   - $score(query, key) = V^Ttahh(W_1key + W_2query)$

from keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from keras import Input, Model
from keras import optimizers

class BahdanauAttention(tf.keras.Model):

  def __init__(self, units):  # units 가 결국 가중치의 크기!
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, values, query): # 단, key 와 values 는 값은
    # score 계산시 뒤에서 할 덧셈을 위해서 차원 변경
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score 계산
    # score 의 shape => (batch, max_length, 1)
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention weights
    #   shape => (batch, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context vector
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)  # => (batch, hidden_size)

    return context_vector, attention_weights

# ## bi-LSTM + Attention

sequence_input = Input(shape=(max_len,), dtype='int32')
embedded_seqeunces = Embedding(vocab_size, 128, mask_zero=True)(sequence_input)

# 이번 예제에선 양방향 LSTM 층을 두개 쌓아봅니다.
lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))(embedded_seqeunces)

lstm, forward_h, forward_c, backward_h, backward_c =\
  Bidirectional(LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)

lstm.shape, forward_h.shape
# >>> 출력:
# ((None, 500, 128), (None, 64))

# bi-LSTM 의 hidden state 와 cell state 를 사용하려면
# 두 방향의 LSTM 의 상태들을 연결 (concatenate) 해주어야 한다

state_h = Concatenate()([forward_h, backward_h])  # hidden state
state_c = Concatenate()([forward_c, backward_c])  # cell state

state_h.shape, state_c.shape
# >>> 출력:
# ((None, 128), (None, 128))

attention = BahdanauAttention(64)  # 가중치 크기 정의
context_vector, attetion_weights = attention(lstm, state_h)   # 어텐션 메커니즘의 입력은 hidden state
# >>> 출력:
# /usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py:938: UserWarning: Layer 'bahdanau_attention' (of type BahdanauAttention) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
#   warnings.warn(

context_vector.shape
# >>> 출력:
# (None, 128)

dense1 = Dense(20, activation='relu')(context_vector)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation='sigmoid')(dropout)

model = Model(inputs=sequence_input, outputs=output)

# ## 학습

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=3, batch_size=246, validation_data=(X_test, y_test))
# >>> 출력:
# Epoch 1/3
# /usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py:938: UserWarning: Layer 'bahdanau_attention' (of type BahdanauAttention) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
#   warnings.warn(
# [1m102/102[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m343s[0m 3s/step - accuracy: 0.6358 - loss: 0.6131 - val_accuracy: 0.8728 - val_loss: 0.3005
# Epoch 2/3
# [1m102/102[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m321s[0m 3s/step - accuracy: 0.8881 - loss: 0.2980 - val_accuracy: 0.8860 - val_loss: 0.2840
# Epoch 3/3
# [1m102/102[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m324s[0m 3s/step - accuracy: 0.9280 - loss: 0.2076 - val_accuracy: 0.8762 - val_loss: 0.3261
# <keras.src.callbacks.history.History at 0x78bcc69e7990>

# 평가 결과
model.evaluate(X_test, y_test)[1]
# >>> 출력:
# [1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m137s[0m 175ms/step - accuracy: 0.8765 - loss: 0.3291
# 0.8762400150299072

# ## 예측하기

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
# >>> 출력:
# [nltk_data] Downloading package punkt_tab to /root/nltk_data...
# [nltk_data]   Unzipping tokenizers/punkt_tab.zip.
# True

# word to index
word_index = imdb.get_word_index()
# >>> 출력:
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
# [1m1641221/1641221[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step

# index to word
reverse_word_index = {value: key for key, value in word_index.items()}

# 문장이 주어지면 예측하는 함수
def predict_review(review):
  # 리뷰할 문장을 토큰화 하고 인덱스로 변환.  사전에 없는 단어는 OOV (0)
  sample_review_index = [word_index.get(word, 0) for word in word_tokenize(review.lower())]

  # 사전크기(vocab_size) 넘어가는 것 은 OOV 처리
  sample_review_index = [(idx if idx <= vocab_size else 0) for idx in sample_review_index]

  # 패딩
  sample_review_index_padded = pad_sequences([sample_review_index], maxlen=max_len)

  # 예측
  prediction = model.predict(sample_review_index_padded)
  print(f'Prediction: {prediction[0][0]}')

sentences = [
    'movie is very good',
    'The best documentary I have watched in a very long time. This is definitely a must see for everyone. This family and their love and support for each other is truly amazing.',
    'movie is terrible',
    "This is a very one sided documentary about a woman who is sentenced to a 15 year mandatory prison sentence for dealing drugs. The documentary is made by her family and they want you to believe she doesn't deserve her sentence. However if you read the court reports you will see that she was a drug dealer, she lied to the police, and she was found guily after a trial. The tragic part of this documentary is that the woman left behind a husband and three little girls who will forever be damaged by not having their mother around. Yes you will feel sorry for the children because none of this was their fault. However their drug dealing mom got the sentence she deserved and now she has to spend the rest of her life making it up to these kids. The big lesson of this documentary is if you committ a crime in the US you will get locked up!",
    "Absolutely awful. The special effects were laughable, and the ending was so disappointing.",
    "I regret watching this. It's one of the worst movies I've ever seen, and I wouldn't recommend it to anyone."
]

for sentence in sentences:
  print(sentence)
  predict_review(sentence)
  print()
# >>> 출력:
# movie is very good
# /usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py:938: UserWarning: Layer 'bahdanau_attention' (of type BahdanauAttention) was passed an input with a mask attached to it. However, this layer does not support masking and will therefore destroy the mask information. Downstream layers will not see the mask.
#   warnings.warn(
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 991ms/step
# Prediction: 0.49900180101394653
# 
# The best documentary I have watched in a very long time. This is definitely a must see for everyone. This family and their love and support for each other is truly amazing.
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 140ms/step
# Prediction: 0.9654848575592041
# 
# movie is terrible
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 132ms/step
# Prediction: 0.49690112471580505
# 
# This is a very one sided documentary about a woman who is sentenced to a 15 year mandatory prison sentence for dealing drugs. The documentary is made by her family and they want you to believe she doesn't deserve her sentence. However if you read the court reports you will see that she was a drug dealer, she lied to the police, and she was found guily after a trial. The tragic part of this documentary is that the woman left behind a husband and three little girls who will forever be damaged by not having their mother around. Yes you will feel sorry for the children because none of this was their fault. However their drug dealing mom got the sentence she deserved and now she has to spend the rest of her life making it up to these kids. The big lesson of this documentary is if you committ a crime in the US you will get locked up!
# [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 131ms/step
# Prediction: 0.7508363723754883
# 
