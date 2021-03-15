# 1. 코로나 확진자 예측<br/>
|Title|Contents|Explain|
|:----:|:----:|:----:|
|Project Name|covid19Pre.py|데이터를 바탕으로, 코로나 확진자를 예측|
|Data Name|corona_confirm.csv|본 프로젝트에 해당 url를 긁어옴으로 데이터를 다운로드 하도록 코딩해두었으나,<br/> 다운로드 시간 절약을 위해 본 레퍼짓토리에 수록|<br/>
<br/>

## Author<br/>
Ian/@Ian(aoa8538@gmail.com)<br/>
<br/>

## Requirements<br/>
- Pandas<br/>
- Numpy<br/>
- Plotly (데이터의 시각화를 위하여)<br/>
- Fbprophet (시계열 데이터를 다루기 위하여)<br/>
<br/>

## Explain
- 현재도 진행중인, 세계적으로 문제가 되고 있는 Covid-19에 대하여 확진자의 수는 줄어들지 혹은 늘어날지 궁금하여 프로젝트 진행
- 작년 한해를 기준으로 하였으므로, 데이터는 최신의 것이 아닌 상황
- 따라서, 올해 까지의 데이터를 추가하여 파일을 세팅하고 구동시키면 그에 맞게 구동이 되도록 설정되어있음<br/>
<br/>

# 2. 주가 예측<br/>
|Title|Contents|Explain|
|:----:|:----:|:----:|
|Project Name|LSTM_stockAMZN.py|2020년 한해, 아마존 주가 데이터를 기반으로<br/> 미래의 주가를 예측|
|Data Name|AMZN.csv|2020년 한해를 기준, 야후 파이낸스로부터 데이터를 다운받았음<br/> 더 긴 시간의 데이터를 개인적으로 다운받는다면<br/> 더욱 정확한 예측을 수행할 수 있도록 설계|<br/>
<br/>

## Author<br/>
Ian/@Ian(aoa8538@gmail.com)<br/>
<br/>

## Requirements<br/>
- Tensorflow 2.4.1<br/>
- Numpy<br/>
- Matplotlib<br/>
<br/>

## Explain
- 지금까지도, 많은 이들의 관심 메인이 되고 있는 주식에 대하여 생각해보았고
- 심지어, 예측을 기반으로 자동으로 매매해주는 프로그램까지 사용되고 있는 현재 상황에 대하여
- 그 핵심 기반인, 예측에 대하여 고민하고 코딩하게 되었음<br/>
<br/>

# 3. 선형회귀 예제의 대표주자, 당뇨병 예측
|Title|Contents|Explain|
|:----:|:----:|:----:|
|Project Name|diabetes.py|선형회귀를 공부한다면, 반드시 한번쯤은 다루고 가는 '당뇨병'의 데이터를 다루어 예측하는 프로그램<br/> 쉽게 설명해준 이가 없어, 직접 작성하여 왜 해당 코드가 작성 되었는지 쉽고 상세히 기술해놓았음|<br/>
<br/>

## From<br/>
- https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
- 해당 코드를 참조한, 본 원문에 대한 링크
- 작성된 주소의 링크에 접속하면, 어떻게 짜여있는지 알 수 있으나
- 이를 쉽게 한글 해석하여 해당 파일에 '해당 코드가 작성된 이유 및 작동방법 등' 상세히 기록해놓았음<br/>
<br/>

## Requirements<br/>
- Matplotlib<br/>
- Numpy<br/>
- Scikit-Learn<br/>
<br/>

# Motivation<br/>
본인 또한, 인공지능에 대하여 공부하고 많은 프로젝트를 경험하며 능력을 향상시키고 있는 가운데<br/>
아직까지도 많은 분들이 어떻게 공부해야 할지 모르고, 또한 어떻게 적절히 적용시켜야 하는가에 따라 어려움을 겪고 계시다는 것을 보게되었고<br/>
특히, 비전공자분들은 더욱 헷갈려 하시는 분야 중 하나로 꼽힌다는 것까지 알게 됨으로 이렇게 3개의 프로젝트를 개인적으로 시행해보았음.<br/>
어떻게 사용하던, 전혀 상관이 없으므로 학습에 참고해주신다면 매우 감사할 따름
