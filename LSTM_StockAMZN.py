import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

# Min Max 정규화 (0~1 값), (data - min) / (max - min)
# np.max(data,0) 명령에서 data는 array이며, 0은 axis=0를 뜻하고, 아래 방향 즉 컬럼방향으로 최대값을 찾는다는 의미이다.
# 즉 컬럼 별로 normalization이 이루어진다.

def MinMaxScaler(data):
    numerator = data - np.min(data, 0) # 데이터 모든 숫자들을 최소 값만큼 뺀다 (data - min).
    denominator = np.max(data, 0) - np.min(data, 0) # (max - min)
    # (data - min) / (max - min) 분모 0이 나오는걸 방지하기 위해 1e-7을 더한다.
    return numerator / (denominator + 1e-7)


# 데이터를 로딩한다.
# 일자, 시작가, 고가, 저가, 종가, 수정 종가, 거래량
xy = np.loadtxt("AMZN.csv", skiprows=1, usecols=(1,2,3,4,5,6), delimiter=",")
# 다운 받은 데이터를 엑셀에서 첫번째 행에 해당하는 제목과 열에 해당하는 날짜를 지운다.

xy = MinMaxScaler(xy)
print("xy[0][0]: ", xy[0][0])
x = xy # 입력 데이터
y = xy[:, [-2]] # 마지막 두번째 열이 정답(주식 종가)이다.
print("x[0]: ", x[0])
print("y[0]: ", y[0])

#하이퍼 파라미터
seq_length = 7 # 1개 시퀀스의 길이(시계열데이터 입력의 갯수), 즉 7일의 입력 데이터
data_dim = 6 # Variable 갯수, RNN 셀의 입력 크기 (시작가, 고가, 저가, 종가, 수정 종가, 거래량)
hidden_size = 10 # 각 RNN 셀의 출력 크기
num_classes = 1 # 결과 분류 총 수, 최종 출력 (RNN, Softmax 등) 클래스의 크기, 종가
learning_rate = 0.01 # 학습률, 학습 속도
epoch_num = 500 # 에폭 횟수(학습용 전체 데이터를 몇 회 반복해서 학습할지를 입력)

# 입력 데이터 배치 및 라벨 데이터 생성
dataX = [] # 입력 데이터 시퀀스 리스트
dataY = [] # 출력 라벨 데이터 시퀀스 리스트
for i in range(0, len(y) - seq_length):
    _x = x[i : i + seq_length] # 시퀀스 길이로 분할된 입력
    _y = y[i + seq_length] # 시퀀스 길이 후 종가(정답), 7일 후의 종가
    if i is 0:
        print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# 첫번째 시퀀스의 입력과 라벨의 출력
print("dataX[0] = ", dataX[0])
print("dataY[0] = ", dataY[0])

# 학습용/테스트용 데이터 분할 생성
train_size = int(len(dataY) * 0.7) # 70%를 학습용 데이터로 사용
test_size = len(dataY) - train_size # 나머지(30%)를 테스트용 데이터로 사용

# 데이터를 잘라 학습용 입력 데이터 생성
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

# 30% 데이터를 잘라 테스트용 데이터 생성
testX = np.array(dataX[train_size : len(dataX)])
testY = np.array(dataY[train_size : len(dataY)])

# 텐서플로우 플레이스홀더 생성
# 학습용/테스트용으로 X,Y 텐서를 생성한다
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, 1]) # 실제값 텐서
predictions = tf.placeholder(tf.float32, [None, 1]) # 예측값 텐서

# 평가용 RMSE(Root Mean Square Error) 텐서 생성
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

print("X: ", X)
print("Y: ", Y)
print("targets: ", targets)
print("predictions: ", predictions)
print("rmse: ", rmse)

# 모델(LSTM 네트워크) 생성 # activation=tf.sigmoid 또는 tf.tanh
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, activation=tf.tanh)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# 완전 연결 계층 구성
# hidden size = 10으로 설정하여 각 셀에서 10개의 출력값을 얻을 수 있지만, 
# 마지막 7번째 셀의 출력만을 fully connected 에 입력하여 하나의 출력 값 즉 그 다음 
# 종가를 학습하도록 한다.

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], num_classes, activation_fn=None
)

print("outputs: ", outputs)
print("states: ", states)
print("Y_pred: ", Y_pred)

loss = tf.reduce_sum(tf.square(Y_pred - Y)) # 에러의 제곱 합 손실 cost
train = tf.train.AdamOptimizer(learning_rate).minimize(loss) # Adam 최적화 알고리즘
print("loss = ", loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        if epoch % 100 == 0:
            print("[step: {} loss: {} ".format(epoch, step_loss))



    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})

    plt.plot(testY, "r")
    plt.plot(test_predict, "b")
    plt.xlabel("Time Period")
    plt.ylabel("Normalized Stock Price")
    plt.show()