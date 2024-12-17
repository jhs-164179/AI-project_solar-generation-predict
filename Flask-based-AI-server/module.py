import tensorflow as tf
from keras import layers


class moving_avg(layers.Layer):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = layers.AveragePooling1D(pool_size=kernel_size, strides=stride, padding='valid')

    def call(self, x):
        # 양쪽 끝에 패딩 추가 (시간 축 기준)
        front = tf.repeat(x[:, 0:1, :], repeats=(self.kernel_size - 1) // 2, axis=1)
        end = tf.repeat(x[:, -1:, :], repeats=(self.kernel_size - 1) // 2, axis=1)
        x = tf.concat([front, x, end], axis=1)

        # AveragePooling1D 적용
        x = self.avg(x)
        return x


class series_decomp(layers.Layer):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def call(self, x):
        moving_mean = self.moving_avg(x)  # 이동 평균을 계산
        res = x - moving_mean  # 잔차 (residual) 계산
        return res, moving_mean  # 잔차와 이동 평균 반환


class DLinear(layers.Layer):
    def __init__(self, seq_len, pred_len, c_in, return_sequences=False):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        self.channels = c_in
        self.linear_seasonal = [layers.Dense(self.pred_len) for _ in range(self.channels)]
        self.linear_trend = [layers.Dense(self.pred_len) for _ in range(self.channels)]
        self.return_sequences = return_sequences

    def call(self, x):
        seasonal_init, trend_init = self.decomposition(x)

        seasonal_outputs = []
        trend_outputs = []

        # 각 채널에 대해 Dense 레이어 적용
        for i in range(self.channels):
            output_s = self.linear_seasonal[i](seasonal_init[:, :, i])  # 각 채널에 대해 Dense 레이어 적용
            output_t = self.linear_trend[i](trend_init[:, :, i])  # 각 채널에 대해 Dense 레이어 적용
            seasonal_outputs.append(output_s)
            trend_outputs.append(output_t)

        seasonal_output = tf.stack(seasonal_outputs, axis=-1)  # 새로운 축으로 채널들을 쌓음
        trend_output = tf.stack(trend_outputs, axis=-1)  # 새로운 축으로 채널들을 쌓음
        x = seasonal_output + trend_output

        if not self.return_sequences:
            # x = x[:, -1, -1]
            x = x[:, :, -1]
        return x

    def get_config(self):
        config = super(DLinear, self).get_config()
        return config


class NLinear(layers.Layer):
    def __init__(self, c_in, output_dim, seq_len, pred_len, return_sequences=False):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = c_in

        # self.linear = []
        # for _ in range(self.channels):
        #     self.linear.append(layers.Dense(self.pred_len))
        self.linear = [layers.Dense(self.pred_len) for _ in range(self.channels)]

        self.return_sequences = return_sequences

    def call(self, x):
        seq_last = x[:, -1:, :]  # 마지막 타임스텝 값 저장
        x = x - seq_last  # 입력에서 마지막 값을 빼줌

        # 각 채널별로 Dense 적용 결과를 저장할 리스트
        outputs = []

        # 각 채널에 대해 Dense 레이어 적용
        for i in range(self.channels):
            output_i = self.linear[i](x[:, :, i])  # 각 채널에 대해 Dense 레이어 적용
            outputs.append(output_i)

        # 각 채널의 결과를 쌓아서 최종 출력 생성 (batch_size, pred_len, channels)
        output = tf.stack(outputs, axis=-1)  # 새로운 축으로 채널들을 쌓음

        # 마지막 타임스텝 값 더해줌
        x = output + seq_last

        if not self.return_sequences:
            # x = x[:, -1, -1]
            x = x[:, :, -1]

        return x

    def get_config(self):
        config = super(NLinear, self).get_config()
        return config


class FeedForward(layers.Layer):
    def __init__(self, output_dim, ratio):
        super(FeedForward, self).__init__()
        self.fc1 = layers.Dense(output_dim * ratio, activation='gelu')
        self.fc2 = layers.Dense(output_dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_config(self):
        config = super(FeedForward, self).get_config()
        return config