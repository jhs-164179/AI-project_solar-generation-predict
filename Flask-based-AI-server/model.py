from keras import Model, Sequential, layers
from module import NLinear, DLinear, FeedForward


class HybridModel(Model):
    def __init__(self, in_shape, hid_dim, pred_len):
        super().__init__()

        self.pred_len = pred_len
        self.lstm = Sequential([
            layers.LSTM(hid_dim, return_sequences=True),
            layers.Dropout(.2),
            layers.LSTM(hid_dim//2, return_sequences=True)
        ])
        self.nlinear = Sequential([
            layers.Dense(hid_dim, activation='gelu'),
            NLinear(c_in=hid_dim, output_dim=None, seq_len=in_shape[0], pred_len=pred_len, return_sequences=True),
            layers.Dense(hid_dim//2, activation='gelu'),
            NLinear(c_in=hid_dim//2, output_dim=None, seq_len=in_shape[0], pred_len=pred_len, return_sequences=True)
        ])
        self.dlinear = Sequential([
            layers.Dense(hid_dim, activation='gelu'),
            DLinear(c_in=hid_dim, seq_len=in_shape[0], pred_len=pred_len, return_sequences=True),
            layers.Dense(hid_dim//2, activation='gelu'),
            DLinear(c_in=hid_dim//2, seq_len=in_shape[0], pred_len=pred_len, return_sequences=True)
        ])

        self.att = layers.MultiHeadAttention(num_heads=8, key_dim=hid_dim//2)
        self.ffn = FeedForward(hid_dim//2, ratio=2)
        self.out = Sequential([
            layers.Dense(hid_dim//2, activation='gelu'),
            layers.Dense(1, activation='linear')
        ])

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        # q
        nlienar_out = self.nlinear(x)
        # k
        dlinear_out = self.dlinear(x)
        # v
        x = self.lstm(x)[:, :self.pred_len, :]
        # self-attention
        att = self.att(nlienar_out, x, dlinear_out)
        x = self.norm1(x + att)
        ffn = self.ffn(x)
        x = self.norm2(x + ffn)
        # make final output
        x = self.out(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config