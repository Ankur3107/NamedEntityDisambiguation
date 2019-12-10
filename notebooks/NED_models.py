from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, concatenate, Input, Dropout, Bidirectional


def biLSTM_KGE_context_model(lstm_input_shape, graph_input_shape, dict_size):
    inp1 = Input(shape = (lstm_input_shape[1], ))
    inp2 = Input(shape = (graph_input_shape[1], ))
    embed = Embedding(output_dim=100, input_dim=dict_size, input_length=lstm_input_shape[1])(inp1)
    out1 = Bidirectional(LSTM(128))(embed)
    w = concatenate([out1, inp2])
    w = Dense(256, activation = 'relu')(w)
    w = Dropout(0.2)(w)
    out = Dense(1, activation = 'sigmoid')(w)
    model = Model(inputs=[inp1, inp2], outputs=out)
    return model