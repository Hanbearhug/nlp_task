import tensorflow as tf

class Model:
    def __init__(self, hidden_size, pretrain_embedding=None, embed_size=16, vocab_size=10000):
        self.hidden_size = hidden_size
        self.train = True
        if pretrain_embedding:
            self.pretrain_embedding = tf.Variable(pretrain_embedding, trainable=False)
        else:
            self.pretrain_embedding = tf.Variable(
                tf.random_uniform([vocab_size, embed_size], -1.0, 1.0), name='pretrain_embedding',
                dtype=tf.float32, trainable=True)

    def forward(self, cell_type='rnn', model_type='rnn'):
        """
        构建一个前向静态图，首先尝试一般rnn
        :return:
        """
        if model_type == 'rnn':
            sent = tf.placeholder(tf.int32, shape=(None, None), name='sent')
            sent_lengths = tf.placeholder(tf.int32, shape=(None,), name='sent_lengths')

            sent_embed = tf.nn.embedding_lookup(self.pretrain_embedding, sent, name='sent_embed')
            if cell_type == 'rnn':
                cell_fw = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)

            elif cell_type == 'lstm':
                cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

            else:
                print('not finished!')
                exit(1)
            #batch_size = tf.shape(sent_embed)[0]
            #initial_state = dynamic_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            outputs, last_state = tf.nn.dynamic_rnn(
                cell=cell_fw,
                inputs=sent_embed,
                #initial_state=initial_state,
                dtype=tf.float32,
                sequence_length=sent_lengths,
                parallel_iterations=128
            )
            if cell_type == 'rnn':
                #final_output = tf.concat(outputs, 2)
                hidden1 = tf.keras.layers.GlobalMaxPool1D()(outputs)
                hidden2 = tf.keras.layers.GlobalAveragePooling1D()(outputs)
                hidden = tf.concat([hidden1, hidden2], axis=1)
            elif cell_type == 'lstm':
                hidden = tf.keras.layers.GlobalMaxPool1D()(outputs.h)
            else:
                print('not finished!')
                exit(1)

            fc1 = tf.layers.dense(hidden, units=32, activation=tf.nn.relu, kernel_initializer=tf.initializers.glorot_uniform())

            logits = tf.layers.dense(fc1, units=2,
                                     kernel_initializer=tf.initializers.glorot_uniform(), name='logits')

        elif model_type=='cnn':
            sent = tf.placeholder(tf.int32, shape=(None, 256), name='sent')
            sent_embed = tf.nn.embedding_lookup(self.pretrain_embedding, sent, name='sent_embed')
            conv1 = tf.layers.conv1d(sent_embed, filters=8, kernel_size=2, strides=1, activation=tf.nn.relu)
            conv2 = tf.layers.conv1d(sent_embed, filters=8, kernel_size=3, strides=1, activation=tf.nn.relu)
            conv3 = tf.layers.conv1d(sent_embed, filters=8, kernel_size=4, strides=1, activation=tf.nn.relu)

            pool1 = tf.keras.layers.GlobalMaxPool1D()(conv1)
            pool2 = tf.keras.layers.GlobalMaxPool1D()(conv2)
            pool3 = tf.keras.layers.GlobalMaxPool1D()(conv3)
            pool = tf.concat([pool1, pool2, pool3], 1)

            dropout = tf.layers.dropout(pool, training=self.train, rate=0.5)

            fc1 = tf.layers.dense(dropout, units=16, activation=tf.nn.relu, kernel_initializer=tf.initializers.glorot_uniform())

            logits = tf.layers.dense(fc1, units=2,
                                     kernel_initializer=tf.initializers.glorot_uniform(), name='logits')

        else:
            print('not finished!')
            exit(1)


        return logits

    def backword(self, logits):
        """
        构建静态图
        :return:
        """
        y_true = tf.placeholder(tf.int64, shape=(None,), name='y_true')
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true), name='loss')
        return loss




