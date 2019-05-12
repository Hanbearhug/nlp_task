import tensorflow as tf

class Model:
    def __init__(self, hidden_size, pretrain_embedding=None, embed_size=64, vocab_size=10000):
        self.hidden_size = hidden_size
        if pretrain_embedding:
            self.pretrain_embedding = tf.Variable(pretrain_embedding, trainable=False)
        else:
            self.pretrain_embedding = tf.Variable(
                tf.random_uniform([vocab_size, embed_size], -1.0, 1.0), name='pretrain_embedding',
                dtype=tf.float32, trainable=True)

    def forward(self, cell_type='rnn'):
        """
        构建一个前向静态图，首先尝试一般rnn
        :return:
        """
        if cell_type=='rnn':
            sent = tf.placeholder(tf.int32, shape=(None, None), name='sent')
            sent_embed = tf.nn.embedding_lookup(self.pretrain_embedding, sent, name='sent_embed')

            dynamic_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)
            batch_size = tf.shape(sent_embed)[0]
            initial_state = dynamic_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            output, last_state = tf.nn.dynamic_rnn(
                cell=dynamic_cell,
                inputs=sent_embed,
                initial_state=initial_state,
                dtype=tf.float32
            )

            logits = tf.layers.dense(last_state, units=2,
                                     kernel_initializer=tf.initializers.glorot_uniform(), name='logits')
        return logits

    def backword(self, logits):
        """
        构建静态图
        :return:
        """
        y_true = tf.placeholder(tf.int64, shape=(None,), name='y_true')
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true), name='loss')
        return loss




