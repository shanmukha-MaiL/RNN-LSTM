import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn,rnn_cell

mnist = input_data.read_data_sets('/home/shanmukha/AnacondaProjects/Spyder_projects/mnist',one_hot=True)

rnn_size = 128
chunk_size = 28
n_chunks = 28
n_classes = 10
batch_size = 128
n_epochs = 10

x = tf.placeholder('float',[None,n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    hidden_layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1,chunk_size])
    x = tf.split(x,n_chunks,axis=0)
    
    lstm = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs,state = rnn.static_rnn(lstm,x,dtype=tf.float32)
    
    output = tf.matmul(outputs[-1],hidden_layer['weights']) + hidden_layer['biases']
    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape(batch_size,n_chunks,chunk_size)
                j,c = sess.run([optimizer,loss],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c
            print('Epoch ',epoch,' completed.Epoch loss = ',epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ',accuracy.eval({x:mnist.test.images.reshape(-1,n_chunks,chunk_size),y:mnist.test.labels}))
        
train_neural_network(x)        