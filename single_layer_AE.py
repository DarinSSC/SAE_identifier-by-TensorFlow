import tensorflow as tf
import numpy
import data

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

train_set,eval_set = data.packets_data()
train_set_num = len(train_set.payloads)
batch_size = 100
sample_length = 28*28#the length of sample--[1,1000]

h1_num = 400 #the number of neurons in the 1st hidden layer

x = tf.placeholder("float",[None,sample_length])

W1 = weight_variable([sample_length,h1_num])
b1 = bias_variable([h1_num])

W1_reverse = weight_variable([h1_num,sample_length])
b1_reverse = bias_variable([sample_length])

h1_out = tf.nn.sigmoid(tf.matmul(x,W1) + b1)

keep_prob = tf.placeholder("float")
h1_out_drop = tf.nn.dropout(h1_out,keep_prob)

x_ = tf.nn.sigmoid(tf.matmul(h1_out_drop,W1_reverse) + b1_reverse)

loss = (tf.reduce_sum(tf.pow(x_-x , tf.to_float(tf.convert_to_tensor(2*numpy.ones([sample_length])))))/batch_size)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for i in xrange(len(train_set.payloads)/batch_size * 20):
		batch_x,batch_y = train_set.next_batch(batch_size)
		batch_x = numpy.reshape(batch_x,[batch_size,sample_length])
		
		sess.run(train_step,feed_dict={x:batch_x, keep_prob:1.0})
#		print sess.run(W1,feed_dict={x:batch_x, keep_prob:0.5})
		if i%10 == 0:
			print sess.run(loss,feed_dict={x:batch_x, keep_prob:1.0})
