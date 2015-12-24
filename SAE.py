import tensorflow as tf
import numpy
import data

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

start = 0
def next_batch(payloads,labels,batch_size):
	global start
	begin = start
	end = begin + batch_size
	if end > len(payloads):
		# Shuffle the data
		perm = numpy.arange(len(payloads))
		numpy.random.shuffle(perm)
		payloads = payloads[perm]
		labels = labels[perm]
        # Start next epoch
		start = 0
		begin = start
		end = batch_size
		assert batch_size <= len(payloads)
	start = end
	return payloads, labels, payloads[begin:end], labels[begin:end]

train_set,eval_set = data.packets_data()
train_set_num = len(train_set.payloads)
batch_size = 100
sample_length = 28*28#the length of sample--[1,1000]
train_payloads = train_set.payloads
train_labels = train_set.labels

epoches = len(train_set.payloads)/batch_size * 20

sess = tf.Session()

#************************* 1st hidden layer **************
x = tf.placeholder("float",[None,sample_length])
h1_num = 500 #the number of neurons in the 1st hidden layer

W1 = weight_variable([sample_length,h1_num])
b1 = bias_variable([h1_num])

W1_reverse = weight_variable([h1_num,sample_length])
b1_reverse = bias_variable([sample_length])

h1_out = tf.nn.sigmoid(tf.matmul(x,W1) + b1)

keep_prob = tf.placeholder("float")
h1_out_drop = tf.nn.dropout(h1_out,keep_prob)

x_1 = tf.nn.sigmoid(tf.matmul(h1_out_drop,W1_reverse) + b1_reverse)

loss1 = (tf.reduce_sum(tf.pow(x_1-x , tf.to_float(tf.convert_to_tensor(2*numpy.ones([sample_length])))))/batch_size)
train_step_1 = tf.train.GradientDescentOptimizer(0.01).minimize(loss1)

sess.run(tf.initialize_variables([W1,b1,W1_reverse,b1_reverse]))
for i in xrange(epoches):# training 
		batch_x,batch_y = train_set.next_batch(batch_size)	
		sess.run(train_step_1,feed_dict={x:batch_x, keep_prob:1.0})

h1_out = tf.nn.sigmoid(tf.matmul(tf.to_float(numpy.reshape(train_payloads, [len(train_payloads), sample_length])),W1) + b1)#get result of 1st layer as well as the input of next layer

#************************** 2nd hidden layer *************
h2_in = h1_out
h2_in = sess.run(h2_in)#change tensor to array

h2_num = 100 #the number of neurons in the 2nd hidden layer
h2_x = tf.placeholder("float", shape = [None, h1_num])
W2 = weight_variable([h1_num,h2_num])
b2 = bias_variable([h2_num])

W2_reverse = weight_variable([h2_num,h1_num])
b2_reverse = bias_variable([h1_num])

h2_out = tf.nn.sigmoid(tf.matmul(h2_x,W2) + b2)

h2_out_drop = tf.nn.dropout(h2_out,keep_prob)
h2_in_reverse = tf.nn.sigmoid(tf.matmul(h2_out_drop,W2_reverse) + b2_reverse)

loss2 = tf.reduce_sum(tf.pow(h2_in_reverse-h2_x, tf.to_float(tf.convert_to_tensor(2*numpy.ones([h1_num])))))/batch_size
train_step_2 = tf.train.GradientDescentOptimizer(0.01).minimize(loss2)

sess.run(tf.initialize_variables([W2,b2,W2_reverse,b2_reverse]))
for i in xrange(epoches):
		h2_in, train_labels, batch_x,batch_y = next_batch(h2_in, train_labels, batch_size)
		if i == 0:
			print "*****************************"
			print sess.run(tf.shape(batch_x))
			
		#batch_x = numpy.reshape(batch_x,[batch_size,sample_length])		
		sess.run(train_step_2,feed_dict={h2_x:batch_x, keep_prob:1.0})
h2_out = tf.nn.sigmoid(tf.matmul(h2_in,W2) + b2)#get result of 2nd layer as well as the input of next layer

#************************** softmax layer ****************
soft_in = sess.run(h2_out)
class_num = len(train_set.labels[0])

y_ = tf.placeholder("float", shape = [None, class_num])
soft_x = tf.placeholder("float", shape = [None, h2_num])

W_soft = weight_variable([h2_num,class_num])
b_soft = bias_variable([class_num])

y_out = tf.nn.softmax(tf.matmul(soft_x, W_soft) + b_soft)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_out))
train_step_soft = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_variables([W_soft,b_soft]))
for i in xrange(epoches):
		soft_in, train_labels, batch_x,batch_y = next_batch(soft_in, train_labels, batch_size)
			
		#batch_x = numpy.reshape(batch_x,[batch_size,sample_length])		
		sess.run(train_step_soft,feed_dict={soft_x:batch_x, y_:batch_y, keep_prob:1.0})
		if i%50 == 0:
			print sess.run(accuracy, feed_dict={soft_x:batch_x, y_:batch_y, keep_prob:1.0})
print sess.run(accuracy, feed_dict = {soft_x:soft_in, y_:train_labels, keep_prob:1.0})



#fine-tuning
print "************************* fine-tuning ***************************"
h1_out = tf.nn.sigmoid(tf.matmul(x,W1) + b1)

h2_out = tf.nn.sigmoid(tf.matmul(h1_out,W2) + b2)

h2_out_drop = tf.nn.dropout(h2_out,keep_prob)

y_out = tf.nn.softmax(tf.matmul(h2_out_drop, W_soft) + b_soft)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_out))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for i in xrange(epoches):
		batch_x,batch_y = train_set.next_batch(batch_size)
			
		#batch_x = numpy.reshape(batch_x,[batch_size,sample_length])		
		sess.run(train_step,feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})
		if i%50 == 0:
			print sess.run(accuracy, feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})
print sess.run(accuracy, feed_dict = {x:eval_set.payloads, y_:eval_set.labels, keep_prob:1.0})
