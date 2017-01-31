import tensorflow as tf

x_labels = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_labels = [[1,0],  [0,1],  [0,1], [1,0]]


x = tf.placeholder("float", [None,2])
y = tf.placeholder("float", [None, 2]) 


num_nodes_h1 = 10 
num_nodes_h2 = 10



def nn_model(data):
	h1_layer = {'weights' : tf.Variable(tf.random_normal([2,num_nodes_h1])),
	 'biases' : tf.Variable(tf.zeros([num_nodes_h1]))}
	h2_layer = {'weights' : tf.Variable(tf.random_normal([num_nodes_h1,num_nodes_h2])),
	'biases' : tf.Variable(tf.zeros([num_nodes_h2]))}
	output_layer = {'weights' : tf.Variable(tf.random_normal([num_nodes_h1,2])),
	'biases' : tf.Variable(tf.random_normal([2]))}

	tf.summary.histogram("W_h1",h1_layer["weights"])
	tf.summary.histogram("W_h2",h2_layer["weights"])
	tf.summary.histogram("bias_h1",h1_layer["biases"])
	tf.summary.histogram("bias_h2", h2_layer["biases"])


	l1 = tf.add(tf.matmul(data,h1_layer["weights"]) , h1_layer["biases"] )
	l1 = tf.nn.relu(l1)
	# l1 = tf.nn.sigmoid(l1)

	l2 = tf.add(tf.matmul(l1,h2_layer["weights"]) , h2_layer["biases"] )
	l2 = tf.nn.relu(l2)
	# l2 = tf.nn.sigmoid(l2)

	out = tf.add(tf.matmul(l2,output_layer["weights"]) , output_layer["biases"] )
	
	return out



def train_nn(x):
	prediction = nn_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	tf.summary.scalar("cost",cost)

	# tf.summary.scalar('cross_entropy', cross_entropy)

	optimizer = tf.train.AdamOptimizer().minimize(cost)
	# optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(cost)
	# optimizer = tf.train.AdagradOptimizer(0.2).minimize(cost)
	# merged = tf.summary.merge_all()
	# train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
    #                                   sess.graph)
	correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
	tf.summary.scalar("accuracy",accuracy)

	with tf.Session() as sess:
		writer = tf.train.SummaryWriter("./logs",sess.graph)
		merged = tf.merge_all_summaries()
		#merged = tf.summary.merge_all()


		sess.run(tf.initialize_all_variables())

		for step in range(1000):
			e,c=sess.run([optimizer,cost],feed_dict = {x: x_labels , y :y_labels })
			print ("step %d : cost(loss): %s" % (step,c))
			summary, acc = sess.run([merged,accuracy],feed_dict = {x: x_labels, y: y_labels})
			print("acc: %s" %(acc));
			writer.add_summary(summary,step)		
		
		print ("accuracy %s"%(accuracy.eval({x: x_labels, y: y_labels})))

train_nn(x)

