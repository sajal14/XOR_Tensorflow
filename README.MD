We ran several experiments with different configurations of the deep net. We measured the accuracy over 
different configurations and the results are the following. We 1000 iterations for the experiments.

Number of 		 Number of 			     Optimizer	    Activation			     Cost            Accuracy
Hidden Layers	 Hidden Layer neurons					         Function
	1					      10				           Adam		   Relu + Softmax	       0.16-0.278           1
														                      at the end	
	1					      10				           Adam		   Relu + Softmax	       0.37-0.342          0.75
														                      at the end
	1				        100				           Adam		   Relu + Softmax	       0.002 - 0.09         1 
														                      at the end 
	2					      10 				           Adam 		 Relu + Softmax	       0.003 - 0.11			    1
														                      at the end					
	2					      10				           SGD		   Relu + Softmax	       0.0002 - 0.002		    1       
														                      at the end
	2					      10				         AdaGrad	   Relu + Softmax	       0.0005 - 0.001		    1
														                      at the end
	2					      10 				          Adam 		   Sigmoid + Softmax     0.029 - 0.5754		    1
														                      at the end
      
From here we see that the dnn for xor reaches the accuracy even with 1 hidden layer and 10 neurons. As we increase the 
number of layers or the number of neurons in the layers, the minimized cost decreases. The best 
minimized cost comes at with Relu activation functions and SGD optimizer. Adam optimation does not perform that well.
