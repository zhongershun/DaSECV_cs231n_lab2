from builtins import range
from builtins import object
import numpy as np

from daseCV.layers import *
from daseCV.layer_utils import *


class TwoLayerNet(object):
    """
    采用模块化设计实现具有ReLU和softmax损失函数的两层全连接神经网络。
    假设D是输入维度，H是隐藏层维度，一共有C类标签。
   
    网络架构应该是：affine - relu - affine - softmax.
    
    注意，这个类不实现梯度下降；它将与负责优化的Solver对象进行交互。
    
    模型的可学习参数存储在字典self.params中。键是参数名称，值是numpy数组。
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        ## 保存weight和bias参数
        
        ## 要求：w--正态分布，均值为0，标准差为weight_scale
        ##     b--零
        
        ## 隐藏层：input_dim * hidden_dim
        ## 输出层：hidden_dim * num_classes
        
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        
    def loss(self, X, y=None):
        """
        对小批量数据计算损失和梯度

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        
#         print("self.reg: ",self.reg)
        
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        ## 取出参数
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        ## 隐藏层计算
        hidden_layer_out, cache1 = affine_relu_forward(X, W1, b1)
        
        ## 输出层计算
        scores, cache2 = affine_forward(hidden_layer_out, W2, b2)
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        ## 计算损失以及反向传播导数
        loss, dscore = softmax_loss(scores, y)
        
        ## 反向传播计算出W2,b2的导数以及对隐藏层的导数
        
        dhidden, grads['W2'], grads['b2'] = affine_backward(dscore, cache2)
        
        ## 反向传播计算出W2,b2的导数以及对X的导数
        dX, grads['W1'], grads['b1'] = affine_relu_backward(dhidden, cache1)
        
        
        ## 对loss和梯度进行正则化
        
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        
        pass
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
    
class FullyConnectedNet(object):
    """
    一个任意隐藏层数和神经元数的全连接神经网络，其中 ReLU 激活函数，sofmax 损失函数，同时可选的
    采用 dropout 和 batch normalization(批量归一化)。那么，对于一个L层的神经网络来说，其框架是：
    
    {affine ‐ [batch norm] ‐ relu ‐ [dropout]} x (L ‐ 1) ‐ affine ‐ softmax
    
    其中的[batch norm]和[dropout]是可选非必须的，框架中{...}部分将会重复L‐1次，代表L‐1 个隐藏层。
    
    与我们在上面定义的 TwoLayerNet() 类保持一致，所有待学习的参数都会存在self.params 字典中，
    并且最终会被最优化 Solver() 类训练学习得到。
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        ## W1, b1 初始化
        self.params['W1'] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
        self.params['b1'] = np.zeros(hidden_dims[0])
        
        ## 初始化W2,W3,W4,...,W[num_layers-1]
        ## b2,b3,...,b[num_layers-1]
        for i in range(self.num_layers - 2):
            self.params['W' + str(i+2)] = np.random.randn(hidden_dims[i], hidden_dims[i+1]) * weight_scale
            self.params['b' + str(i+2)] = np.zeros(hidden_dims[i+1])
            
            if self.normalization:
                self.params['gamma' + str(i+1)] = np.ones(hidden_dims[i])
                self.params['beta' + str(i+1)] = np.zeros(hidden_dims[i])
        if self.normalization:
            self.params['gamma' + str(self.num_layers - 1)] = np.ones(hidden_dims[-1])
            self.params['beta' + str(self.num_layers - 1)] = np.zeros(hidden_dims[-1])    
        self.params['W' + str(self.num_layers)] = np.random.randn(hidden_dims[-1], num_classes)
        self.params['b' + str(self.num_layers)] = np.random.randn(num_classes)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        ##   cache1     cache2              cache[num_layers]
        ##    I         I                    I
        ## X->[ ]->layout1->[ ]->...layout[num_layers-1]->[ ]->layout[num_layers]=score
        ##   hidden     hidden                not hidden
        ##
        ##
        
        
        
#         layer_outs = [0 for _ in range(self.num_layers + 1)]
#         caches = [0 for i in range(self.num_layers + 1)]
        layer_outs = []
        caches = []
        dp_caches = list(range(self.num_layers - 1))
        
#         layer_outs[0] = X
        layer_outs.append(X)
        cache_tmp = (X,self.params['W1'], self.params['b1'])
        caches.append(cache_tmp)
#         caches[0] = (X,self.params['W1'], self.params['b1']) ## 没有用只是占个位置同时表示一下数据类型caches[1]开始记录
        
        for i in range(self.num_layers):
            W, b = self.params['W' + str(i+1)], self.params['b' + str(i+1)]
            
            if i == self.num_layers - 1:
                ## layers_out[num_layers], caches[num_layers] = affine_forward(hidden_layers[num_layers-1],W_num_layers,n_num_layers) 
                layer_out_tmp, cache_tmp = affine_forward(layer_outs[i], W, b)
                layer_outs.append(layer_out_tmp)
                caches.append(cache_tmp)
#                 layer_outs[i+1], caches[i] = affine_forward(layer_outs[i], W, b)
            
            else:
                if self.normalization == 'batchnorm': ##使用batch normalization
                    gamma, beta = self.params['gamma' + str(i+1)], self.params['beta' + str(i+1)]
                    try:
                        layer_out_tmp, cache_tmp = affine_bn_relu_forward(layer_outs[i], W, b, gamma, beta, self.bn_params[i])
                    except:
                        print(self.normalization)
                        print(self.bn_params)
                elif self.normalization == 'layernorm':
                    gamma, beta = self.params['gamma' + str(i+1)], self.params['beta' + str(i+1)]
                    try:
                        layer_out_tmp, cache_tmp = affine_ln_relu_forward(layer_outs[i], W, b, gamma, beta, self.bn_params[i])
                    except:
                        print(self.normalization)
                        print(self.bn_params)
                else:
                    layer_out_tmp, cache_tmp = affine_relu_forward(layer_outs[i], W, b)
                if self.use_dropout:
                    layer_out_tmp,dp_caches[i] = dropout_forward(layer_out_tmp,self.dropout_param)
                layer_outs.append(layer_out_tmp)
                caches.append(cache_tmp)
                ## layer_outs[i+1], caches[i+1] = affine_relu_forward(layer_outs[i],W,b)
                
        scores = layer_outs[self.num_layers]
#         print(layer_outs[2])
        
#         print(caches[self.num_layers])
        pass


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        ## 计算损失以及反向传播导数
        
        loss, dscore = softmax_loss(scores, y)
        
        ## 反向传播计算出W2,b2的导数以及对隐藏层的导数
        
        dlayer_outs = [dscore for _ in range(self.num_layers + 1)]
#         dlayer_outs[self.num_layers] = dscore
#         print(dlayer_outs[self.num_layers-1]==dlayer_outs[self.num_layers])

        
        ##   cache1     cache2              cache[num_layers]
        ##    I         I                    I
        ## X->[ ]->layout1->[ ]->...layout[num_layers-1]->[ ]->layout[num_layers]=score
        ##   hidden     hidden                not hidden
        ##
        ## dX=dlayout0 dlayout1    dlayout[num_layers-1]    dlayour[num_layers]=dscore
        
        ## 反向传播计算出W2,b2的导数以及对X的导数
        
        for i in range(self.num_layers):
            if i == 0:
#                 print(caches[self.num_layers - i])
                idx = self.num_layers - i
                dlayer_outs[idx - 1], grads['W' + str(idx)], grads['b' + str(idx)] = affine_backward(dscore,caches[idx])
                
            else:
                idx = self.num_layers - i
#                 print(caches[idx])
                if self.use_dropout:
                    dlayer_outs[idx] = dropout_backward(dlayer_outs[idx], dp_caches[idx-1])
                if self.normalization=='batchnorm': ## 进行batch normalization
                    dlayer_outs[idx - 1], grads['W' + str(idx)], grads['b' + str(idx)], grads['gamma' + str(idx)], grads['beta' + str(idx)] = affine_bn_relu_backward(dlayer_outs[idx],caches[idx])
                elif self.normalization=='layernorm': ## 进行layer normalization
                    dlayer_outs[idx - 1], grads['W' + str(idx)], grads['b' + str(idx)], grads['gamma' + str(idx)], grads['beta' + str(idx)] = affine_ln_relu_backward(dlayer_outs[idx],caches[idx])
                else: ## 不进行normalization
                    dlayer_outs[idx - 1], grads['W' + str(idx)], grads['b' + str(idx)] = affine_relu_backward(dlayer_outs[idx],caches[idx])
                
        
            ## 对loss和梯度进行正则化
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(self.num_layers - i)] ** 2)
            
            grads['W' + str(self.num_layers - i)] += self.reg * self.params['W' + str(self.num_layers - i)]
        
        dX = dlayer_outs[0]
        
        
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
