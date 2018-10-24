def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = (self.act_hidden).fn(z)
            activations.append(activation)

        # backward pass
        a_prime = (self.act_output).derivative(zs[-1]) 
        c_prime = (self.cost).derivative(activations[-1], y) 
        if self.act_output == Softmax:
            delta = np.dot(a_prime, c_prime)
        else:
            delta = c_prime * a_prime 

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = (self.act_hidden).derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)



    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)      
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        if self.regularization == 'L1': #%%%
            self.weights = [w - (eta*lmbda*np.sign(w))/n - (eta*nw)/len(mini_batch) 
                            for w, nw in zip(self.weights, nabla_w)]             
        elif self.regularization == 'L2': #%%%
            self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw 
                            for w, nw in zip(self.weights, nabla_w)]                           
        else:
            self.weights = [w-(eta/len(mini_batch))*nw #%%%
                            for w, nw in zip(self.weights, nabla_w)]        
        
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y): 
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.
        """        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x 
        activations = [x] 
        zs = []       
        if self.masks: 
            for b, w, m in zip(self.biases[:-1], self.weights[:-1], self.masks):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = (self.act_hidden).fn(z) 
                dropout_activation = activation * m.reshape((len(m),1)) 
                activations.append(dropout_activation)
        else:
            for b, w in zip(self.biases[:-1], self.weights[:-1]):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = (self.act_hidden).fn(z)
                activations.append(activation)         
        b = self.biases[-1] 
        w = self.weights[-1]
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = (self.act_output).fn(z) 
        activations.append(activation) 

        # backward pass 
        a_prime = (self.act_output).derivative(zs[-1]) 
        c_prime = (self.cost).derivative(activations[-1], y)    
        if self.act_output == Softmax:
            delta = np.dot(a_prime, c_prime)
        else:
            delta = c_prime * a_prime 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = (self.act_hidden).derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)
