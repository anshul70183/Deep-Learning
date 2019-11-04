
# defining sigmoid, tanh, relu functions and their first order derivatives
def sigmoid(x):
    # to avoid overflow and underflow, input is clipped b/w -500 and 500 and the function gives same output  
    # for values beyond them
    x = np.clip(x, -500, 500)
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)* (1- sigmoid(x))

def tanh(x):
    # to avoid overflow and underflow, input is clipped b/w -500 and 500 and the function gives same output  
    # for values beyond them
    x = np.clip(x, -500, 500)
    return (np.exp(x)- np.exp(-x))/(np.exp(x)+ np.exp(-x))

def tanh_prime(x):
    return 1 - (tanh(x)**2)

def relu(x):
    x[x<0] = 0
    return x

def relu_prime(x):
    x[x<0] = 0
    x[x>0] = 1
    return x

# defining softmax function
def softmax(x):
    m = np.amax(x, axis=1)
    m= m[:, np.newaxis]
    # to avoid overflow, x-m is used, where m is max feature value of a datapoint
    tot = np.sum((np.exp(x-m)), axis =1)
    tot = tot[:, np.newaxis]
    return (np.exp(x-m)/ tot)

# defining weights randomly with random seed of 1234
def weight_init(sizes):
    num_layers = len(sizes)
    biases = []
    weights = []
    np.random.seed(1234)
    for i in range(num_layers-1):  
        biases.append(np.random.randn(1, sizes[i+1]))
        # scaling down of weights to an appropriate level
        biases[-1] = biases[-1] / np.sqrt(sizes[i])
        weights.append(np.random.randn(sizes[i], sizes[i+1]))
        # scaling down of weights to an appropriate level
        weights[-1] = weights[-1] / np.sqrt(sizes[i])
    return biases, weights

# defining dropout 
class Dropout():

    def __init__(self,prob):
        self.prob = 1- prob
        self.params = []

    def forward(self,X):
        self.mask = np.random.binomial(1,self.prob,size=X.shape) / self.prob
        out = X * self.mask
        return out.reshape(X.shape)
    
    def backward(self,dout):
        dX = dout * self.mask
        return dX,[]
    
# defining forward propagation function
def forward_pass(x, y, sizes, b, w, activation, dropout_p):
    
    ai = x
    a = [x]
    z = []
    
    # if dropout is not used 
    if dropout_p== 0:
        if activation == 'sigmoid':
            for bi, wi in zip(b[:-1], w[:-1]):
                zi = np.dot(ai,wi)+bi
                z.append(zi)
                ai= sigmoid(zi)
                a.append(ai)
                
        elif activation == 'tanh':
            for bi, wi in zip(b[:-1], w[:-1]):
                zi = np.dot(ai,wi)+bi
                z.append(zi)
                ai= tanh(zi)
                a.append(ai)  
                
        elif activation == 'relu':
            for bi, wi in zip(b[:-1], w[:-1]):
                zi = np.dot(ai,wi)+bi
                z.append(zi)
                ai= relu(zi)
                a.append(ai)              
                
        for bi, wi in zip(b[-1:], w[-1:]):
            zi = np.dot(ai,wi)+bi
            z.append(zi)
            ai= softmax(zi)
            a.append(ai)
            
    # if dropout is used   
    elif dropout_p!=0:
        if activation == 'sigmoid':
            for bi, wi in zip(b[:-1], w[:-1]):
                zi = np.dot(ai,wi)+bi
                z.append(zi)
                ai= sigmoid(zi)
                ai_dropout = Dropout(prob= dropout_p).forward(ai)            
                a.append(ai_dropout)
    
                
        elif activation == 'tanh':
            for bi, wi in zip(b[:-1], w[:-1]):
                zi = np.dot(ai,wi)+bi
                z.append(zi)
                ai= tanh(zi)
                ai_dropout = Dropout(prob= dropout_p).forward(ai)            
                a.append(ai_dropout)
                
        elif activation == 'relu':
            for bi, wi in zip(b[:-1], w[:-1]):
                zi = np.dot(ai,wi)+bi
                z.append(zi)
                ai= relu(zi)
                ai_dropout = Dropout(prob= dropout_p).forward(ai)            
                a.append(ai_dropout)

        for bi, wi in zip(b[-1:], w[-1:]):
            zi = np.dot(ai,wi)+bi
            z.append(zi)
            ai= softmax(zi)
            a.append(ai)
        
    return z, a

# one hot vector function
def one_hot(y):
    m= np.shape(y)[0]
    n = 10
    y_one_hot = np.zeros((m,n))
    for i in range(m):
        y_one_hot[i, int(y[i])] = 1
    return y_one_hot

# loss function 
def loss_fn(x, y, y_pred, sizes, b, w, loss, lambd):
    y_one_hot = one_hot(y)
    wei_tot =0
    for i in range(len(w)):
        wei_tot+= np.sum(w[i]*w[i])
        
    if loss =='ce':
        tot = -1* (np.sum(y_one_hot* np.log(y_pred))) + (lambd * wei_tot)
    elif loss =='sq':
        tot = 0.5 * np.sum((y_one_hot- y_pred)**2) + (lambd * wei_tot)
    return tot


# back propagation 
def back_prop(x, y, sizes, b ,w , z, a, activation, loss, lambd):
    y_one_hot = one_hot(y)
    grad_b =[]
    grad_w =[]
    
    for bi, wi in zip(b,w):
        grad_b.append(np.zeros(bi.shape))
        grad_w.append(np.zeros(wi.shape))
    
    if loss == 'ce':
        delta=  a[-1] - y_one_hot 
        grad_b[-1] = np.sum(delta, axis =0).reshape(1,np.shape(delta)[1]) 
        grad_w[-1] = np.dot(a[-2].T , delta) + lambd * w[-1]
        
    elif loss == 'sq':
        delta=  a[-1] * (a[-1] - y_one_hot)
        grad_b[-1] = np.sum(delta, axis =0).reshape(1,np.shape(delta)[1]) 
        grad_w[-1] = (np.dot(a[-2].T , delta)) + lambd * w[-1]
    
    if activation == 'sigmoid':
        for i in range(2, len(sizes)):
            delta = np.dot(delta , w[1-i].T)* sigmoid_prime(z[-i])
            grad_b[-i] = np.sum(delta, axis =0).reshape(1,np.shape(delta)[1]) 
            grad_w[-i] = (np.dot(a[-1-i].T , delta)) + lambd * w[-i]
            
    elif activation =='tanh':
        for i in range(2, len(sizes)):
            delta = np.dot(delta , w[1-i].T)* tanh_prime(z[-i])
            grad_b[-i] = np.sum(delta, axis =0).reshape(1,np.shape(delta)[1]) 
            grad_w[-i] = (np.dot(a[-1-i].T , delta)) + lambd * w[-i]
            
    elif activation =='relu':
        for i in range(2, len(sizes)):
            delta = np.dot(delta , w[1-i].T)* relu_prime(z[-i])
            grad_b[-i] = np.sum(delta, axis =0).reshape(1,np.shape(delta)[1]) 
            grad_w[-i] = (np.dot(a[-1-i].T , delta))+ lambd * w[-i]
            
    return grad_b, grad_w


# function for error output (rounded off to 2 decimal places)
def error(x, y, y_pred, sizes, b,w):
    n = len(y)
    result =  np.argmax(y_pred, axis=1)
    result = result[:, np.newaxis]
    
    return np.round(((1- ((result ==y).astype(int).sum()/ n))*100 ), 2)

#for saving weights
def save_weights(list_of_weights, epoch, save_dir):
    with open(save_dir + 'weights_{}.pkl'.format(epoch), 'wb') as f:
        pickle.dump(list_of_weights, f)

# for loading weights
def load_weights(state, save_dir):
    with open(save_dir +'weights_{}.pkl'.format(state), 'rb') as f:
        list_of_weights = pickle.load(f)
    return list_of_weights

# gradient descent algorithm
def gd(train_data, valid_data, test_data, epochs, batch_size, hidden_sizes, lr, activation, loss, lambd, dropout_p, save_dir, expt_dir,
       anneal, pretrain, state, testing):
    
    train_data_arr = np.array(train_data)
    n = np.shape(train_data_arr)[1]
    x = train_data_arr[:,:n-1]
    y = train_data_arr[:,n-1:]
    m = np.shape(x)[0]
    
    valid_data_arr = np.array(valid_data)
    x_valid = valid_data_arr[:,:n-1]
    y_valid = valid_data_arr[:,n-1:]
    m_valid = np.shape(x_valid)[0]
    
    test_data_arr = np.array(test_data)
    #n_test = np.shape(test_data_arr)[1]
    x_test = test_data_arr[:,:]
    #y_test = test_data_arr[:,n_test:]
    m_test = np.shape(x_test)[0]
    
    sizes = hidden_sizes
    sizes.append(len(np.unique(y)))
    sizes.insert(0, np.shape(x)[1])
     
    # using pretrained weights
    if (pretrain == 'true' and testing =='false'):
        params = load_weights(state, save_dir)
        b= params[:len(sizes)-1]
        w= params[len(sizes)-1:]
        
    # using random weight initialization   
    elif (pretrain == 'false' and testing =='false'):
        b, w = weight_init(sizes)
    
    train_loss_epoch= []
    valid_loss_epoch= []
    
    valid_error_epoch =[]
        
    file = open( expt_dir + "log_train.txt", "w")
    file.close()
    file1 = open(expt_dir +"log_val.txt", "w")
    file1.close()
    
    epoch = 1
    t=1
    while epoch <= epochs:
        np.random.shuffle(train_data_arr)
        n = np.shape(train_data_arr)[1]
        x = train_data_arr[:,:n-1]
        y = train_data_arr[:,n-1:]
        
        for k in range(0, m, batch_size):
            xi = x[k: k+batch_size, :] 
            yi = y[k: k+batch_size, :] 
            
            xi_valid =  x_valid[k: k+batch_size, :] 
            yi_valid = y_valid[k: k+batch_size, :] 
            
                
            zi, ai = forward_pass(xi, yi, sizes, b, w, activation, dropout_p)
                
            grad_b, grad_w = back_prop(xi, yi, sizes, b , w , zi, ai, activation, loss, lambd)   
            g_w = grad_w
            g_b = grad_b

                
            
            for j in range(len(w)):
                w[j] = w[j]- (lr* g_w[j])
                b[j] = b[j]- (lr* g_b[j])
            
            if t%100==0:
                z, a = forward_pass(x, y, sizes, b, w, activation, dropout_p)
                z_valid, a_valid = forward_pass(x_valid, y_valid, sizes, b, w,  activation, dropout_p)
                
                train_loss= (loss_fn(x, y, a[-1], sizes, b, w, loss, lambd)) / m
                valid_loss=(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd)) / m_valid
                    
                train_error=(error(x, y, a[-1], sizes, b,w))
                valid_error=(error(x_valid, y_valid, a_valid[-1], sizes, b, w))
                
                file = open(expt_dir +"log_train.txt", "a") 
                print("Training: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4}".format(epoch, 
                                                t, train_loss, train_error, lr))
                file.write("Training: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4} \n".format(epoch, 
                                                t, train_loss, train_error, lr))
                file.close()
                
                file1 = open(expt_dir +"log_val.txt", "a") 
                print("Validation: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4}".format(epoch, 
                                        t, valid_loss, valid_error, lr))
                file1.write("Validation: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4} \n".format(epoch, 
                                                t, train_loss, train_error, lr))
                file1.close()
                print('')
                
            t+=1
        z, a = forward_pass(x, y, sizes, b, w, activation, dropout_p)
        z_valid, a_valid = forward_pass(x_valid, y_valid, sizes, b, w,  activation, dropout_p)

        
        valid_error_epoch.append(error(x, y, a[-1], sizes, b,w))
        
        if (anneal == 'true' and epoch != 1):
            if (valid_error_epoch[-1]> valid_error_epoch[-2]):
                lr = lr/2
                params = load_weights(epoch-1, save_dir)
                b= params[:len(sizes)-1]
                w= params[len(sizes)-1:]
         
            else:
                params = b + w
                save_weights(params, epoch, save_dir)
                epoch+=1
                
                train_loss_epoch.append(loss_fn(x, y, a[-1], sizes, b, w, loss, lambd) / m)
                valid_loss_epoch.append(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd) / m_valid)
            
        else:
            params = b+ w
            save_weights(params, epoch, save_dir)
            epoch+=1
            
            train_loss_epoch.append(loss_fn(x, y, a[-1], sizes, b, w, loss, lambd) / m)
            valid_loss_epoch.append(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd) / m_valid)
            
    return b,w, train_loss_epoch, valid_loss_epoch

# momentum based gradient descent 
def momentum_gd(train_data, valid_data, test_data, epochs, batch_size, hidden_sizes, lr, activation, loss,  lambd, dropout_p,save_dir, expt_dir,
         gamma, anneal, pretrain, state , testing):
    
    train_data_arr = np.array(train_data)
    n =np.shape(train_data_arr)[1]
    x = train_data_arr[:,:n-1]
    y = train_data_arr[:,n-1:]
    m = np.shape(x)[0]
    
    valid_data_arr = np.array(valid_data)
    n =np.shape(train_data_arr)[1]
    x_valid = valid_data_arr[:,:n-1]
    y_valid = valid_data_arr[:,n-1:]
    m_valid = np.shape(x_valid)[0]
    
    test_data_arr = np.array(test_data)
    #n_test = np.shape(test_data_arr)[1]
    x_test = test_data_arr[:,:]
   # y_test = test_data_arr[:,n_test:]
    m_test = np.shape(x_test)[0]
    
    sizes = hidden_sizes
    sizes.append(len(np.unique(y)))
    sizes.insert(0, np.shape(x)[1])
     
    # using pretrained weights 
    if (pretrain == 'true' and testing =='false'):
        params = load_weights(state, save_dir)
        b= params[:len(sizes)-1]
        w= params[len(sizes)-1:]
        
    # using random weight initialization    
    elif (pretrain == 'false' and testing =='false'):
        b, w = weight_init(sizes)
        
    update_b =[]
    update_w = []
    
    train_loss_epoch= []
    valid_loss_epoch= []
    
    valid_error_epoch =[]
    
    file = open( expt_dir + "log_train.txt", "w")
    file.close()
    file1 = open(expt_dir +"log_val.txt", "w")
    file1.close()
    
    epoch = 1
    t=1
    for bi, wi in zip(b,w):
        update_b.append(np.zeros(bi.shape))
        update_w.append(np.zeros(wi.shape))
       
    while epoch <= epochs:
        np.random.shuffle(train_data_arr)
        n = np.shape(train_data_arr)[1]
        x = train_data_arr[:,:n-1]
        y = train_data_arr[:,n-1:]
        
        
        for k in range(0, m, batch_size):
            xi = x[k: k+batch_size, :] 
            yi = y[k: k+batch_size, :] 
            
            xi_valid =  x_valid[k: k+batch_size, :] 
            yi_valid = y_valid[k: k+batch_size, :] 
                
            zi, ai = forward_pass(xi, yi, sizes, b, w, activation, dropout_p)
                
            grad_b, grad_w = back_prop(xi, yi, sizes, b , w , zi, ai, activation, loss, lambd)   
            g_w = grad_w
            g_b = grad_b

            
            for j in range(len(w)):
                update_w[j] = (gamma * update_w[j]) + (lr* g_w[j])
                update_b[j] = (gamma * update_b[j]) + (lr* g_b[j])
                
                w[j] = w[j]- (update_w[j])
                b[j] = b[j]- (update_b[j])
            
            if t%100 ==0:
                z, a = forward_pass(x, y, sizes, b, w, activation, dropout_p)
                z_valid, a_valid = forward_pass(x_valid, y_valid, sizes, b, w,  activation, dropout_p)
                
                train_loss= (loss_fn(x, y, a[-1], sizes, b, w, loss, lambd)) / m
                valid_loss=(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd)) / m_valid
                    
                train_error=(error(x, y, a[-1], sizes, b,w))
                valid_error=(error(x_valid, y_valid, a_valid[-1], sizes, b, w))
                
                file = open(expt_dir +"log_train.txt", "a") 
                print("Training: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4}".format(epoch, 
                                                t, train_loss, train_error, lr))
                file.write("Training: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4} \n".format(epoch, 
                                                t, train_loss, train_error, lr))
                file.close()
                
                file1 = open(expt_dir +"log_val.txt", "a") 
                print("Validation: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4}".format(epoch, 
                                        t, valid_loss, valid_error, lr))
                file1.write("Validation: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4} \n".format(epoch, 
                                                t, train_loss, train_error, lr))
                file1.close()
                print('')
            t+=1    
            
        z, a = forward_pass(x, y, sizes, b, w, activation, dropout_p)
        z_valid, a_valid = forward_pass(x_valid, y_valid, sizes, b, w,  activation, dropout_p)

        
        
        valid_error_epoch.append(error(x, y, a[-1], sizes, b,w))
        
        if (anneal == 'true' and epoch != 1):
            if (valid_error_epoch[-1]> valid_error_epoch[-2]):
                lr = lr/2
                params = load_weights(epoch-1, save_dir)
                b= params[:len(sizes)-1]
                w= params[len(sizes)-1:]
         
            else:
                params = b + w
                save_weights(params, epoch, save_dir)
                epoch+=1
                train_loss_epoch.append(loss_fn(x, y, a[-1], sizes, b, w, loss, lambd) / m)
                valid_loss_epoch.append(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd) / m_valid)
        else:
            params = b+ w
            save_weights(params, epoch, save_dir)
            epoch+=1
            train_loss_epoch.append(loss_fn(x, y, a[-1], sizes, b, w, loss, lambd) / m)
            valid_loss_epoch.append(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd) / m_valid)
    return b,w, train_loss_epoch, valid_loss_epoch


# nestrov based gradient descent 
def nestrov_gd(train_data, valid_data, test_data, epochs, batch_size, hidden_sizes, lr, activation, loss,  lambd, dropout_p,save_dir, expt_dir,
         gamma, anneal, pretrain, state , testing ):
    
    train_data_arr = np.array(train_data)
    n = np.shape(train_data_arr)[1]
    x = train_data_arr[:,:n-1]
    y = train_data_arr[:,n-1:]
    m = np.shape(x)[0]
    
    valid_data_arr = np.array(valid_data)
    n = np.shape(train_data_arr)[1]
    x_valid = valid_data_arr[:,:n-1]
    y_valid = valid_data_arr[:,n-1:]
    m_valid = np.shape(x_valid)[0]
    
    test_data_arr = np.array(test_data)
    x_test = test_data_arr[:,:]
    #y_test = test_data_arr[:,n_test:]
    m_test = np.shape(x_test)[0]
    
    sizes = hidden_sizes
    sizes.append(len(np.unique(y)))
    sizes.insert(0, np.shape(x)[1])
    
    # using pretrained weigths
    if (pretrain == 'true' and testing =='false'):
        params = load_weights(state, save_dir)
        b= params[:len(sizes)-1]
        w= params[len(sizes)-1:]
        
    #using random initialization for weights
    elif (pretrain == 'false' and testing =='false'):
        b, w = weight_init(sizes)
        
    update_b =[]
    update_w = []
    b_la =[]
    w_la= []
    
    train_loss_epoch= []
    valid_loss_epoch= []
    
    valid_error_epoch =[]
        
    file = open( expt_dir + "log_train.txt", "w")
    file.close()
    file1 = open(expt_dir +"log_val.txt", "w")
    file1.close()
    t=1
    epoch = 1
    
    for bi, wi in zip(b,w):
        update_b.append(np.zeros(bi.shape))
        b_la.append(np.zeros(bi.shape))
                
        update_w.append(np.zeros(wi.shape))
        w_la.append(np.zeros(wi.shape))
       
    while epoch <= epochs:
        np.random.shuffle(train_data_arr)
        n = np.shape(train_data_arr)[1]
        x = train_data_arr[:,:n-1]
        y = train_data_arr[:,n-1:]
        
        
        for k in range(0, m, batch_size):
            xi = x[k: k+batch_size, :] 
            yi = y[k: k+batch_size, :] 
            
            xi_valid =  x_valid[k: k+batch_size, :] 
            yi_valid = y_valid[k: k+batch_size, :] 

            for j in range(len(w)):
             
               w_la[j] = w[j]- gamma * update_w[j]
               b_la[j] = b[j]- gamma * update_b[j]
                
            zi, ai = forward_pass(xi, yi, sizes, b_la, w_la, activation, dropout_p)
                
            grad_b, grad_w = back_prop(xi, yi, sizes, b_la , w_la , zi, ai, activation, loss, lambd)   
            g_w = grad_w
            g_b = grad_b

            
            for j in range(len(w)):
                update_w[j] = (gamma * update_w[j]) + (lr* g_w[j])
                update_b[j] = (gamma * update_b[j]) + (lr* g_b[j])
                
                w[j] = w[j]- (update_w[j])
                b[j] = b[j]- (update_b[j])
            
            if t%100 ==0:
                z, a = forward_pass(x, y, sizes, b, w, activation, dropout_p)
                z_valid, a_valid = forward_pass(x_valid, y_valid, sizes, b, w,  activation, dropout_p)
                
                train_loss= (loss_fn(x, y, a[-1], sizes, b, w, loss, lambd)) / m
                valid_loss=(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd)) / m_valid
                    
                train_error=(error(x, y, a[-1], sizes, b,w))
                valid_error=(error(x_valid, y_valid, a_valid[-1], sizes, b, w))
                
                file = open(expt_dir +"log_train.txt", "a") 
                print("Training: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4}".format(epoch, 
                                                t, train_loss, train_error, lr))
                file.write("Training: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4} \n".format(epoch, 
                                                t, train_loss, train_error, lr))
                file.close()
                
                file1 = open(expt_dir +"log_val.txt", "a") 
                print("Validation: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4}".format(epoch, 
                                        t, valid_loss, valid_error, lr))
                file1.write("Validation: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4} \n".format(epoch, 
                                                t, train_loss, train_error, lr))
                file1.close()
                print('')
            t+=1    
            
        z, a = forward_pass(x, y, sizes, b, w, activation, dropout_p)
        z_valid, a_valid = forward_pass(x_valid, y_valid, sizes, b, w,  activation, dropout_p)

        
        
        valid_error_epoch.append(error(x, y, a[-1], sizes, b,w))
        
        if (anneal == 'true' and epoch != 1):
            if (valid_error_epoch[-1]> valid_error_epoch[-2]):
                lr = lr/2
                params = load_weights(epoch-1, save_dir)
                b= params[:len(sizes)-1]
                w= params[len(sizes)-1:]
         
            else:
                params = b + w
                save_weights(params, epoch, save_dir)
                epoch+=1
                train_loss_epoch.append(loss_fn(x, y, a[-1], sizes, b, w, loss, lambd) / m)
                valid_loss_epoch.append(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd) / m_valid)
        else:
            params = b+ w
            save_weights(params, epoch, save_dir)
            epoch+=1
            train_loss_epoch.append(loss_fn(x, y, a[-1], sizes, b, w, loss, lambd) / m)
            valid_loss_epoch.append(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd) / m_valid)
    return b,w, train_loss_epoch, valid_loss_epoch


# adam 
def adam(train_data, valid_data, test_data, epochs, batch_size, hidden_sizes, lr, activation, loss, lambd, dropout_p, save_dir, expt_dir,
           anneal, pretrain, state , testing, beta1 =0.9, beta2= 0.999, eps= 1e-08):
    
    train_data_arr = np.array(train_data)
    n = np.shape(train_data_arr)[1]
    x = train_data_arr[:,:n-1]
    y = train_data_arr[:,n-1:]
    #print(y.shape)
    m = np.shape(x)[0]
    
    valid_data_arr = np.array(valid_data)
    n = np.shape(train_data_arr)[1]
    x_valid = valid_data_arr[:,:n-1]
    y_valid = valid_data_arr[:,n-1:]
    m_valid = np.shape(x_valid)[0]
    
    test_data_arr = np.array(test_data)
    #n_test = np.shape(test_data_arr)[1]
    x_test = test_data_arr[:,:]
    #y_test = test_data_arr[:,n-1:]
    m_test = np.shape(x_test)[0]
    
    sizes = hidden_sizes
    sizes.append(len(np.unique(y)))
    sizes.insert(0, np.shape(x)[1])
    
    
    # using pretrained weights    
    if (pretrain == 'true' and testing =='false'):
        params = load_weights(state, save_dir)
        b= params[:len(sizes)-1]
        w= params[len(sizes)-1:]
        
     # using random initialization  
    elif (pretrain == 'false' and testing =='false'):
        b, w = weight_init(sizes)
    

    
    m_b =[]
    m_w =[]
    v_b =[]
    v_w = []
    
    m_b1 =[]
    m_w1 =[]
    v_b1 =[]
    v_w1 = []
    
    train_loss_epoch= []
    valid_loss_epoch= []
    
    valid_error_epoch =[]
    
    file = open( expt_dir + "log_train.txt", "w")
    file.close()
    file1 = open(expt_dir +"log_val.txt", "w")
    file1.close()
    
    t = 1
    epoch = 1
    
    
    for bi, wi in zip(b,w):
        v_b.append(np.zeros(bi.shape))
        m_b.append(np.zeros(bi.shape))
                
        v_w.append(np.zeros(wi.shape))
        m_w.append(np.zeros(wi.shape)) 
        
        v_b1.append(np.zeros(bi.shape))
        m_b1.append(np.zeros(bi.shape))
                
        v_w1.append(np.zeros(wi.shape))
        m_w1.append(np.zeros(wi.shape))
       
    while epoch <= epochs:
        np.random.shuffle(train_data_arr)
        n = np.shape(train_data_arr)[1]
        x = train_data_arr[:,:n-1]
        y = train_data_arr[:,n-1:]
        
        
        for k in range(0, m, batch_size):
            xi = x[k: k+batch_size, :] 
            yi = y[k: k+batch_size, :] 
            
            xi_valid =  x_valid[k: k+batch_size, :] 
            yi_valid = y_valid[k: k+batch_size, :] 
  
                
            zi, ai = forward_pass(xi, yi, sizes, b, w, activation, dropout_p)
                
            grad_b, grad_w = back_prop(xi, yi, sizes, b , w , zi, ai, activation, loss, lambd)   
            g_w = grad_w
            g_b = grad_b
            
            
            for j in range(len(w)):
                m_w[j] = beta1 * m_w[j] + (1-beta1) *(g_w[j])
                m_b[j] = beta1 * m_b[j] + (1-beta1) *(g_b[j])
                
                v_w[j] = beta2 * v_w[j] + (1-beta2) * (g_w[j]**2)
                v_b[j] = beta2 * v_b[j] + (1-beta2) *(g_b[j]**2)
                
                m_w1[j] = m_w[j] / (1- (math.pow(beta1,t)))
                m_b1[j] = m_b[j] / (1- (math.pow(beta1,t)))
                
                v_w1[j] = v_w[j] / (1- (math.pow(beta2,t)))
                v_b1[j] = v_b[j] / (1- (math.pow(beta2,t)))
                
                
                w[j] = w[j]- (lr/ (np.sqrt(v_w1[j] + eps)) * m_w1[j])
                b[j] = b[j]- (lr/ (np.sqrt(v_b1[j] + eps)) * m_b1[j])
            
            
            
            if (t%100)==0:
                z, a = forward_pass(x, y, sizes, b, w, activation, dropout_p)
                z_valid, a_valid = forward_pass(x_valid, y_valid, sizes, b, w,  activation, dropout_p)
                
                train_loss= (loss_fn(x, y, a[-1], sizes, b, w, loss, lambd)) / m
                valid_loss=(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd)) / m_valid
                    
                train_error=(error(x, y, a[-1], sizes, b,w))
                valid_error=(error(x_valid, y_valid, a_valid[-1], sizes, b, w))
                
                file = open(expt_dir +"log_train.txt", "a") 
                print("Training: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4}".format(epoch, 
                                                t, train_loss, train_error, lr))
                file.write("Training: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4} \n".format(epoch, 
                                                t, train_loss, train_error, lr))
                file.close()
                
                file1 = open(expt_dir +"log_val.txt", "a") 
                print("Validation: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4}".format(epoch, 
                                        t, valid_loss, valid_error, lr))
                file1.write("Validation: Epoch {0}, Step {1}, Loss {2}, Error {3}, lr {4} \n".format(epoch, 
                                                t, train_loss, train_error, lr))
                file1.close()
                print('')
                
            t+=1
            
        z, a = forward_pass(x, y, sizes, b, w, activation, dropout_p)
        z_valid, a_valid = forward_pass(x_valid, y_valid, sizes, b, w,  activation, dropout_p)

        valid_error_epoch.append(error(x, y, a[-1], sizes, b,w))
        
        if (anneal == 'true' and epoch != 1):
            if (valid_error_epoch[-1]> valid_error_epoch[-2]):
                lr = lr/2
                params = load_weights(epoch-1, save_dir)
                b= params[:len(sizes)-1]
                w= params[len(sizes)-1:]
         
            else:
                params = b + w
                save_weights(params, epoch, save_dir)
                epoch+=1
                train_loss_epoch.append(loss_fn(x, y, a[-1], sizes, b, w, loss, lambd) / m)
                valid_loss_epoch.append(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd) / m_valid)
        else:
            params = b+ w
            save_weights(params, epoch, save_dir)
            epoch+=1
            train_loss_epoch.append(loss_fn(x, y, a[-1], sizes, b, w, loss, lambd) / m)
            valid_loss_epoch.append(loss_fn(x_valid, y_valid, a_valid[-1], sizes, b, w, loss, lambd) / m_valid)
    return b,w, train_loss_epoch, valid_loss_epoch




# final error printing 
def total_error(data, hidden_sizes, b, w, activation, dropout_p):
    data_arr = np.array(data)
    n = np.shape(data_arr)[1]
    x = data_arr[:,:n-1]
    y = data_arr[:,n-1:]
    m = np.shape(x)[0]
    
    sizes = hidden_sizes
    sizes.append(len(np.unique(y)))
    sizes.insert(0, np.shape(x)[1])
    z, a = forward_pass(x, y, sizes, b, w, activation, dropout_p)
    
    return error(x, y, a[-1],sizes, b, w)

#testing function
def testing_true(test_data, hidden_sizes, activation, state, save_dir, expt_dir,  dropout_p):
        sizes = hidden_sizes
        sizes.append(10)
        sizes.insert(0, np.shape(test_data)[1])
        
        params = load_weights(state, save_dir)
        b= params[:len(sizes)-1]
        w= params[len(sizes)-1:]
        
        y = np.zeros((np.shape(test_data)[0],1))
        
        z,a = forward_pass(test_data, y, sizes, b, w, activation, dropout_p)
        y_pred = a[-1]
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_pred_df_labels = pd.DataFrame(y_pred_labels)
        y_pred_df_labels.index.names = ['id']
        y_pred_df_labels.columns = ['label']
        y_pred_df_labels.to_csv(expt_dir + 'predictions_{}.csv'.format(state))
        return
# =============================================================================
# def test_pred_export(data, hidden_sizes, b, w, activation, expt_dir, out, dropout_p):
#     test_data_arr = np.array(data)
#     n = np.shape(test_data_arr)[1]
#     x = test_data_arr[:,:]
#     
#     m = np.shape(x)[0]
#     y = np.zeros((m,1))
#     
#     sizes = hidden_sizes
#     sizes.append(len(np.unique(y)))
#     sizes.insert(0, np.shape(x)[1])
#     
#     z,a = forward_pass(x, y, sizes, b, w, activation, dropout_p)
#     y_pred = a[-1]
#     y_pred_labels = np.argmax(a[-1], axis=1)
#     y_pred_df_labels = pd.DataFrame(y_pred_labels)
#     y_pred_df_labels.index.names = ['id']
#     y_pred_df_labels.columns = ['label']
#     y_pred_df_labels.to_csv(expt_dir + 'predictions_{}.csv'.format(out))
#     return
# =============================================================================

# pca function
def pca_trans(pca_n_components, train_data, valid_data, test_data,state):
    train_data_arr = np.array(train_data)
    n = np.shape(train_data_arr)[1]
    x_train = train_data_arr[:,1:n-1]
    y_train = train_data_arr[:,n-1:]
    
    valid_data_arr = np.array(valid_data)
    x_valid = valid_data_arr[:,1:n-1]
    y_valid = valid_data_arr[:,n-1:]
    
    test_data_arr = np.array(test_data)
    x_test = test_data_arr[:,1:]
    #y_test = test_data_arr[:,n-1:]
    
    pca = PCA(n_components=pca_n_components, random_state = 1234).fit(x_train)
    x_train =pca.transform(x_train)
    x_valid= pca.transform(x_valid)
    x_test= pca.transform(x_test)
    
    train_data = np.concatenate((x_train,y_train), axis =1)
    valid_data = np.concatenate((x_valid,y_valid), axis =1)
    test_data = np.array(x_test)
    
# =============================================================================
#     train_data_df = pd.DataFrame(train_data)
#     valid_data_df = pd.DataFrame(valid_data)
#     test_data_df = pd.DataFrame(test_data)
#     
#     
#     train_data_df.to_csv('../save_dir/best/train_{}.csv'.format(state)) 
#     valid_data_df.to_csv('../save_dir/best/valid_{}.csv'.format(state)) 
#     test_data_df.to_csv('../save_dir/best/test_{}.csv'.format(state)) 
# =============================================================================

    return train_data, valid_data, test_data

# importing necessary libraries
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA 
import pickle

# parsing arguments using argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type = float, default = 0.0001)
parser.add_argument("--momentum",type=float, default = 0.9)
parser.add_argument("--num_hidden", type= int, default = 3)
parser.add_argument("--sizes", type= str, default = '200,200,200')
parser.add_argument("--activation", type= str, choices= ['sigmoid', 'tanh', 'relu'], default= 'sigmoid')
parser.add_argument("--loss", type= str, choices= ['sq', 'ce'],default= 'ce')
parser.add_argument("--opt", type= str, choices= ['gd', 'momentum', 'nag', 'adam'], default= 'adam')
parser.add_argument("--batch_size", type= int, default= 20)
parser.add_argument("--epochs", type= int, default= 10)
parser.add_argument("--anneal", type= str,choices= ['true', 'false'], default= 'true')
parser.add_argument("--save_dir", type= str,  default = '../save_dir/')
parser.add_argument("--expt_dir", type= str, default = '../expt_dir/')
parser.add_argument("--train", type= str, default = '../data/train.csv')
parser.add_argument("--val", type= str, default='../data/valid.csv')
parser.add_argument("--test", type= str, default= '../data/test.csv')

parser.add_argument("--pretrain", type= str, choices= ['true', 'false'], default= 'false')
parser.add_argument("--state", type= int, default= 1)
parser.add_argument("--testing", type= str, choices= ['true', 'false'], default= 'false')

parser.add_argument("--lambd", type= float, default= 0.0)
parser.add_argument("--dropout_p", type= float, default= 0.0)
parser.add_argument("--pca", type= str, choices= ['true', 'false'], default= 'false')
parser.add_argument("--pca_n_components", type= int, default= 200)

#taking arguments in args
args = parser.parse_args() 

# loading params from args
lr= args.lr
gamma = args.momentum
num_hidden = args.num_hidden
hidden_sizes = [int(item) for item in args.sizes.split(',')]
activation = args.activation
loss = args.loss
opt = args.opt
batch_size = args.batch_size
epochs = args.epochs
anneal = args.anneal
save_dir = args.save_dir
expt_dir = args.expt_dir
train_dir = args.train
val_dir = args.val
test_dir = args.test

pretrain = args.pretrain
state = args.state
testing = args.testing

lambd = args.lambd
dropout_p = args.dropout_p
pca = args.pca
pca_n_components = args.pca_n_components

# loading data 
if (testing == 'false'):
    train_data = pd.read_csv(train_dir)
    valid_data = pd.read_csv(val_dir)
    test_data = pd.read_csv(test_dir)

elif (testing =='true'):
    test_data = pd.read_csv(test_dir)
    
#if pca is enabled, transforming data based on pca_n_components
if (pca == 'true' and testing == 'false'):
    train_data, valid_data, test_data = pca_trans(pca_n_components, train_data, valid_data, test_data, state)

# if pca is disable, taking default data after removing the index column
elif (pca =='false' and testing == 'false'):
    train_data_arr = np.array(train_data)
    train_data = train_data_arr[:,1:]
    valid_data_arr = np.array(valid_data)
    valid_data= valid_data_arr[:,1:]
    test_data_arr = np.array(test_data)
    test_data = test_data_arr[:,1:]

elif testing =='true':
    test_data_arr = np.array(test_data)
    test_data = test_data_arr[:,1:]

# exporting when testing is enabled
if testing == 'true':
    testing_true(test_data, hidden_sizes, activation, state, save_dir, expt_dir, dropout_p)

#training when testing is disabled   
elif opt == 'gd':    
    b, w, train_loss, valid_loss= gd(train_data, valid_data, test_data, epochs, batch_size, hidden_sizes, lr, activation, loss, lambd, dropout_p, save_dir, expt_dir,
       anneal, pretrain, state, testing)
    
    if testing == 'false':
        print('Final Training Error:', total_error(train_data, hidden_sizes, b, w, activation, dropout_p))
        print('Final Validation Error:', total_error(valid_data, hidden_sizes, b, w, activation, dropout_p))
    
elif opt == 'momentum':
    b, w, train_loss, valid_loss = momentum_gd(train_data, valid_data, test_data, epochs, batch_size, hidden_sizes, lr, activation, loss, lambd, dropout_p, save_dir, expt_dir,
         gamma, anneal, pretrain, state , testing)
    if testing == 'false':
        print('Final Training Error:', total_error(train_data, hidden_sizes, b, w, activation, dropout_p))
        print('Final Validation Error:', total_error(valid_data, hidden_sizes, b, w, activation, dropout_p))
    
elif opt =='nag':
    b, w, train_loss, valid_loss = nestrov_gd(train_data, valid_data, test_data, epochs, batch_size, hidden_sizes, lr, activation, loss,lambd, dropout_p,  save_dir, expt_dir,
         gamma, anneal, pretrain, state , testing)
    if testing == 'false':
        print('Final Training Error:', total_error(train_data, hidden_sizes, b, w, activation, dropout_p))
        print('Final Validation Error:', total_error(valid_data, hidden_sizes, b, w, activation, dropout_p))
    
elif opt =='adam':

    b, w, train_loss, valid_loss = adam(train_data, valid_data, test_data, epochs, batch_size, hidden_sizes, lr, 
           activation , loss , lambd, dropout_p,save_dir , expt_dir ,  anneal , pretrain, state , testing, beta1= 0.9, beta2=0.999, eps =1e-08 )
    if testing =='false':
        print('Final Training Error:', total_error(train_data, hidden_sizes, b, w, activation, dropout_p))
        print('Final Validation Error:', total_error(valid_data, hidden_sizes, b, w, activation, dropout_p))



