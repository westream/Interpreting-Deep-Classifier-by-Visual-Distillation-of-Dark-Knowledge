# Interpreting-Deep-Classifier-by-Visual-Distillation-of-Dark-Knowledge
## Background
>>  A natural idea for Interpreting black box classifiers is to visualize the deep network’s rep-  
>>  resentations,so as to “see what the network sees”.   However, standard dimension   
>>  reduction methods in this setting can yield uninformative or even misleading visualizations.  

##  Contribution
>>  Present DarkSight, which visually summarizes the predictions of a classifier in a way  
>>  inspired by notion of dark knowledge.

>>  DarkSight embeds the data points into a low-dimensional space such that it is easy   
>>  to compress the deep classifier into a simpler one, essentially combining model   
>>  compression and dimension reduction.

##  Methods   
###  1.  Goal   
>>>  interpret the predictions of a black-box classifier by visualizing them in a lower    
>>>  dimensional space.     
###  2.  Important parameters     
####  a.  Validation set: $D_V = {(x_i , c_i )}$   
####  b.  Prediction vector: $π_i = P_T (c_i |x_i )$ ,prediction vector produced by the teacher    
####  c.  Low-dimensional embedding: $y_i$ ,represent each  $x_i$ in the visualization embedding
####  d.  student classifier: $P_S(·|y; θ)$





##  Code treasure
###  1.  Class Knowledge  
>>>  input : teacher logit  
>>>  output: log(softmax(teacher logit_div_by_T))

```python
class Knowledge:

    def __init__(self, logit_np, T=1):

        self.T = T

        logit = torch.from_numpy(logit_np).float()
        
        print("[Knowledge.__init__] {0} with size of {1} is loaded".format(type(logit), logit.size()))

        # Generate teacher's label
        # np.argmax(logit.numpy(), axis=1) ：axis=0 ，对列求最大，行不变 , shape = H*1；
        #                                    axis=1 ，对行求最大，列不变 , shape = 1*W；
        self.label_pred_np = np.argmax(logit.numpy(), axis=1)
        
        # N - #data points, H
        # C - #classes, W
        N, C = logit.size()

        # Convert logit to probability
        logit_div_by_T = logit / T
        
        #  Generate teacher's output
        p = torch.exp(logit_div_by_T) / torch.sum(torch.exp(logit_div_by_T), 1).view(N, 1).expand(N,C)

        # Log for numerical stability
        log_p = logit_div_by_T - log_sum_exp_stable_mat(logit_div_by_T)

        self.N = N
        self.C = C

        self.logit = logit
        self.log_p = log_p

    def ready(self, use_cuda, gpu_id):

        if use_cuda:
            
            self.logit = self.logit.cuda(gpu_id)
            self.log_p = self.log_p.cuda(gpu_id)
```
 
