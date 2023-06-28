# Machine Learning Course （Li Hongyi 2023)

## 1.ChatGPT Introduction

### Possible Method ：

Pre-Train 预训练  => Self-Supervised  自督导式学习 => Supervised learning 督导式学习 => Intensive training 强化训练

### Foreground:

Prompting 工程，Neural Editing , AI 检测，Machine Unlearning



------



## 2.Regression

### Step1 Linear Model

$$
y = b+\omega*x_{cp} => y = b+\sum{w_i*x_i}
$$

$$
x_i:feature,w_i:weight,b:bias
$$



### Step2 Goodness of function

Loss Function L 
$$
L(f)= L(\omega,b)
$$
Normal 
$$
L(f)=\sum^{10}_{n=1}{(\hat{y}^n-(b+\omega*x^n_{cp}))^2}
$$

### Step3 Gradient Descent 

#### For one parameter

$$
\omega^*=argmin_\omega{L(\omega)}
$$

1.pick initial value $\omega^0$

2.$\omega^1 \leftarrow \omega^0-\eta*\frac{dL}{d\omega}|_{\omega=\omega^0}$ 

3.$\omega^2 \leftarrow \omega^1-\eta*\frac{dL}{d\omega}|_{\omega=\omega^1}$

.... => Local optimal (not global)

#### For two parameters

1.$\omega^1 \leftarrow \omega^0-\eta*\frac{dL}{d\omega}|_{\omega=\omega^0,b=b^0}$  

$b^1 \leftarrow b^0-\eta*\frac{dL}{db}|_{\omega=\omega^0,b=b^0}$

2.$\omega^2 \leftarrow \omega^1-\eta*\frac{dL}{d\omega}|_{\omega=\omega^1,b=b^1}$  

$b^2 \leftarrow b^1-\eta*\frac{dL}{db}|_{\omega=\omega^1,b=b^1}$

......

### For Many Types

#### back to design model

$$
y=b_1*\delta(x_s=pidgey)+\omega_1*\delta(x_s=pidgey)*x_{cp}+....+b_4*\delta(x_s=Eevee)+\omega_4*\delta(x_s=Eevee)
$$

$$
\delta=\begin{cases}
1, &x_s=type\\
0, &x_s\neq type
\end{cases}
$$

#### Regularization

$$
L=\sum_n{(\hat{y}-(b+\sum{\omega_ix_i})})^2+\lambda\sum{w_i^2}
$$

$\lambda$越大，找到的越平滑



------



## 3.Classification

$$
x\rightarrow Function \rightarrow ClassN
$$

### Ideal Alternatives

#### Function(Model)

$$
x \Rightarrow \begin{cases}
g(x) >0|Output=class1\\
else|Output=class2
\end{cases}
$$

#### Loss Function

$$
L(f)=\sum_n{\delta(f(x^n)\neq\hat{y}^n)}
$$

Number of times f get incorrect results on training data.

#### Find best fusnction

Perceptron,SVM



### Generative Model(Gaussian Distribution)

$$
f_{\mu,\sum(x)}=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\sum|^{1/2}}exp\lbrace\frac{-1}{2}(x-\mu)^T*{\sum}^{-1}(x-\mu)\rbrace
$$

Determined by Mean $\mu$ ,convariance matrix $\sum$

#### Maximum Likelihood

![](![](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/Likelihood.png?token=AR2JMK2OEKMHKFKROJVI4WLESC6GA)
$$
L(\mu,\sum)=f_{\mu,\sum}(x^1)*f_{\mu,\sum}(x^2).....f_{\mu,\sum}(x^79)
$$

$$
\Rightarrow(\mu^*,{\sum}^*)=argMax_{\mu,\sum}L(\mu,\sum)
$$

$$
\mu^*=\frac{1}{79}\sum^{79}_{n=1}x^n\\
{\sum}^*=\frac{1}{79}\sum_{n=1}^{79}(x^n-\mu^*)(x^n-\mu^*)^T
$$

#### Back To Classification

![](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/Classification_Operation.png?token=AR2JMKYYYNCC44SL6JODZKLESC6V6)
$$
P(C_1|x)>0.5\Rightarrow x\in Class1
$$
从多维空间来看增加更多参数容易导致overfitting 

### Resolution

给两边分类相同的$\sum$ 
$$
L(\mu_1,\mu_2,\sum)=f_{\mu_1,\sum}(x^1)*f_{\mu_1,\sum}(x^2).....f_{\mu_1,\sum}(x^{79})f_{\mu_2,\sum}(x^{80})...f_{\mu_2,\sum}(x^{140})
$$

$$
\Rightarrow\mu_1=\mu_2\\
\Rightarrow\sum=\frac{79}{140}{\sum}^1+\frac{61}{140}{\sum}^2
$$

![](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/After_Modifying.png?token=AR2JMK7RXQGK3OGN5LRCT3TESC6VS)

### Three Steps

#### Function Set(Model)

$$
x\Rightarrow P(C_1|x)=\frac{P(x|C_1)*P(C_1)}{P(x|C_1)*P(C_1)+P(x|C_2)*P(C_2)}\Rightarrow\begin{cases}P(C_1|x)>0.5\rightarrow class1\\P(C_1|x)<0.5\rightarrow class2\end{cases}
$$

#### Goodness of a function

mean $\mu$ ,convariance $\sum$ Maximizing the likelihood

#### Transformation



![](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/Posterior_Probability.png?token=AR2JMK6JZW26S2EHRJOUDS3ESC6UW)

![](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/Posterior_Probability_1.png?token=AR2JMKY3UDWINDC7WIYPQU3ESC6VA)

##### If $\sum_1=\sum_2=\sum$

$$
z=(\mu^1-\mu^2)^T{\sum}^{-1}x-\frac{1}{2}(\mu^1)^T({\sum}^1)^{-1}\mu^1+\frac{1}{2}(\mu^2)^T({\sum}^1)^{-1}\mu^2+\ln\frac{N1}{N2}
$$

$$
z=w^Tx-b
$$

$$
\Rightarrow P(C_1|x)=\sigma(w*x+b)
$$

------

## 4.Logistic Regression

### Comparison with Linear Regression

#### Step1

##### For Logistic Regression

$$
f_{w,b}(x)=\sigma(\sum_iw_ix_i+b)
$$

Output: between 0 and 1

##### For Linear Regression

$$
f_{w,b}(x)=\sum_iw_ix_i+b
$$

Output: any value

#### Step2 Goodness of a Function

Training Data : 

![image-20230626204553421](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/image-20230626204553421.png)
$$
L(w,b)=f_{w,b}(x^1)f_{w,b}(x^2)(1-f_{w,b}(x^3))...f_{w,b}(x^N)
$$
The most likely $w^*$ and $b^*$ is the one with the largest $L(w,b)$
$$
w^*,b^*=arg\max_{w,b}L(w,b)
$$

$$
\Rightarrow w^*,b^*=arg\max_{w,b}-\ln L(w,b)
$$

$$
-lnf_{w,b}(x^1)\Rightarrow-[\hat{y}^1lnf(x^1)+(1-\hat{y}^1ln(1-f(x^1))]\\
-lnf_{w,b}(x^2)\Rightarrow-[\hat{y}^2lnf(x^2)+(1-\hat{y}^2ln(1-f(x^2))]\\
-lnf_{w,b}(x^3)\Rightarrow-[\hat{y}^3lnf(x^3)+(1-\hat{y}^3ln(1-f(x^3))]\\
....
$$

$$
\Rightarrow -lnL(w,b)=\sum_n{-[\hat{y}^nlnf_{w,b}(x^n)+(1-\hat{y}^nln(1-f_{w,b}(x^n))]}
$$

Cross entropy between two Bernoulli distribution

![image-20230626205500605](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/image-20230626205500605.png)

##### For Logistic Regression

$\hat{y}^n$:1 for class 1, 0 for class 2
$$
L(f)=\sum_nC(f(x^n),\hat{y}^n)
$$

##### For Linear Regression

$$
L(f)=\frac{1}{2}\sum_n(f(x^n)-\hat{y}^n)^2
$$

#### Step3 Find the best function

![image-20230626210646281](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/image-20230626210646281.png)

![image-20230626210740806](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/image-20230626210740806.png)

![image-20230626211035757](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/image-20230626211035757.png)

##### For Logistic Regression and  LInear Regression

The same
$$
w_i\leftarrow w_i-\eta\sum_n{-(\hat{y}^n-f_{w,b}(x^n))x_i^n}
$$

### If Use Logistic Regression with square error

![image-20230626212043790](img/image-20230626212043790.png)

用Square error 离目标很远时趋势也很小,导致update很慢.

### Discriminative(Logistic Regression) vs Generative Model

#### Generative Model

 做了假设 数据来源于某种分布 故适合于训练量少的情况

With the assumption of probability distribution, less training data is needed

With the assumption of probability distribution, more robust to the noise

Priors and class-dependent probabilities can be estimated from different sources

#### Discriminative 

没有做任何假设  

### Multi-class Classification

![image-20230628203829769](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/image-20230628203829769.png)

![image-20230628203900555](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/image-20230628203900555.png)

### Limitation of Logistic Regression

Boundary 为一根直线

#### Feature Transformation

Not alaways easy to find a good transformation

##### Use to cascade logistic regression models

![image-20230628204109807](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/img/image-20230628204109807.png)

Every Logistic regression called "Neuron" , Get Neural Network 神经网络
