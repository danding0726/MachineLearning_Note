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

#### Find best function

Perceptron,SVM



### Generative Model(Gaussian Distribution)

$$
f_{\mu,\sum(x)}=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\sum|^{1/2}}exp\lbrace\frac{-1}{2}(x-\mu)^T*{\sum}^{-1}(x-\mu)\rbrace
$$

Determined by Mean $\mu$ ,convariance matrix $\sum$

#### Maximum Likelihood

![Likelihood](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/Likelihood.png?token=AR2JMK7QXY27QM2G7GJMAB3ESCYZY)
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

![image-20230619215322797](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/Classification_Operation?token=AR2JMK7RMROKMAW67H3DS6LESCZMO)
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

![image-20230619220430716](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/After_Modifying.png?token=AR2JMKYBSWAK42SGIRBOHY3ESC2TI)

### Three Steps

#### Function Set(Model)

$$
x\Rightarrow P(C_1|x)=\frac{P(x|C_1)*P(C_1)}{P(x|C_1)*P(C_1)+P(x|C_2)*P(C_2)}\Rightarrow\begin{cases}P(C_1|x)>0.5\rightarrow class1\\P(C_1|x)<0.5\rightarrow class2\end{cases}
$$

#### Goodness of a function

mean $\mu$ ,convariance $\sum$ Maximizing the likelihood

#### Transformation

![image-20230619221322687](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/Posterior_Probability.png?token=AR2JMK4ZDEPVGOSHW5PR3LTESC3UU)

![image-20230619221506375](https://raw.githubusercontent.com/danding0726/MachineLearning_Note/main/Posterior_Probability_1.png?token=AR2JMKZJT3X4JKVP6QWI7KTESC322)

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

