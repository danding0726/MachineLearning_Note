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



### Generative Model
