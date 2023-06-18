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



