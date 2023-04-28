# HW1

## 问题描述

实现SQP求解器

## 原理

####首先考虑等式约束问题

$$\begin{equation*}
\begin{split}
&\min_{x} \,\, f(x)\\
&s.t.\quad  \left\{\begin{array}{lc}
c(x)=0\\
\end{array}\right.
\end{split}
\end{equation*}$$

其中$c(x)=(c_1(x),\cdots,c_m(x))^T$

记$A(x)=[\nabla c(x)]^T=(\nabla c_1(x),\cdots,\nabla c_m(x))^T$

由最优性条件知：x是等式约束的K-T点当且仅当存在乘子$\lambda \in R^m$使得

$\nabla f(x)-A(x)^T\lambda=0$

且x是一个可行点，即$c(x)=0$

于是得到方程组

$$\left\{
\begin{aligned}
\nabla f(x)-A(x)^T\lambda=0 \\
-c(x)=0
\end{aligned}
\right.
$$

我们使用Newton_Raphson迭代求解以上方程组

$$\begin{pmatrix}
W(x,\lambda)&-A(x)^T\\
-A(x)&0\\
\end{pmatrix}
\begin{pmatrix}
\delta_x\\
\delta_{\lambda}\\
\end{pmatrix}
=-\begin{pmatrix}
\nabla f(x)-A(x)^T\lambda\\
-c(x)\\
\end{pmatrix}$$

其中$W(x,\lambda)=\nabla^2f(x)-\sum_{i=1}^m\lambda_i\nabla^2c_i(x)$

由最优性条件，$\delta_{x^{(x)}}$为下列二次规划问题的K-T点：

$$\begin{equation*}
\begin{split}
&\min \,\, \frac{1}{2}d^TW(x^{(k)},\lambda^{(k)})d+\nabla f(x^{(k)})^Td\\
&s.t.\quad  \left\{\begin{array}{lc}
c(x^{(k)})+A(x^{(k)})d=0\\
\end{array}\right.
\end{split}
\end{equation*}$$

#### 我们可以将该方法推广到一般的非线性约束最优化问题

$$\begin{equation*}
\begin{split}
&\min_{x} \,\, f(x)\\
&s.t.\quad  \left\{\begin{array}{lc}
c_i(x)=0,\, i\in \mathcal{E}\\
c_i(x)\geq 0,\, i\in \mathcal{I}\\
\end{array}\right.
\end{split}
\end{equation*}$$

在第k次迭代中求解子问题

$$\begin{equation*}
\begin{split}
&\min \,\, \frac{1}{2}d^TW(x^{(k)},\lambda^{(k)})d+\nabla f(x^{(k)})^Td\\
&s.t.\quad  \left\{\begin{array}{lc}
c(x^{(k)})+A(x^{(k)})d=0\\
\end{array}\right.
\end{split}
\end{equation*}$$

#### Han(1977)提出的逐步二次规划法为：

0. 给定$x^{(0)},W_0 \in \Reals^{n\times n},\sigma >0, \rho \in (0,1), \epsilon \geq 0, k=0$

1. 求解上述子问题给出$d^{(k)}$,如果$||d^{(k)}||\leq \epsilon$则停止，否则求$\alpha_k \in [0,\rho]$使得
$P(x^{(k)}+\alpha_k d^{(k)},\sigma)\leq \min_{0 \leq \alpha \leq \rho} P(x^{(k)}+\alpha d^{(k)},\sigma)+\epsilon_k$

2. 置$x^{(k+1)}=x^{(k)}+\alpha_k\delta_{x^(k)}$,计算$W_{k+1}$,令$k=k+1$,返回第一步

其中

$P(x,\sigma)=f(x)+\sigma(\sum_{i=1}^{m_e}|c_i(x)|+\sum_{i=m_e+1}^{m}|c_i(x)_-|)$

#### 该方法的收敛性结果为：

假定f(x)和$c_i(x)$连续可微，且存在常数$M_1 , M_2 > 0$使得
$M_1||d||^2 ≤ d^T W_k d ≤ M_2||d||^2 , ∀k ∈ \N, ∀d ∈ \R^n ,$如果$||λ(k)||_\infty ≤ \sigma$均成立，则Han(1977)算法产生的点列{$x^{(k)} $} 的任何聚点都是问题(40)的K-T点。

## 结果