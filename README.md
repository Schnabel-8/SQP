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
