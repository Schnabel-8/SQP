# HW1

## 问题描述

实现SQP求解器

## 原理

首先考虑等式约束问题

$min: f(x)$

$s.t.: c(x)=0$

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