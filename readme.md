# Rank-Models for Robust Regression Below 200 Observations

The (current) purpose of this repository is test some models and collect
data about their performance, compared to existing state-of-the-art
models. It appears that building robust regression models gets more
difficult with fewer observations. Suppose you have a regression (or
classification) problem with **less than 200** labeled observations,
perhaps even less than **20**! Obtaining a model with good
generalization then is difficult.

We propose to transform the data to ranks, using its estimated
distribution. Conversely, we then transform back the result using an
inverse CDF. This is very similar to generalized linear models. In fact,
we are currently testing using a linear model. However, we also
introduce non-linearity to allow the model to be fit to more difficult
problems. For CDF and PPF, we currently support the Gaussian Kernel, an
empirical Kernel, and a smoothed version thereof. It is conceivable, for
example, to implement distribution fitting in the future.

Currently, the model looks like this: smooth_min(1, smooth_max(0, x))


\begin{aligned}
    \mathsf{S}(x)&=\frac{1}{1+\exp{(-x)}},\\;\text{the Sigmoid function used for non-linear activation,}
    \\\\\[1ex\]
    \mathsf{Swish}(x,\beta)&=x\\;\mathsf{S}\left(\beta x\right),\\;\text{where}\\;\frac{1}{10}\leq\beta\leq{2}\\;\text{(typically),}
    \\\\\[1ex\]
    \max\_{\mathrm{soft}}(x_1,x_2)&=\frac{1}{2}\left(x_1 + x_2 + \sqrt{(x_1 - x_2)^2 + \alpha}\right),\\;\text{a smooth approximation of max, where}\\;1e^{-3}\leq\alpha\leq{5e^{-2}}\\;\text{(typically),}
    \\\\\[0ex\]
    \min\_{\mathrm{soft}}(x_1,x_2)&=\dots\\;\text{same as max, except we flip the sign in front of the square root,}
    \\\\\[1ex\]
    \mathsf{S}\_{\mathrm{hard}}(x)&=\min\_{\mathrm{soft}}\left(1,\max\_{\mathrm{soft}}\left(0,x\right)\right),\\;\text{a linear and hard "Sigmoid" with continuous behavior around}\\;x=\\{0,1\\},
    \\\\\[1em\]
    m&=\mathsf{PPF}\_{Y}\left(\mathsf{S}\_{\mathrm{hard}}\left(b_m+\left\[w_1\times \mathsf{Swish}(a_1+b_1\times F\_{X_1}(x_1))\\;+\\;\dots\\;+\\;w_n\times\mathsf{Swish}\left(a_n+b_n\times F\_{X_n}(x_n)\right)\right\]\right)\right).
    \\\\\[0ex\]
    &=\mathsf{PPF}\_{Y}\left(\mathsf{S}\_{\mathrm{hard}}\left(b_m+\left\[\sum\_{i=1}^N\\,w_i\times\mathsf{Swish}\left(a_i+b_i\times F\_{X_i}(x_i)\right)\right\]\right)\right),
    \\\\\[0ex\]
    &=\mathsf{PPF}\_{Y}\left(\mathsf{S}\_{\mathrm{hard}}\left(b_m+ \mathbf{w}^\top\mathbf{k}\right)\right),
    \\\\\[0ex\]
    &\qquad\text{where}\\;k_i=\mathsf{Swish}\left(a_i+b_i\times F\_{X_i}\left(x_i\right)\right).
\end{aligned}


For when the model has multiple outputs, the model has one PPF per
output, each with its own linear activation (model weight
*a*<sub>*m*<sub>*i*</sub></sub> and -bias
*b*<sub>*m*<sub>*i*</sub></sub>):


\begin{aligned}
    m&=\begin{cases}
        \mathsf{PPF}\_{Y_1}\left(\mathsf{S}\_{\mathrm{hard}}\left(a\_{m_1}+b\_{m_1}\times\mathbf{w}^\top\mathbf{k}\right)\right),
        \\\\
        \quad\vdots
        \\\\
        \mathsf{PPF}\_{Y_n}\left(\mathsf{S}\_{\mathrm{hard}}\left(a\_{m_n}+b\_{m_n}\times\mathbf{w}^\top\mathbf{k}\right)\right),
    \end{cases}
\end{aligned}


This way, all the weights related to *k*<sub>*i*</sub> are shared. An
alternative would be to use a complete own set of weights for each
output, which will essentially result in an extra model for each output.
