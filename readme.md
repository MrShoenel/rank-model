# Rank-Models for Robust Regression Below 200 Observations

The (current) purpose of this repository is to test some models and
collect data about their performance, compared to existing
state-of-the-art models. It appears that building robust regression
models gets more difficult with fewer observations. Suppose you have a
regression (or classification) problem with **less than 100** labeled
observations, perhaps even fewer than **20**! Obtaining a model with
good generalization then is difficult.

We propose to transform the data to ranks, using its estimated
distribution. Conversely, we then transform back the result using an
inverse CDF. This is very similar to generalized linear models. In fact,
we are currently testing using a linear model. However, we also
introduce non-linearity to allow the model to be fit to more difficult
problems. For CDF and PPF, we currently support the Gaussian Kernel, an
empirical Kernel, and a smoothed version thereof. It is conceivable, for
example, to implement distribution fitting in the future.

Currently, the model looks like this:

$$
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
    m\left(\mathbf{x},a_m,b_m,\mathbf{w},\mathbf{a},\mathbf{b}\right)&=\mathsf{PPF}\_{Y}\left(\mathsf{S}\_{\mathrm{hard}}\left(a_m+b_m\times\left\[w_1\times \mathsf{Swish}(a_1+b_1\times F\_{X_1}(x_1))\\;+\\;\dots\\;+\\;w_n\times\mathsf{Swish}\left(a_n+b_n\times F\_{X_n}(x_n)\right)\right\]\right)\right),
    \\\\\[0ex\]
    &=\mathsf{PPF}\_{Y}\left(\mathsf{S}\_{\mathrm{hard}}\left(a_m+b_m\times\left\[\sum\_{i=1}^N\\,w_i\times\mathsf{Swish}\left(a_i+b_i\times F\_{X_i}(x_i)\right)\right\]\right)\right),
    \\\\\[0ex\]
    &=\mathsf{PPF}\_{Y}\left(\mathsf{S}\_{\mathrm{hard}}\left(a_m+b_m\times\mathbf{w}^\top\mathbf{k}\right)\right),
    \\\\\[1em\]
    &\qquad\quad\text{where}\\;k_i=\mathsf{Swish}\left(a_i+b_i\times F\_{X_i}\left(x_i\right)\right).
\end{aligned}
$$

Above model predicts the response *y* for a single datum **x** that has
1…*n* features. For when the model has multiple outputs, the model has
one PPF per output, each with its own linear activation (model bias
*a*<sub>*m*<sub>*i*</sub></sub> and -weight
*b*<sub>*m*<sub>*i*</sub></sub>):

$$
\begin{aligned}
    m&=\begin{cases}
        \mathsf{PPF}\_{Y_1}\left(\mathsf{S}\_{\mathrm{hard}}\left(a\_{m_1}+b\_{m_1}\times\mathbf{w}^\top\mathbf{k}\right)\right),
        \\\\
        \quad\vdots
        \\\\
        \mathsf{PPF}\_{Y_n}\left(\mathsf{S}\_{\mathrm{hard}}\left(a\_{m_n}+b\_{m_n}\times\mathbf{w}^\top\mathbf{k}\right)\right),
    \end{cases}
\end{aligned}
$$

This way, all the weights related to *k*<sub>*i*</sub> are shared. An
alternative would be to use a complete own set of weights for each
output, which will essentially result in an extra model for each output.
This is not what we will do, because if this were the purpose, one would
simply create separate models instead.

It is obvious that this model is somewhat similar to a **Generalized
Linear Model** (GLM). However, here are some differences:

-   For each feature in the training data, we estimate a corresponding
    CDF. This could be an ECDF or a fitted parametric distribution. As
    of writing this, we support the ECDF, an interpolated and smoothed
    version of it, as well as Gauss- and Logistic distributions. Next,
    we will implement distribution fitting.
-   We introduce **non-linearity** by wrapping each rank-transformed
    feature using an additional **activation**, very similar to how
    neural networks do it. Before passing the dot-product to the
    activation, we also apply a linear transformation to allow the
    transformed feature to change its location and scale. We also add an
    additional weight for each such activation.
-   The **link**-function used in GLMs often is a `logit`. Here we
    always use the inverse CDF (PPF) of the response variable. We have
    the same options as for the features (ECDF, parametric, etc.).
-   Since the link-function is a PPF, it has a domain of \[0,1\]. We
    therefore pass the result of the computation into a somewhat
    hardened Sigmoid, that, in actuality, is more like a ReLU with range
    \[0,1\] and is continuously smooth in the neighborhoods of
    *x* = {0, 1} to enable smooth gradients (currently, we obtain a
    numeric gradient, so this is of great importance) (Biswas et al.
    2022). Also, we pass the result with one last additional parameter,
    a “model-bias” *b*<sub>*m*</sub>.

# References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-Biswas_2022_CVPR" class="csl-entry">

Biswas, Koushik, Sandeep Kumar, Shilpak Banerjee, and Ashish Kumar
Pandey. 2022. “Smooth Maximum Unit: Smooth Activation Function for Deep
Networks Using Smoothing Maximum Technique.” In *Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*,
794–803.

</div>

</div>
