Basic GWR Model
===============

For a data set of :math:`n` samples and :math:`p` independent variables,
the basic GWR model at sample :math:`i` is defined as

.. math:: y_i = \beta_{i0} + \sum_{k=1}^p \beta_{ik}x_{ik} + \epsilon_{i}

where :math:`y_i` is the dependent variable,
:math:`x_{ik}` is the :math:`k`-th independent variable,
:math:`\beta_{ik}` is the :math:`k`-th coefficient,
:math:`\beta_{i0}` is the intercept,
:math:`\epsilon_i` is the random error which :math:`\epsilon_i \sim N(0, \sigma^2)`
and :math:`\sigma` is the standard deviation.
Then :math:`\beta_i` is calibrated by the following estimator

.. math:: \beta_i = \left(X' W X \right)^{-1} X' W y 

where :math:`\beta_i=(\beta_{i0},\beta_{i1},\cdots,\beta_{ip})` is the vector of coefficients,
:math:`X` is the matrix of independent variables,
:math:`y` is the vector of the dependent variable,
:math:`W` is called spatial weighting matrix defined by

.. math::

    W = \begin{pmatrix}
    w_{i1} & & & \\
    & w_{i2} & & \\
    & & \ddots & \\
    & & & w_{in} \\
    \end{pmatrix}
    
Each :math:`w_{ij}` in :math:`W` is calculated by a kernel function :math:`k` according to the distance from sample :math:`i` to sample :math:`j`.
Larger distance, lower weights.

Kernel functions
----------------

There are some useful kernel functions:

Gaussian
    .. math:: k(d;b) = \exp{- \frac{d^2}{2 b^2}}

Distance Metrics
----------------

Not only euclidean distance but also any kinds of distance metrics can be applied in GWR.

