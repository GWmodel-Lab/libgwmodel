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

.. math:: 
    \beta_i = \left(X' W X \right)^{-1} X' W y 
    :label: gwr-estimator


where :math:`\beta_i=(\beta_{i0},\beta_{i1},\cdots,\beta_{ip})'` is the vector of coefficients,
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
    .. math:: k(d;b) = \exp\left\{- \frac{d^2}{2 b^2}\right\}
Exponential
    .. math:: k(d;b) = \exp\left\{- \frac{|d|}{b}\right\}
Bi-squared
    .. math:: 
        k(d;b) = \left\{
            \begin{array}{ll}
            \left[ 1 - \left( \frac{d}{b} \right)^2 \right]^2, & \mbox{if } d < b \\
            0, & \mbox{otherwise}
            \end{array}
        \right.
Tri-cube
    .. math:: 
        k(d;b) = \left\{
            \begin{array}{ll}
            \left[ 1 - \left( \frac{d}{b} \right)^3 \right]^3, & \mbox{if } d < b \\
            0, & \mbox{otherwise}
            \end{array}
        \right.
Box-car
    .. math:: 
        k(d;b) = \left\{
            \begin{array}{ll}
            1, & \mbox{if } d < b \\
            0, & \mbox{otherwise}
            \end{array}
        \right.

The parameter :math:`b` is called "bandwidth".
Its value is usually automatically optimized by the golden-selection algorithm from data according to some criterions.
Usually, the following criterions are supported,

Cross-validation (CV)
    For given bandwidth :math:`b`, the CV value is defined by
    
    .. math:: CV(b) = \sum_{i=1}^n \left( y - x_i \hat{\beta}_{-i} \right)^2

    where :math:`x_i` is the :math:`i`-th row of :math:`X`,
    and :math:`\hat{\beta}_{-i}` is the coefficient vector estimated without sample :math:`i`.
    It is also calibrated according to :eq:`gwr-estimator` but set :math:`w_{ii} = 0`.

Corrected Akaike Information Criterion (AIC:sub:`c`)
    For given bandwidth :math:`b`, the AIC value is defined by

    .. math:: AIC(b) = 2n \ln \hat{\sigma} + n \ln 2pi + n \left\{ \frac{n+tr(S)}{n - 2 - tr(S)} \right\}
    
    where :math:`\hat{\sigma}` is the estimated deviation of random error,
    :math:`S` is called the "hat matrix" which is defined by

    .. math::
        S = \begin{pmatrix}
        x_1 (X'W_1X)^{-1}X'W_1 \\
        x_2 (X'W_2X)^{-1}X'W_2 \\
        \vdots \\
        x_n (X'W_nX)^{-1}X'W_n
        \end{pmatrix}
    
    and it works like

    .. math:: \hat{y} = Sy


Distance Metrics
----------------

Not only euclidean distance but also any kinds of distance metrics can be applied in GWR.
Currently, there are two kinds of distance metrics supported.

CRS Distance
    Distance as the crow flies is calculated according to the type of coordinate reference system (CRS).
    When the CRS is projected, for two samples at :math:`(u_i,v_i)` and :math:`(u_j,v_j)`,

    .. math:: d_{ij} = \sqrt{ (u_i - u_j)^2 + (v_i - v_j)^2 }
    
    When the CRS is geographic, their great circle distance is calculated.

Minkwoski Distance
    This metric is only applicable when the CRS is projected. It is defined by

    .. math:: d_{ij} = \sqrt[p]{ |u_i - u_j|^p + |v_i - v_j|^p }

In the future, we will support to set distances by a matrix file.


Example
-------

To calibrate a basic GWR model, use :class:`gwm::GWRBasic`.

Basic Usage
^^^^^^^^^^^

.. code:: cpp

    #include <armadillo>
    using namespace arma;

    mat coords = randr(100, 2, distr_param(0, 25));
    mat x = join_rows(ones(100, 1), randn(100, 2));
    mat beta = join_rows(
        ones(100) * 3.0,
        1.0 + (coords.col(0) + coords.col(1)) / 12.0,
        1.0 + (36.0 - (6.0 - coords.col(0) / 2)) % (36.0 - (6.0 - coords.col(1) / 2)) / 324
    );
    vec y = sum(x % beta, 1);

    CGwmCRSDistance distance(false);
    CGwmBandwidthWeight bandwidth(25, true, CGwmBandwidthWeight::Gaussian);
    CGwmSpatialWeight spatial(&bandwidth, &distance);

    GWRBasic algorithm;
    algorithm.setCoords(coords);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    mat beta_hat = algorithm.fit();

Bandwidth Optimization
^^^^^^^^^^^^^^^^^^^^^^

If you are not confident about the bandwidth value,
you can also let the algorithm optimize it by making the following changes:

.. code:: cpp

    GWRBasic algorithm;
    algorithm.setCoords(coords);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::AIC);
    mat beta_hat = algorithm.fit();

The argument passing to :func:`GWRBasic::setBandwidthSelectionCriterion`
can be either value of :enum:`GWRBasic::BandwidthSelectionCriterionType`.

Independent Variable Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you do not want to include all independent variables into the model and just include some significant variables,
you can let the algorithm optimize variables by making the following changes:

.. code:: cpp

    GWRBasic algorithm;
    algorithm.setCoords(coords);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setIsAutoselectIndepVars(true);
    algorithm.setIndepVarSelectionThreshold(3.0);
    mat beta_hat = algorithm.fit();

The argument passing to :func:`GWRBasic::setIndepVarSelectionThreshold` is the threshold of AIC change
determining whether one model is significantly different from another.
Generally speaking, the size of this value depends on the number of samples.
Data set of larger number of samples may need a larger threshold.
