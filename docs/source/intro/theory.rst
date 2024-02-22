.. _theory:

######################
Introduction to CLUPS
######################

The Constrained Linear Unbiased Predictor is an extension of the standard BLP/BLUP formulation. The full explanation can be found in *Fraser-Govil \& Gration (2024)*, what follows is a high-level overview. 

***********************
Defining a Predictor
***********************

Suppose that you have a series of observables :math:`\{X_t\}_{t\in T}`. A *linear predictor* for the random variable :math:`Z` is given by the dot product of the vector of observables :math:`\vec{X}` and a suitable vector :math:`\vec{a}`.

The act of generating a predictor is therefore equivalent to choosing a vector :math:`\vec{a}` which maximises some objective function of your choosing. In the case of a *predictor-sequence* (or equally, *predictor-series*), we wish to predict a set of :math:`\{Z_\tau\}_{\tau \in T^\prime}`, and must therefore formulate a set of :math:`\{\vec{a}_\tau\}` for each point in our prediction space such that:

.. math::

	\hat{Z}_\tau = \vec{a}_\tau^T \vec{X}

The obvious metric for producing the 'best' such optimiser is to minimise the expected variance between the real value of :math:`Z`, and the predictor, :math:`\langle (Z - \hat{Z})^2 \rangle`. In the case of a predictor-sequence, we must simultaneously minimise the entire set of variances:

.. math::

	\mathcal{L} = \sum_{\tau \in T^\prime} \langle (Z_\tau - \hat{Z}_\tau)^2 \rangle

This, however, is predicated on our knowledge of the behaviour of :math:`Z` -- a quantity we often do not have access to (hence why it is being predicted). If, however, we are willing to make some assumptions about the behaviour of the variance kernel (or, in the case of the BLUP, the second-moment kernel), we can nevertheless construct a predictor. 

The Best Linear Predictor (BLP) assumes that the variance :math:`\text{Var}(X_i,X_j) = k(t_i,t_j)` is a known quantity, and optimises :math:`\vec{a}` accordingly. The Best Linear Unbiased Predictor (BLUP) adds in the additional requirement that :math:`\langle Z - \hat{Z} \rangle = 0` -- requiring the additional assumption (or approximation) that the mean-value of the function can be constructed from a provided, finite-dimensional basis set of functions. 

********************
The CLUPS
********************

The BLP and the BLUP are both commonly used in many different fields - however they are somewhat limited in that they are something of a black box: data goes in, and a prediction comes out. Aside from the choice of the kernel (and, in the case of the BLUP, the functional basis set), there is no way to impose on the predictor any restrictions that arise from domain knowledge. 

The CLUPS allows one to specify a constraint on the value of the predictors, provided that it can be expressed in the form

.. math::

	\sum_i B_{ij} \hat{Z}_{\tau_i} = c_j

This allows one to, for example, constraint a predictor such that 1) the first value of the predictor-sequence is equal to 0.2, and the predictor (approximately) integrates to 1, by specifying the matrix :math:`B` and vector :math:`\vec{c}` as:

.. math::

	\begin{align}
	B & = \begin{pmatrix} 1 & 0 & 0 & ... & 0
	\\
	\frac{1}{2} & 1 & 1 & ... & \frac{1}{2}\end{pmatrix}
	\\
	\vec{c} & = \begin{pmatrix} 0.2 \\ 1 \end{pmatrix}
	\end{align}

----------------------
Inexact Constraints
----------------------

One can also use the CLUPS to express inexact constraints, such as 'greater than :math:`a`'. 

In this case, we specify the constraint as an exact constraint in terms of some unknown parameters, :math:`\vec{w}`:

.. math::

	\sum_i B_{ij} \hat{Z}_{\tau_i} = c_j(\vec{w}) = \xi_j + \psi_j(\vec{w})

Here the function :math:`\psi(\vec{w}) \geq 0` is defined to be an invertible transform which ensures that the constraint is always met -- so the nature of the transform is determined by the constraint. For example, in the case of 'greater than :math:`a`, we would use:

.. math::

	\begin{align}
		\xi_j & = a 
		\\
		\psi_j(\vec{w}) = \exp(w_j)
	\end{align}

Given this transform, we then numerically optimise the values of :math:`\vec{w}`. `pyclups` uses a slightly modified version of the ADAM optimiser of Kingma & Ba (2014).


