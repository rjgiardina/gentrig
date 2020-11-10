# gentrig

This is a quick and dirty implementation of the two-parameter generalized trigonometric functions, also referred to as Ateb function or Linqvist functions. Its main purpose is to demonstrate the behavior of these functions and how changes to their parameters or initial conditions can alter their expression. I will continue to update this project as time goes on. Currently, I am looking into using the scipy implementation of the inverse of the incomplete Beta function, scipy.special.betaincinv, and whether or not this approach offers any practical computational advantages over using the more general scipy.integrate.solve_ivp function. The solve_ivp method was surprisingly robust in this little program. I expected to have issues when the exponents p or q were set equal to one, but it handled this marvelously. There were potentially some fairly standard convergence issues with this particular choice of exponent, but solve_ivp handled them deftly and numerically output the expected model behaviors, or at least an acceptable approximation of them.

The autonomous system of two first order ordinary differential equations we are modeling:

y' =  |x|^(p-1) sign(x), x(0) = x_0

x' = -|y|^(q-1) sign(y), y(0) = y_0

for '=d/dt, with phase portrait defined by


|x|^p / p + |y|^q / q = |x_0|^p / p + |y_0|^q / q

which we refer as a generalized ellipse, and periodic solution expressions

x(t) and y(t)

The degenerate case of this expression when p = q = 2 is the linear first order system of two equations uniquely solved by sine and cosine whose phase portrait is the unit circle x^2 + y^2. These functions are periodic in their expression and their phase plane is defined by a set of generalized ellipses which form a minimal covering of the plane R^2.

In the future as this project progresses, I hope to implement more general expressions in these systems than simple power-law relations, and hopefully make their use and calculation more efficient. These fascinating little functions are fun to play with and explore how nonlinearities are expressed in their behavior. The current program is very simplistic. Suggestions are welcome. Come back periodically to see how it is progressing.
