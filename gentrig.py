
from scipy.special import beta, betainc

from scipy.integrate import solve_ivp, cumtrapz

from numpy import sign, absolute, arange, meshgrid

from matplotlib import pyplot as plt

"""
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        TWO-PARAMETER (twopara) CLASS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================

The two-parameter (twopara) class defines the periodic differential system of
autonomous expressions which are specific power-law based generalizations of
the sine and cosine relationships often referred to as Generalized Trigonometric
functions, Ateb functions, or Lindqvist functions. This class models the
autonomous system of two first-order differential equations

                    y' = a*sign(x)*|x|^(p-1),   x(0) = x_0              (1)
                    x' = -b*sign(y)*|y|^(q-1),   y(0) = y_0

for p,q > 1 and a,b > 0.

    twopara( p , q , initial_conditions , coef=None )

    exponents:

            The exponents are given individually for each of p and q described
            in the differential system (1).

    initial_conditions:

        The initial conditions argument is a tuple which provides the initial
        conditions x_0 and y_0 necessary to solve the system of two first-order
        equations (1). It is required in order to define the class.

                initial_conditions = (x_0, y_0)

    coef:

        The coef argument is a tuple which provides the coefficients a and b in
        the differential system (1). It is not a required argument. If coef is
        not provided it is assumed that a=b=1.

                coef = (a, b)   or  (1, 1)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    FUNCTIONS OF THE TWO-PARAMETER CLASS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two-parameter class contains several functions which performs operations
with respect to the system in (1); namely, its inherent behaviors, qualities,
and representation.

    s_max()

        This function returns the maximum/minimum extrema of the generalized
        sine function - i.e.,

                s_max = ((q / a) * (b * |x_0|^p / p + a * |y_0|^q / q))^(1 / q)

    c_max()

        This function returns the maximum/minimum extrema of the generalized
        cosine function - i.e.,

                c_max = ((p / b) * (b * |x_0|^p / p + a * |y_0|^q / q))^(1 / p)

    sum_constant()

        This function returns the total constant equivalent of the generalized
        ellipse - i.e.,

                c = a * |x_0|^p / p + b * |y_0|^q / q

    frequency()

        This function returns the frequency of the parameterized functions
        described by the system (1). The frequency is a function of all given
        parameters of the system.

                f = (p^(1 - 1/p) / q^(1/q)) * (|x_0|^p / p + |y_0|^q / q)^(1 - 1/p - 1/q)

    period()

        This function returns the period of the parameterized functions
        described by the system (1). The period changes depending on the choice
        of initial conditions, except in the case of Holder equality systems
        when the period is invariant, such as the degenerate case of the
        classical trigonometric functions when the period is 2pi. The period
        functions uses the Beta function and the inherent symmetry of the
        behaviors of the system between the four quadrants to return the value
        of the period

                P = 4 * Beta(1/p,1/q) / (q * f)

    phase()

        This function returns the phase shift of the parameterized functions
        described by the system (1) for some set of initial conditions.

    func()

        This function defines the differential system (1).

    para_func( tspan = None, max_step = None, solver = None )

        This functions solves the differential system (1) using solve_ivp and
        returns the two parametrized functions x(t) and y(t) which can be
        thought of as the generalized cosine and generalized sine functions,
        respectively. This function may take one argument for the tspan value
        necessary in solve_ivp. If no value is given, the default value shall be
        one full period of the system as calculated by period().

                                                    default:

                tspan =     USER_DEFINED        or      period()
                max_step =  USER_DEFINED        or      tspan / 100
                solver =    USER_DEFINED        or      'RK45'


================================================================================
================================================================================
"""

class twopara:

    #The indexed ordering of all values is intended to preserve that of traditional
    #spatial coordinates, i.e.
    #
    #          self.i_c = [self.i_c[0], self.i_c[1]] = (x_0, y_0)
    #   or
    #          y[0] -> generalized cosine, which traditionaly corresponds to x
    #          y[1] -> generalized sine, which traditionaly corresponds to y
    #

    def __init__(self, p, q, i_c, coef = None):

        self.p = p

        self.q = q

        self.i_c = i_c

        self.coef = coef or [1,1]

    #For reference, this class concerns the power-law based system of two
    #first order autonomous equations
    #
    #   dy[0] = self.coef[1]*sign(y[1])*|y[1]|^(self.p - 1),   y[1](0) = self.i_c[0]
    #   dy[1] = -self.coef[0]*sign(y[0])*|y[0]|^(self.q - 1),   y[0](0) = self.i_c[1]
    #

    #This function returns the maximum value of the generalized sine function y[0]
    def s_max(self):

        return ((self.q / self.coef[1]) * (self.coef[0] * absolute(self.i_c[0])**self.p / self.p + self.coef[1] * absolute(self.i_c[1])**self.q / self.q))**(1 / self.q)

    #This function returns the maximum value of the generalized cosine function y[1]
    def c_max(self):

        return ((self.p / self.coef[0]) * (self.coef[0] * absolute(self.i_c[0])**self.p / self.p + self.coef[1] * absolute(self.i_c[1])**self.q / self.q))**(1 / self.p)

    def sum_constant(self):

        return self.coef[0] * absolute(self.i_c[0])**self.p / self.p + self.coef[1] * absolute(self.i_c[1])**self.q / self.q

    #This function returns the frequency of the system
    def frequency(self):

        return (self.coef[0] * (self.p / self.coef[0])**(1 - 1/self.p) / (self.q / self.coef[1])**(1/self.q)) * ((self.coef[0] / self.p) * absolute(self.i_c[0])**self.p + (self.coef[1] / self.q) * absolute(self.i_c[1])**self.q)**(1 - 1/self.p - 1/self.q)

    #This function returns the period of system using the Euler Beta function
    def period(self):

        return 4 * beta(1/self.p, 1/self.q) / (self.q * self.frequency())

    #This function returns the phase shift using the incomplete Beta function
    def phase(self):

        if self.i_c[0] == 0:

            return 0.0

        else:

            UPPER_BOUND = self.i_c[1] / ((self.coef[0] / self.p) * absolute(self.i_c[0])**self.p + (self.coef[1] / self.q) * absolute(self.i_c[1])**self.q)**(1 - 1/self.p)

            return betainc(UPPER_BOUND, 1/self.p, 1/self.q) / (self.q * self.frequency())

    #This function returns the expressions of the derivatives dy[i] of each
    #respective function y[i] as defined by y[j]
    def func(self, t, y):

        return [ sign(y[1]) * absolute(y[1])**(self.p - 1), -sign(y[0]) * absolute(y[0])**(self.q - 1) ]

    #This function returns the solution to the system using solve_ivp. Because
    #we adhere to the spatial coordinate ordering, self.i_c = (x_0, y_0), we
    #must reverse this order in solve_ivp because it indexes by the given
    #ordering of dy. It returns the solution set such that it respects the
    #traditional Cartesian ordering, i.e. - t, x(t), y(t)
    def para_func(self, tspan = None, max_step = None, solver = None):

        tspan = tspan or self.period()

        max_step = max_step or tspan / 100

        solver = solver or 'RK45'

        SOLUTION = solve_ivp(self.func, [0, tspan], [self.i_c[1], self.i_c[0]], max_step = max_step, method = solver)

        return SOLUTION.t, SOLUTION.y[1], SOLUTION.y[0]


"""
================================================================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                PARAPLOT CLASS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================================================================================

The paraplot class plots and organizes the various output from the twopara
class.

    paraplot( twopara )

    twopara:

        The twopara argument is an object from the twopara class which caries
        all of the associated values for displaying the appropriate plots.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    FUNCTIONS OF THE PARAPLOT CLASS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    gen_ellipse()

        This function returns a plot of the generalized ellipse with respect to
        the system described by (1). The generalized ellipse is the phase
        portrait representation of the system for some set of parameters.

            a * |x|^p / p + b * |y|^q / q = a * |x_0|^p / p + b * |y_0|^q / q

    para_plot( single_plot = None )

        This functions returns a plot of the generalized trigonometric
        functions with respect to solving (1). The default is to plot both
        parameterized functions on the same graph, but the user may select only
        one plot or the other by passing the argument 's' for the generalized
        sine or 'c' for the generalized cosine using the single_plot parameter

            single_plot = s        or      c       or      None for BOTH


================================================================================
================================================================================
"""

class paraplot:

    def __init__(self, twopara_object):

        self.twopara_object = twopara_object


    #This function plots the generalized ellipse which defines the phase
    #portrait for the specific set of initial conditions, which are plotted as
    #a phase shifted line along the circumference.
    def gen_ellipse(self, grid_step = None):

        grid_step = grid_step or [self.twopara_object.c_max() / 1000, self.twopara_object.s_max() / 1000]

        x = arange(-1.15*self.twopara_object.c_max(), 1.15*self.twopara_object.c_max(), grid_step[0])

        y = arange(-1.15*self.twopara_object.s_max(), 1.15*self.twopara_object.s_max(), grid_step[1])

        X, Y = meshgrid(x, y)

        Z = self.twopara_object.coef[0] * absolute(X)**self.twopara_object.p / self.twopara_object.p + self.twopara_object.coef[1] * absolute(Y)**self.twopara_object.q / self.twopara_object.q

        level = self.twopara_object.coef[0] * absolute(self.twopara_object.i_c[0])**self.twopara_object.p / self.twopara_object.p + self.twopara_object.coef[1] * absolute(self.twopara_object.i_c[1])**self.twopara_object.q / self.twopara_object.q

        plt.contour(X, Y, Z, colors='black', levels = [level])
        plt.plot([0,self.twopara_object.i_c[0]], [0,self.twopara_object.i_c[1]], 'k:', self.twopara_object.i_c[0], self.twopara_object.i_c[1], 'ko')
        plt.title('p = ' + str(self.twopara_object.p) + ', q = ' + str(self.twopara_object.q) + '\ncoefficients a = ' + str(self.twopara_object.coef[0]) + ' and b = ' + str(self.twopara_object.coef[1]) + ', initial conditions = ' + '(' + str(self.twopara_object.i_c[0]) + ',' + str(self.twopara_object.i_c[1]) + ')')
        plt.text(1.1*self.twopara_object.i_c[0], 1.1*self.twopara_object.i_c[1], '(' + str(self.twopara_object.i_c[0]) + ',' + str(self.twopara_object.i_c[1]) + ')')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    #This function plots the generalized sine and cosine values of the specified
    #system. Default plots both. Changing single_plot to 's' or 'c' plots each
    #individually.
    def para_plot(self, tspan = None, max_step = None, single_plot = None):

        t, c, s = self.twopara_object.para_func(tspan, max_step)

        if single_plot == 's':

            plt.plot(t, s, 'k')
            plt.title('Generalized Sine function')
            plt.ylabel('y')

        elif single_plot == 'c':

            plt.plot(t, c, 'k--')
            plt.title('Generalized Cosine function')
            plt.ylabel('x')

        else:

            plt.plot(t, s, 'k', t, c, 'k--')
            plt.title('Generalized Sine and Cosine functions')
            plt.legend(['gen sine', 'gen cosine'], loc=3)

        plt.xlabel('t')
        plt.show()
