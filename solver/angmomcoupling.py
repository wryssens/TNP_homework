#!/usr/bin/env python3
import sympy
import functools
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner     import wigner_3j

@functools.cache
def clebsch(j1,m1,j2,m2,j3,m3):
  """
     Calculate the Clebsch-Gordan coefficient  <j1 m1 j2 m2 | j3 m3> with the Sympy routines.
     Careful:
      - there are factors of two for j1,m1,j2,m2 - these indices correspond to single-particle states.
      - there are NO factors of two for j3,m3 - these indices correspond to two-body states

     Input:
        j1/2 : TWICE the angular momentum of the single-particle states
        m1/2 : TWICE the projection of the angular momentum of the single-particle states
        j3   : the coupled angular momentum of the single-particle states
        m3   : the projection of the coupled angular momentum of the single-particle states

     Output:
        CG : the Clebsch-Gordan coefficient

     Note: this function is cached by functools, such that repeated calls with the same arguments
           don't require recalculation.

  """

  return CG(  sympy.S(j1)/2, sympy.S(m1)/2, \
              sympy.S(j2)/2, sympy.S(m2)/2, \
              sympy.S(j3)  , sympy.S(m3)).doit().evalf()

@functools.cache
def wigner(j1,j2,j3,m1,m2,m3):
  """
     Calculate the Wigner 3j symbol with a Sympy routine

     Careful:
      - there are factors of two for j1,m1,j3,m3 - these indices correspond to single-particle states.
      - there are NO factors of two for j2,m2 - these indices correspond to the angular momentum of a multipole operator!

     Input:
        j1/3 : TWICE the angular momentum of the single-particle states
        m1/3 : TWICE the projection of the angular momentum of the single-particle states
        j2   : the angular momentum of a multipole operator
        m2   : the projection of the angular momentum of a multipole operator

     Output:
        W    : the 3j symbol

     Note: this function is cached by functools, such that repeated calls with the same arguments
           don't require recalculation.

  """

  return wigner_3j(  sympy.S(j1)/2, sympy.S(j2), sympy.S(j3)/2,
                     sympy.S(m1)/2, sympy.S(m2), sympy.S(m3)/2).doit().evalf()
