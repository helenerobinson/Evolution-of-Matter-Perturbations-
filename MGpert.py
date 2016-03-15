from __future__ import division
import math, random
import numpy as np
# from numpy.random import uniform, seed
from scipy import integrate
from scipy.integrate import quad
from matplotlib import pyplot as plt
from copy import deepcopy


# perturbation class
class MGpert():

	delta_i = 0.01
	y_i = 0.01
	x_i = -4.6

	As_norm =  0.829**2 / 4.666788092695281e-07 # inputed this value of I_z[-1] by hand
	As = 1.0
	nsp = 0.958
	R = 8.0 #R is in Mpc/h
	h = 0.7

	# This is where the LCDM simulation is stored
	LCDM = 0

	# # LCDM datapoints
	# dataPoints = [{ 'z': 0.067, 'f': 0.42, 'e': 0.06}, { 'z': 0.17, 'f': 0.51, 'e': 0.06}, { 'z': 0.22, 'f': 0.42, 'e': 0.07},
	# 			  { 'z': 0.25, 'f': 0.35, 'e': 0.06 }, { 'z': 0.37, 'f': 0.46, 'e': 0.04}, { 'z': 0.41, 'f': 0.45, 'e': 0.04},
	# 			  { 'z': 0.57, 'f': 0.43, 'e': 0.03 }, { 'z': 0.6,  'f': 0.43, 'e': 0.04}, { 'z': 0.78, 'f': 0.38, 'e': 0.04}]

	# # # from planck
	# dataPoints = [{ 'z': 0.067, 'f': 0.423, 'e': 0.055}, { 'z': 0.15, 'f': 0.49, 'e': 0.15}, { 'z': 0.3, 'f': 0.49, 'e': 0.08},
	# 			  { 'z': 0.44, 'f': 0.413, 'e': 0.08 }, { 'z': 0.57, 'f': 0.447, 'e': 0.028}, { 'z': 0.6, 'f': 0.39, 'e': 0.063},
	# 			  { 'z': 0.73, 'f': 0.437, 'e': 0.072 }, { 'z': 0.8,  'f': 0.47, 'e': 0.08}]

	# fsigma8 data points
	dataPoints = [{ 'z': 0.067, 'f': 0.423, 'e': 0.055}, { 'z': 0.17, 'f': 0.51, 'e': 0.06},
				 { 'z': 0.22, 'f': 0.42, 'e': 0.07}, { 'z': 0.25, 'f': 0.39, 'e': 0.05}, { 'z': 0.37, 'f': 0.4302, 'e': 0.0378},
				 { 'z': 0.41, 'f': 0.45, 'e': 0.04}, { 'z': 0.57, 'f': 0.43, 'e': 0.03 }, { 'z': 0.6,  'f': 0.43, 'e': 0.04},
				 { 'z': 0.78, 'f': 0.38, 'e': 0.04}, { 'z': 0.8,  'f': 0.47, 'e': 0.08}]

	# constructor default is LCDM
	def __init__(self, mu0 = 0, alpha_t0 = 0 , alpha_b0 = 0 , q0 = 0):

		# mandatory variables
		self.mu0 = mu0
		self.alpha_t0 = alpha_t0
		self.alpha_b0 = alpha_b0
		self.q0 = q0

		self.Storage = {}
		self.omega_m = 0.308
		self.omega_v = 1 - self.omega_m
		self.dx = 0.01 	#step size

		# This boolean controls if the sumulation is physical
		self.physical = True

		# Stores the chisquared value of the run
		self.chi = 0

		# This automatically creats a LCDM object the first time this constructor is called
		if MGpert.LCDM == 0:
			MGpert.LCDM = 1
			MGpert.LCDM = MGpert()
			MGpert.LCDM.runSimulation()
			#print 'hi 1'

		# Runs the integration
		# Makes sure LCDM only runs once
		if MGpert.LCDM  != 1:
			self.runSimulation()


	# create the arrays for the integration method
	def createArrays(self):

		self.delta = [MGpert.delta_i]
		self.y = [MGpert.y_i] 		# y = delta'
		self.X = [MGpert.x_i] 		# corresponds to a = 0.01 where X = log a
		self.scalefactor = [0.01]

		self.mu = [1]
		self.growth = [1] 	#growth factor

		self.z = [(1 / 0.01) - 1]
		self.I_z = [1e-8]

		self.Omega_m_a = [0]
		self.Omega_v_a = [0]

		self.alpha_b = [0]
		self.alpha_b_derivative = [0]
		self.alpha_t = [0]
		self.alpha_m = [0]
		self.M2 = [1]
		self.cs2 = [0] #sound speed
		self.alpha_k0_min = -10**5

		#used planck result of sigma8
		self.sigma8 = [0.8]
		self.f_sigma8 = [0.8]

		self.v = 10.0 - 2.0 - 1 ##for reduced chi v = N - M - 1

		#defining an end point to the for loop - when a = 1 (today)
		self.lim = math.ceil(( 0.0 - self.X[0] )/ self.dx)


	# This checks the physical constraints
	def checkConstraints(self,i):

		#contraints on M2
		if self.M2[i] < 0:
			self.physical = False

		#constraints on alphat
		if self.alpha_t[i] < -1 or self.alpha_t[i] > 0:
			self.physical = False

		#adding constraints on cs2
		if self.cs2[i] < 0: 
			self.physical = False

		#constraints on mu
		if self.mu[i] < 0:
			self.physical = False



	# integration to determine sigma8
	def intergrand_redshift(self, x, i):

		E = self.omega_m * MGpert.h #Gamma = Omega_m * h

		return ( (MGpert.As * x **(2 + MGpert.nsp)  * ( math.log(1 + (2.34 * x) / E) / (   ((2.34 * x) / E)  * 
			(1  +   (3.89 * x) / E   +  ( (16.2 * x) / E)**2   +  ((5.47 * x) / E)**3   +   ((6.71 * x) / E)**4  )** 0.25 ) )**2 ) 
			 *  (  3 * ( math.sin(x * MGpert.R/ math.exp(self.X[i])  )  -   (x * (MGpert.R / math.exp(self.X[i])) * math.cos(x * MGpert.R / math.exp(self.X[i])) ) ) / (x**3 * math.exp(self.X[i])**(-3)* MGpert.R**3  )  )**2 ) / (2 * math.pi * math.pi)


	# computes chi squared as we go
	def chiSquare(self,i):
		for data in MGpert.dataPoints:
			if self.z[i-1] >= data['z'] >= self.z[i]:
				self.chi += (data['f'] - self.f_sigma8[i])**2 / data['e']**2
			
		

	# This is the main integration method
	def runSimulation(self):

		# initalise the arrays
		self.createArrays()

		for i in range(1, int(self.lim) + 1):
			# updates X at each step
			self.X.append(self.X[0] + i * self.dx)

			# varying the Omega parameters wrt scalefactor
			self.Omega_m_a.append( ( self.omega_m * math.exp(self.X[i]*-3) ) / ( ( self.omega_m * math.exp(self.X[i]*-3) ) +  self.omega_v  ) )
			self.Omega_v_a.append( self.omega_v   / ( self.omega_v   +  (self.omega_m * math.exp(self.X[i]*-3) ) ) ) 

			#evolving EFT parameters
			self.alpha_b.append(self.alpha_b0 * ( self.Omega_v_a[i] / self.omega_v ) )
			self.alpha_t.append(self.alpha_t0 * ( self.Omega_v_a[i] / self.omega_v ) )
			self.M2.append(1 + self.q0 *  ( self.Omega_v_a[i] / self.omega_v ))

			self.alpha_m.append( (math.log(self.M2[i]) - math.log(self.M2[i-1])) / self.dx ) 
			self.alpha_b_derivative.append( (self.alpha_b0 / self.omega_v ) * ( (self.Omega_v_a[i] -  self.Omega_v_a[i-1]) / self.dx) ) 

			# H'/H
			H_derivative_over_H = - ((1.0/2.0) *((3.0*self.omega_m*math.exp(-3*self.X[i-1])   / (self.omega_m*math.exp(-3*self.X[i-1]) + self.omega_v) ) ) )  

			# sound of speed
			self.cs2.append(- self.alpha_b_derivative[i] - (1 + self.alpha_t[i] )* (1 + self.alpha_b[i] )**2 + ( 1 + self.alpha_m[i] - H_derivative_over_H )*(1 + self.alpha_b[i]) - (3* self.Omega_m_a[i]) / (2 * self.M2[i]))  
			self.mu.append( ( (self.alpha_b[i]*(1 + self.alpha_t[i]) - self.alpha_m[i] + self.alpha_t[i] )**2 + (1 + self.alpha_t[i])*self.cs2[i] ) / (self.cs2[i] * self.M2[i] ) ) 

			# checks the constrints
			self.checkConstraints(i)

			A = (2.0 + H_derivative_over_H )
			B = self.mu[i-1] * ((3.0/2.0) *((self.omega_m*math.exp(-3 * self.X[i-1]))/(self.omega_m*math.exp(-3*self.X[i-1]) + self.omega_v  )))


			# Euler intergration method for my second order differential equation
			# updating delta
			self.delta.append(self.delta[i-1] + (self.y[i-1] * self.dx) )
			# updating delta'
			self.y.append((B * self.delta[i-1] - A * self.y[i-1]) * self.dx + self.y[i-1])

			# delta' / delta, also known as f
			self.growth.append(self.y[i] / self.delta[i])

			# calculating a redshift
			self.z.append( (1.0 / math.exp(self.X[i]))  - 1.0  )
	
			# create your scale factor array
			self.scalefactor.append( math.exp(self.X[i]) )

			# In the case that this is not LCDM we need to normalise the values
			if self.mu0 != 0 or self.alpha_t0 != 0 or self.alpha_b0 != 0 or self.q0 != 0:
				self.sigma8.append( self.delta[i] * MGpert.LCDM.sigma8[i] / MGpert.LCDM.delta[i])

			# This is the LCDM case
			else:
				self.I_z.append( quad(self.intergrand_redshift, 0, 100, args = (i) ))
				self.sigma8.append( math.sqrt(MGpert.As_norm * self.I_z[i][0]) )

			# calculating fsigma8
			self.f_sigma8.append(self.growth[i] * self.sigma8[i])

			# calculate the chisquared
			self.chiSquare(i)
			
	
	# compute the error on a certain run - only used for LCDM fsimga8 
	def error(self,d_Om, d_sigma8):
		upper = []
		lower = []
		upperOm_error = deepcopy(self)
		lowerOm_error =deepcopy(self)
		uppersigma8_error = deepcopy(self)
		lowersigma8_error = deepcopy(self)

		upperOm = deepcopy(self)
		lowerOm = deepcopy(self)
		uppersigma8 = deepcopy(self)
		lowersigma8 = deepcopy(self)

		upperOm.omega_m += 0.0001 # d_Om
		lowerOm.omega_m -=0.0001 #d_Om

		for i in range(0,len(self.scalefactor)):
			uppersigma8.sigma8[i] +=0.0001 #d_sigma8
			lowersigma8.sigma8[i] -= 0.0001 #d_sigma8

		upperOm.runSimulation()
		lowerOm.runSimulation()
		uppersigma8.runSimulation()
		lowersigma8.runSimulation()

		for i in range(0,len(self.scalefactor)):
			upperOm_error.f_sigma8[i] = ((upperOm.f_sigma8[i] - self.f_sigma8[i] ) / 0.0001 ) **2 *d_Om**2
			lowerOm_error.f_sigma8[i] = ((lowerOm.f_sigma8[i] - self.f_sigma8[i] ) / 0.0001 ) **2 *d_Om**2
			uppersigma8_error.f_sigma8[i] = ((uppersigma8.f_sigma8[i] - self.f_sigma8[i]) / 0.0001) **2 *d_sigma8**2
			lowersigma8_error.f_sigma8[i] = ((uppersigma8.f_sigma8[i] - self.f_sigma8[i] ) / 0.0001) **2 *d_sigma8**2


			upper.append(upperOm.f_sigma8[i] + math.sqrt(upperOm_error.f_sigma8[i] + uppersigma8_error.f_sigma8[i] ))
			lower.append(lowerOm.f_sigma8[i] - math.sqrt( lowerOm_error.f_sigma8[i] + lowersigma8_error.f_sigma8[i] )) 

		return upper, lower




