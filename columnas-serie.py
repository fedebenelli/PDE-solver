import numpy as np
import matplotlib.pyplot as plt
import math as math
import pandas as pd
from scipy.integrate import simps
from scipy.integrate import odeint



kb = 1.38*10**-23
pi = np.pi

# Estimación viscosidades
T = 25 + 273

# Definición de propiedades
porcentajeAC = 0.06
dens = 557.82
densb = 180
densL = 935.69
Deff = (1.3*(10**-10))/60       
phi = 2.012453362               # Factor de forma
a = 0.000706498099              # Longitud característica
ap = 3.362*10**3                # Área equivalente
Deq = 6/ap

# Propiedades estimadas
viscH2O = math.e**(-52.843 + 3703.6/T + 5.866*math.log(T) - 5.98*10**(-29)*(T)**10)
viscEtOH = math.e**(7.875+781.98/T -3.0418*math.log(T)) 
visc = 0.01
rad = math.pow(326.5*(3./(4.*math.pi)), 1./3.)                                          # radio molecular de van der waals
Dab = kb*T/(6*math.pi*visc*rad)                                                         # Ecuación de Einstein para difusividad

# Definición de parámetros de diseño
eps = 1.-densb/dens     # Porosidad del lecho
L = 4.421               # Longitud del equipo
Dc = 0.2                # Diámetro del equipo
A = math.pi*(Dc/2)**2   # Área del equipo
S = 1.072*10**-6        # Flujo volumétrico de sólidos (m³/s)
nu = 2.249*10**-6       # Flujo volumétrico de solvente (m³/s)
tau = 24*3600           # Tiempo máximo a calcular en la simulación

# Estimación de parámetros
uz = nu/(A)                                         # Velocidad lineal de solvente
Re = uz*Deq*densL/(visc*eps)                        # Número de Reynolds
Pe = 0.2/eps + 0.011/eps + math.pow(eps*Re, 0.48)   # Número de Peclet
Sc = visc/(densL*Dab)                               # Numero de Schmidt
Sh = 2+1.1*math.pow(Sc, 0.33)*math.pow(Re, 0.6)     # Número de Sherwood
Dax = Deq*uz/(eps*Pe)                               # Dispersión axial
kL = Sh*Dab/a                                       # Transferencia de masa en fase líquida

print(f'''Parámetros:
uz: {uz}
Re: {Re}
Pe: {Pe}
Sc: {Sc}
Sh: {Sh}
Dax: {Dax}
kL: {kL}
''')

# Cálculo de tiempos de residencia para el líquido y el sólido
resTime = A*L*eps/nu
resTimeS = A*L*(1-eps)/S

print('\n\n')
print(f'Masa de sólido por hora: {round(S*3600*dens,2)} Kg/h')
print(f'Volumen de solvente por hora: {round(nu*3600*1000,2)} L/h')

print(f'Tiempo de residencia de líquido:  {round(resTime/3600,2)}')
print(f'Tiempo de residencia de sólido: {round(resTimeS/3600,2)}')

# Definición de la constante global de transferencia de masa
K = 8.07*10**-9
K = K*ap



'__Definición de funciones__'


def find_nearest(array, value):
    '''
    Recibe un array y un valor; devuelve el valor más cercano al recibido
    '''
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def find_nearest_pos(array, value):
    '''
    Recibe un array y un valor; devuelve el índice del elemento con el valor más cercano al recibido
    '''
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

# Función de equilibrio
def eqX(y):
    '''
    Devuelve concentración de equilibrio en fase sólida a partir de una concentración en fase líquida
    '''
    KeqX = [15.80011045, 32.02907632, 5.735438305, 0.053726466, 1.972131456]
    eqX = KeqX[0]*KeqX[1]*y/(1+KeqX[1]*y)+KeqX[2] * \
        KeqX[3]*(y-KeqX[4])/(1-KeqX[3]*(y-KeqX[4]))
    return eqX


index = 0
# Funciones de transferencia de masa
def column(F, t, Yini):
    '''
    '''
    x = F[::2]
    y = F[1::2]
    global index
    #y[0] = Yini[index]
    print(index)
    dFdt = np.empty_like(F)

    dxdt = dFdt[::2]
    dydt = dFdt[1::2]
    
    '_________________________________________________________________________________________________________________________'

     # A lo largo de todo el extractor la variación de concentración en el sólido en función del tiempo es igual a la
     # ecuación de transferencia de masa

    dxdt[:] = K * (eqX(y[:]) - x[:]) / (1-eps)

    '_________________________________________________________________________________________________________________________'

     # En z = 0 ingresa el solvente, por lo que se considera que no hay acumulación en ese punto

    dydt[0] = 0
    
     # A lo largo del extractor la variación se puede considerar la ecuación general
    difusion = eps * Dax * np.diff(y[:], 2)/dz**2
    conveccion = uz * np.diff(y[:-1], 1)/dz
    transferencia = K * (eqX(y[1:-1]) - x[1:-1])

    dydt[1:-1] = (difusion - conveccion - transferencia)/eps

    dydt[-1] = (eps*Dax*(2*y[-2]-2*y[-1]) - uz *
                (y[-1]-y[-2])/dz - K*(eqX(y[-1]) - x[-1]))/eps

    '_________________________________________________________________________________________________________________________'

    index += 1
    return dFdt


# Número de puntos
nz = 100
nt = tau

# Condiciones iniciales
eqLiq = 16.03
F0 = np.ones(2*nz)

F0[::2] = porcentajeAC*dens
F0[1::2] = 0

Y0in = np.empty_like(F0[1::2])
Y0in[:] = 0


# Creo conjuntos de tiempo y espacio
t = np.linspace(0, tau, nt)
Z = np.linspace(0, L, nz)
dz = Z[1] - Z[0]
dt = t[1] - t[0]


# Resolución de ecuaciones
equip = 'SC'

print(len(Y0in))

sol0 = odeint(column, F0, t, args = (Y0in,), ml=1, mu=2)

X0 = sol0[:, ::2]
Y0 = sol0[:, 1::2]
Y0out = Y0[:, -1]

sol1 = odeint(column, F0, t, args = (Y0out,), ml=1, mu=2)

X1 = sol1[:,::2]
Y1 = sol1[:,1::2]
Y1out = Y1[:,-1]


plt.plot(t, Y0[:, -1])
plt.plot(t, Y1[:, -1])
plt.show()