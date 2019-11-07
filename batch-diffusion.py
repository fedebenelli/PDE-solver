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

# Funciones de transferencia de masa
def batch(X, t):
    '''
    '''
    x = X[0]
    y = X[1]
    dxdt = K * (eqX(y)-x) / (1-eps)
    dydt = -K * (eqX(y) - x) / eps
    return [dxdt, dydt]


nt = 10000
t = np.linspace(0,tau, nt)

x0 = porcentajeAC*dens
y0 = 0

sol0 = odeint(batch, [x0,y0], t)
sol1 = odeint(batch, [sol0[-1][0],y0], t)
sol2 = odeint(batch, [sol1[-1][0],y0], t)
sol3 = odeint(batch, [sol2[-1][0],y0], t)

lista = []

for i in range(0,10):
    sol1 = odeint(batch, [x0, sol2[-1][1]], t)
    sol2 = odeint(batch, [sol1[-1][0], sol3[-1][1]], t)
    sol3 = odeint(batch, [sol2[-1][0], y0], t)
    lista.append(sol3[-1][1])


plt.plot(t, sol0, label='sol0')
plt.plot(t,sol1, label="sol1")
plt.plot(t,sol2, label="sol2")
plt.plot(t,sol3, label="sol3")
plt.title('Tres equipos en serie')
plt.legend()
plt.show()


print(f'''
Concentración final con un equipo: {sol0[-1][1]}
Concentración final con tres equipos: {sol1[-1][1]}
Mejora Porcentual con tres equipos: {(sol1[-1][1]-sol0[-1][1])/sol0[-1][1]}
''')

lista = []
for i in range(0,10):
    sol1 = odeint(batch, [x0, sol2[-1][1]], t)
    sol2 = odeint(batch, [sol1[-1][0], y0], t)
    lista.append(sol1[-1][1])


plt.plot(t, sol0, label='sol0')
plt.plot(t,sol1, label="sol1")
plt.plot(t,sol2, label="sol2")
plt.title('Dos equipos en serie')
plt.legend()
plt.show()

plt.plot(lista)
plt.show()


print(f'''
Concentración final con dos equipos: {sol1[-1][1]}
Mejora Porcentual con dos equipos: {(sol1[-1][1]-sol0[-1][1])/sol0[-1][1]}
''')