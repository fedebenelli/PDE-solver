# %%
import numpy as np
import matplotlib.pyplot as plt
import math as math
import pandas as pd
from scipy.integrate import simps
from scipy.integrate import odeint
from scipy.misc import derivative

kb = 1.38*10**-23 # Constante de Boltzmann
e = math.e
pi = math.pi

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
eps = 1.-densb/dens             # Porosidad del lecho

# Propiedades estimadas
viscH2O = e**(-52.843 + 3703.6/T + 5.866*math.log(T) - 5.98*10**(-29)*(T)**10)
viscEtOH = e**(7.875+781.98/T -3.0418*math.log(T)) 
visc = 0.01
rad = math.pow(326.5*(3/(4*pi)), 1/3)*10**(-10)                                             # radio molecular de van der waals
Dab = kb*T/(6*pi*visc*rad)                                                                  # Ecuación de Einstein para difusividad

# Definición de parámetros de diseño
L = 4.421               # Longitud del equipo
Dc = 0.2                # Diámetro del equipo
A = math.pi*(Dc/2)**2   # Área del equipo

# Definición de parámetros de proceso
S = 1.072*10**-6        # Flujo volumétrico de sólidos (m³/s)
nu = 2.249*10**-6       # Flujo volumétrico de solvente (m³/s)
tau = 24*3600           # Tiempo máximo a calcular en la simulación

# Estimación de números adimensionales y coeficientes
uz = nu/(A)                                         # Velocidad lineal de solvente
Re = uz*Deq*densL/(visc*eps)                        # Número de Reynolds
Pe = 0.2/eps + 0.011/eps + math.pow(eps*Re, 0.48)   # Número de Peclet
Sc = visc/(densL*Dab)                               # Numero de Schmidt
ShL = 2+1.1*math.pow(Sc, 0.33)*math.pow(Re, 0.6)    # Número de Sherwood
Dax = Deq*uz/(eps*Pe)                               # Dispersión axial
kL = ShL*Dab/a                                      # Transferencia de masa en fase líquida

# Definición de la constante global de transferencia de masa
K = 8.07*10**-9
K = K*ap

# Número de Biot CORREGIR ESTOOAOSOAOSFINOAISASHFOIAHSFOI!!I!!··!=·)=!)=!)=!!!
Bi = K*a/Deff

# Cálculo de tiempos de residencia para el líquido y el sólido
resTime = A*L*eps/nu
resTimeS = A*L*(1-eps)/S

# %%

# %%
# Print de datos
print(f'''Propiedades:

''')

print(f'''Propiedades Estimadas:
Viscosidad: {visc} Pa.s
Radio van der waals: {rad}
''')

print(f'''Parámetros de diseño:
Largo total de equipo:      {L}     m
Diámetro de contacto:       {Dc}    m
''')

print(f'''Parámetros de proceso:
Flujo materia prima:{S*dens*3600} Kg/h
Flujo de solvente: {nu*3600}m³/h
''')


print(f'''Parámetros:
Velocidad lineal:   {uz}    m/s
Reynolds:           {Re}
Peclet:             {Pe}
Schmidt:            {Sc}
SherwoodL:          {ShL}    
Dax:                {Dax}   m²/s
Dab:                {Dab}   m²/s
kL:                 {kL}    m/s
''')

print('\n\n')
print(f'Masa de sólido por hora: {round(S*3600*dens,2)} Kg/h')
print(f'Volumen de solvente por hora: {round(nu*3600*1000,2)} L/h')

print(f'Tiempo de residencia de líquido:  {round(resTime/3600,2)}')
print(f'Tiempo de residencia de sólido: {round(resTimeS/3600,2)}')
# %%


# %%
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
    eqX = KeqX[0]*KeqX[1]*y/(1+KeqX[1]*y)+KeqX[2] * KeqX[3]*(y-KeqX[4])/(1-KeqX[3]*(y-KeqX[4]))
    return eqX

def m(y):
    '''
    Devuelve el valor del coeficiente de distribución para una determinada concentración en fase líquida
    '''
    m = eqX(y)/(y+1e6)

# Funciones de transferencia de masa
def batch(X, t):
    '''
    '''
    x = X[0]
    y = X[1]
    dxdt = K * (eqX(y)-x) / (1-eps)
    dydt = -K * (eqX(y) - x) / eps
    return [dxdt, dydt]

def column(F, t):
    '''
    '''
    x = F[::2]
    y = F[1::2]
    y[0] = 0

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

    return dFdt

def countercurrent(F, t):
    '''
    '''
    x = F[::2]
    y = F[1::2]

    x[-1] = porcentajeAC*dens
    y[0]= 0.01

    dFdt = np.empty_like(F)

    dxdt = dFdt[::2]
    dydt = dFdt[1::2]

    '_________________________________________________________________________________________________________________________'

    # En el punto de ingreso de sólidos (z = L) no hay variación en función del tiempo
    dxdt[-1] = 0
    
    # A lo largo del extractor se considera la ecuación general
    conveccionS = S/A * np.diff(x[1:], 1)/dz
    transferenciaS = K * (eqX(y[1:-1]) - x[1:-1])
    dxdt[1:-1] = (conveccionS + transferenciaS) / (1-eps)

    dxdt[0] = + (S/A *(x[1]-x[0])/dz +  K * (eqX(y[0])-x[0])) / (1-eps)

    '_________________________________________________________________________________________________________________________'

    # A lo largo del extractor se considera la ecuación general
    dydt[-1] =  ( -uz * (y[-1]-y[-2])/dz - K * (eqX(y[-1])-x[-1])) / (eps)
    
    difusionL = eps * Dax * np.diff(y[:], 2)/dz**2
    conveccionL = uz * np.diff(y[:-1], 1)/dz
    transferenciaL = K * (eqX(y[1:-1]) - x[1:-1])

    dydt[1:-1] = (difusionL - conveccionL - transferenciaL)/eps

    # En el punto de ingreso de solvente (z = 0) se considera que no hay acumulación
    dydt[0] = 0

    '_________________________________________________________________________________________________________________________'

    return dFdt
# %%

'____Cálculos____'

# %%

# Número de puntos
nz = 100
nt = tau

# Condiciones iniciales
eqLiq = 16.03
F0 = np.ones(2*nz)

F0[::2] = porcentajeAC*dens
F0[1::2] = 0

# Creo conjuntos de tiempo y espacio
t = np.linspace(0, tau, nt)
Z = np.linspace(0, L, nz)
dz = Z[1] - Z[0]
dt = t[1] - t[0]

# Resolución de ecuaciones
equip = 'CC'
sol = odeint(countercurrent, F0, t, ml=1, mu=2)

# %%


# Definición de un tiempo inicial para analizar y el máximo índice de z

tiniPos = find_nearest_pos(t, resTime)
zEnd = int(nz)

# Cambio conjunto de tiempo a horas
t = t/3600

# Separo los resultados en una variable para las concentraciones en el sólido y otra para las concentraciones en el líquido
X = sol[:, ::2]
Y = sol[:, 1::2]


'__Obtención de gráficos__'

#%%
# Concentración a la salida del extractor
Yin = Y[:, 1]
Yout = Y[:, -1]
Xin = X[:, -1]
Xout = X[:, 1]

# Obtengo tiempos a los cuales deseo extraerles información
tiempos = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8]

for i in range(0, len(tiempos)):
    tiempos[i] = int(nt*tiempos[i])
#%%

#%%
# Grafico concentraciones de fase líquida
for i in tiempos:
    plt.plot(Z[:zEnd], Y[i, :zEnd],
            label=f'tiempo: {round(t[i], 3)} horas')

plt.title('Fase Líquida a distintos tiempos')
plt.xlabel('Distancia (m)')
plt.ylabel('Concentración (Kg/m³)')
plt.legend()
plt.show()
#%%

#%%
# Grafico concentraciones de fase sólida
for i in tiempos:
    plt.plot(Z[:zEnd], X[i, :zEnd],
            label=f'tiempo: {round(t[i], 3)} horas')

plt.title('Fase Sólida a distintos tiempos')
plt.xlabel('Distancia (m)')
plt.ylabel('Concentración (Kg/m³)')
plt.legend()
plt.show()

#%%
# Grafico concentración a la salida del extractor
plt.plot(t[:], Yout, label='Fase líquida')
plt.title('Concentración a la salida')
plt.xlabel('Tiempo (s)')
plt.ylabel('Concentración (Kg/m³)')
plt.legend()
plt.show()

#%%
# Calculo y grafico rendimiento en función del tiempo
rend = 0
rendimientoL = []
rendimientoS = []

if equip == 'CC':
    for out in Yout[:]:
        rend = out*nu/(S*dens*porcentajeAC)
        rendimientoL.append(rend)
    for out in Xout[:]:
        rend = 1 - out/(dens*porcentajeAC)
        rendimientoS.append(rend)

    #plt.plot(t[:], rendimientoL, label='Rendimiento Líquido')
    plt.plot(t[:], rendimientoS, label='Rendimiento Sólido')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Concentración (Kg/m³)')

if equip == 'SC':
    for i in range(0,tiniPos):
        rendimientoL.append(0)
    for out in Yout[tiniPos:]:
        rend += out*nu*dt/(A*L*densb*porcentajeAC)
        rendimientoL.append(rend)
    plt.plot(t[:], rendimientoL, label='Rendimiento')
   
plt.legend()
plt.title('Rendimiento')
plt.xlabel('Tiempo (s)')
plt.ylabel('Rendimiento')
plt.show()
rendimiento = rendimientoL[-1]
#%%


print(f'''
Largo: {L} m
Diámetro: {Dc} m
Caudal: {round(nu*3600*1000,2)} L/h
Sólidos tratados: {round(S*dens*3600,2)} Kg/h

Concentración final: {round(Yout[-1],2)}
Rendimiento: {round(rendimiento,3)}
''')

# Guardado en Excel
dfY = pd.DataFrame(data=Y)
dfX = pd.DataFrame(data=X)

# print(dfY)

"""
with pd.ExcelWriter('data.xlsx') as writer:
    dfY.to_excel(writer, sheet_name='Sheet_name_1')
    dfX.to_excel(writer, sheet_name='Sheet_name_2')
print('Listo!')
"""
