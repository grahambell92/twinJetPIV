# File contains functions that return the ideally expanded Mach number and nozzle diameter.
# Written by Graham Bell 25/07/2016

import numpy as np
from scipy.interpolate import interp1d

def NPRToMachNum(NPR, gamma=1.4):
    # Rearranged isentropic pressure flow equation
    b = gamma/(gamma - 1)
    MachNum = np.sqrt((np.power(NPR,1/b) - 1)*2/(gamma - 1))
    return MachNum

def isentropicTemp(T0, Mach, gamma=1.4):
    b = (gamma-1)/2
    Tstatic = ((1 + b*(Mach)**2)**(-1))*T0
    return Tstatic

def speedOfSound(temp, gamma=1.4, R_sp=287.6):
    a = np.sqrt(gamma * R_sp * temp)
    return a

def isentropicVelocity(NPR=None, Mach=None, T0=(25+273.15), R_sp=287.6, gamma=1.4):
    # Pass in Mj or NPR
    if Mach is None:
        Mach = NPRToMachNum(NPR=NPR, gamma=gamma)
    Tstatic = isentropicTemp(T0=T0, Mach=Mach, gamma=gamma)
    # Speed of sound in the ideally expanded condition
    aStatic = speedOfSound(temp=Tstatic, gamma=gamma, R_sp=R_sp)
    vel = aStatic*Mach
    return vel

def isentropicDensity(rho0, Mach, gamma=1.4):
    e = (gamma-1)/2
    f = 1/(gamma-1)
    rhoStatic = rho0 * ((1.0+e*(Mach**2))**(-f))
    return rhoStatic

def criticalPressRatio(gamma=1.4):
    crit_press_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    return crit_press_ratio

def velocityToTemp(velocity, totalTemp=(25+273), gamma=1.4, sp_gasConst=287.6):
    # total temp is plenum chamber/supply temperature.
    local_temp = totalTemp - ((gamma-1.0)/2.0)*((velocity**2)/(gamma*sp_gasConst))
    return local_temp

def localMachNum(velocity, totalTemp=(25+273), gamma=1.4, sp_gasConst=287.6):
    # Isentropic temperature based on velocity
    local_temp = velocityToTemp(velocity=np.abs(velocity), totalTemp=totalTemp, gamma=gamma, sp_gasConst=sp_gasConst)
    # local speed of sound
    a_local = np.sqrt(gamma*sp_gasConst*local_temp)
    local_mach = velocity/a_local
    return local_mach

def throatVelocity(T0=(25+273), gamma=1.4, sp_gasConst=287.6):
    # What is the velocity at the throat?
    Mthroat = 1.0
    Tthroat = isentropicTemp(T0=T0, Mach=Mthroat)
    aThroat = np.sqrt(gamma*sp_gasConst*Tthroat)
    UThroat = aThroat*1
    return UThroat

def ideallyExpandedArea(AThroat, Mj, gamma=1.4):
    e = (gamma+1)/2
    g = (gamma - 1) / 2
    f = (gamma+1)/(2*(gamma-1))
    Aj = AThroat*(e**(-f) * ((1+g * Mj**2)**f)/Mj)
    return Aj

def ideallyExpandedDiameter(dThroat, Mj, gamma=1.4):
    Athroat = (np.pi*(dThroat**2))/4
    Aj = ideallyExpandedArea(AThroat=Athroat, Mj=Mj, gamma=gamma)
    Dj = np.sqrt((4*Aj)/np.pi)
    return Dj

def prandtlShockSpacing(MachNumber):
    return 1.306*np.sqrt(MachNumber**2 - 1)

def dynamicViscosityAir(temp):
    # Sutherlands law describes link between stag temp and dynamic viscosity.
    # https://www.cfd-online.com/Wiki/Sutherland%27s_law
    muRef = 1.827e-5 #1.716e-5 # kg/ms reference viscosity
    TRef = 291.15  # K, reference temperature.
    #S = 110.4 # Sutherlands constant
    S = 120.4 # Sutherlands constant

    # c1 = 1.458e-6 # kg/(ms*sqrt(K)) #(muRef/(TRef**(3/2)))*(TRef+S)
    c1 = (muRef / (TRef ** (3 / 2))) * (TRef + S)
    dynamVisc = (c1*(temp**(3/2)))/(temp+S)

    dynamVisc = muRef*((temp/TRef)**(3/2))*((TRef+S)/(temp+S))
    return dynamVisc


def reynoldsNumber(diam, vel, temp, density):
    dynamVisc = dynamicViscosityAir(temp=temp)
    print('Dynamic visc', dynamVisc)
    ReNum = vel*diam*density/dynamVisc
    return ReNum


if __name__ == '__main__':
    gamma = 1.4
    R_sp = 287.6

    print('MONASH TWIN-JETS')
    for NPR in [2.0, 5.0]:

        print('')
        print('#'*60)
        print('NPR=', NPR)
        T0 = 15.0+273.0
        Pj = 101325.0  # Pa
        P0 = NPR * Pj
        rho0 = P0 / (R_sp * T0)
        print('rho0', rho0, 'kg/m**3')
        Mj = NPRToMachNum(NPR=NPR, gamma=gamma)
        print('ideally Expanded MachNumber', Mj)

        Tj = isentropicTemp(T0=T0, Mach=Mj)
        print('ideally Expanded Temperature', Tj, 'K')
        print('Tj/T0', Tj/T0)

        rhoJ = isentropicDensity(rho0=rho0, Mach=Mj, gamma=gamma)
        print('ideally expanded density', rhoJ, 'kg/m**3')

        Uj = isentropicVelocity(NPR=NPR, Mach=Mj, T0=T0)
        print('Ideally expanded velocity', Uj, 'm/s')

        Dj = ideallyExpandedDiameter(dThroat=0.010, Mj=Mj)
        print('ideallyExpandedDiameter', Dj, 'm')

        print('Re_j:', reynoldsNumber(diam=Dj, vel=Uj, temp=Tj, density=rhoJ))

        print('Prandtl Shock Spacing for NPR=',NPR, ',',
              prandtlShockSpacing(NPRToMachNum(NPR=NPR, gamma=gamma)))
        print('')

    print('OSU TWIN-JETS')

    for NPR in [2.0, 5.0]:
        print('')
        print('#'*60)
        print('NPR=', NPR)
        T0 = 15.0+273.0
        Pj = 101325.0  # Pa
        P0 = NPR * Pj
        rho0 = P0 / (R_sp * T0)
        print('rho0', rho0, 'kg/m**3')
        Mj = NPRToMachNum(NPR=NPR, gamma=gamma)
        print('ideally Expanded MachNumber', Mj)

        Tj = isentropicTemp(T0=T0, Mach=Mj)
        print('ideally Expanded Temperature', Tj, 'K')
        print('Tj/T0', Tj/T0)

        rhoJ = isentropicDensity(rho0=rho0, Mach=Mj, gamma=gamma)
        print('ideally expanded density', rhoJ, 'kg/m**3')

        Uj = isentropicVelocity(NPR=NPR, Mach=Mj, T0=T0)
        print('Ideally expanded velocity', Uj, 'm/s')

        Dj = ideallyExpandedDiameter(dThroat=0.01905, Mj=Mj)
        print('ideallyExpandedDiameter', Dj, 'm')

        print('Re_j:', reynoldsNumber(diam=Dj, vel=Uj, temp=Tj, density=rhoJ))

        print('Prandtl Shock Spacing for NPR=',NPR, ',',
              prandtlShockSpacing(NPRToMachNum(NPR=NPR, gamma=gamma)))
        print('')

    exit(0)

    for NPR in [4.6, 5.0]:
        print('')
        print('#'*60)
        print('NPR=', NPR)
        T0 = 15.0+273.0
        Pj = 101325.0  # Pa
        P0 = NPR * Pj
        rho0 = P0 / (R_sp * T0)
        print('rho0', rho0, 'kg/m**3')
        Mj = NPRToMachNum(NPR=NPR, gamma=gamma)
        print('ideally Expanded MachNumber', Mj)

        Tj = isentropicTemp(T0=T0, Mach=Mj)
        print('ideally Expanded Temperature', Tj, 'K')
        print('Tj/T0', Tj/T0)

        rhoJ = isentropicDensity(rho0=rho0, Mach=Mj, gamma=gamma)
        print('ideally expanded density', rhoJ, 'kg/m**3')

        Uj = isentropicVelocity(NPR=NPR, Mach=Mj, T0=T0)
        print('Ideally expanded velocity', Uj, 'm/s')

        Dj = ideallyExpandedDiameter(dThroat=0.010, Mj=Mj)
        print('ideallyExpandedDiameter', Dj, 'm')

        print('Re_j:', reynoldsNumber(diam=Dj, vel=Uj, temp=Tj, density=rhoJ))

        print('Prandtl Shock Spacing for NPR=',NPR, ',',
              prandtlShockSpacing(NPRToMachNum(NPR=NPR, gamma=gamma)))
        print('')

    exit(0)

    # Re for Joel's case
    print('#'*60)
    print('')
    NPR = 3.4
    print('Joels 15mm nozzle and NPR=', NPR)
    T0 = 15+273.0
    Pj = 101325.0 # Pa
    P0 = NPR*Pj
    rho0 = P0/(R_sp*T0)

    Mj = NPRToMachNum(NPR=NPR, gamma=gamma)
    print('ideally Expanded MachNumber', Mj)

    Tj = isentropicTemp(T0=T0, Mach=Mj)
    print('ideally Expanded Temperature', Tj, 'K')
    print('Tj/T0', Tj / T0)

    rhoJ = isentropicDensity(rho0=rho0, Mach=Mj, gamma=gamma)
    print('ideally expanded density', rhoJ, 'kg/m**3')

    Uj = isentropicVelocity(NPR=NPR, Mach=Mj, T0=T0)

    print('Rho0', rho0)

    Dj = ideallyExpandedDiameter(dThroat=0.015, Mj=Mj)
    print('ideallyExpandedDiameter', Dj, 'm')

    print('Joels Re_j:', reynoldsNumber(diam=Dj, vel=Uj, temp=Tj, density=rhoJ))