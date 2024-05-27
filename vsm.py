#%%
"""
VSM 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 

def lineal(x,m,n):
    return m*x+n
#%% LAURICO S/ NPM

C = np.loadtxt('Laurico_240411.txt',skiprows=12)
mL=0.2874 #g
f = C[:,0]
g = C[:,1]/mL

f1=f[np.nonzero(f>=2500)]
g1=g[np.nonzero(f>=2500)]
f2=f[np.nonzero(f<=-2500)]
g2=g[np.nonzero(f<=-2500)]

(m1,n1), pcov= curve_fit(lineal,f1,g1)
(m2,n2), pcov= curve_fit(lineal,f2,g2)


chi_mass_laurico = np.mean([m1,m2])
g_ajustado = lineal(f,chi_mass_laurico,0)

plt.plot(f,g,'.-')
plt.plot(f1,g1,'.-')
plt.plot(f2,g2,'.-')
plt.plot(f,g_ajustado,'-',c='tab:red',label=f'$\chi$ = {chi_mass_laurico:.2e} emu/gG')
plt.legend()
plt.grid()
plt.xlabel('H (G)')
plt.ylabel('m (emu/g)')
plt.title('Capsula ac. laurico')

#%% Muestra RANDOM
files=os.listdir(os.path.join(os.getcwd(),'Random'))
files.sort()
lim_ajuste_R=10

masa_muestra_R=0.24885 #g
masa_npm_R=0.0912*masa_muestra_R #g
masa_laurico_R=(1-0.0912)*masa_muestra_R #g

campos=[]
m_correg_norm=[]
fname=[]
Ms_R=[]

H_aux = np.linspace(-50,50,1000)
campo_lineal=[]
m_interp=[]

pendientes_R=[]
err_pend_R=[]
ordenadas_R=[]
Mr=[]
Hc=[]
angulo_R=[]
for f in files:
    B = np.loadtxt(os.path.join(os.getcwd(),'Random',f),skiprows=12)
    H = B[:,0]
    m = B[:,1]
    #calculo la contribucion diamagnetica del ac laurico y se la descuento
    contrib_diamag_R = chi_mass_laurico*masa_laurico_R*H
    m_correg = (m-contrib_diamag_R)
    m_correg_norm_masa=m_correg/masa_npm_R

    #interpolador=interp1d(H,m_correg_norm_masa,fill_value='extrapolate')
    #m_interp.append(interpolador(H_aux))

    m_norm=m_correg_norm_masa/max(m_correg_norm_masa) #Normalizo momento magnetico por valor maximo

    m_correg_norm.append(m_norm)
    campos.append(np.array(H))
    fname.append(f.split('_')[-1].split('.')[0])
    Ms_R.append(max(m_correg_norm_masa))
    angulo_R.append(float(f.split('_')[-1].split('.')[0]))
    #Ajuste lineal
    (chi,n),pcov=curve_fit(lineal, H[np.nonzero(abs(H)<lim_ajuste_R)],m_norm[np.nonzero(abs(H)<lim_ajuste_R)])  
    err_m=pcov[0][0]
    pendientes_R.append(chi)
    err_pend_R.append(err_m)
    ordenadas_R.append(n)

    H_recortado = H[np.nonzero(abs(H)<=lim_ajuste_R)]
    m_recortado = m_correg_norm_masa[np.nonzero(abs(H)<=lim_ajuste_R)]

    H_aux = np.linspace(-lim_ajuste_R,lim_ajuste_R,5000)
    m_aux = lineal(H_aux,chi,n)
    R2= r2_score(m_norm[np.nonzero(abs(H)<=lim_ajuste_R)],lineal(H[np.nonzero(abs(H)<=lim_ajuste_R)],chi,n))
    indx_H = np.nonzero(H_aux>=0)[0][0]
    indx_M = np.nonzero(m_aux>=0)[0][0]
    Hc.append(-H_aux[indx_M])
    Mr.append(m_aux[indx_H])
    print('*'*50)
    print(f.split('_')[0], f.split('_')[-1])
    print('Susceptibilidad =',f'{chi:.3e}','+/-',f'{err_m:.3e}')
    print(f'Mag Remanente = {m_aux[indx_H]:.3e}')
    print(f'Campo Coercitivo = {H_aux[indx_M]:.3e} A/m')
    print(f'R² = {R2:.3f}')


    fig,ax=plt.subplots(constrained_layout=True)
    ax.plot(H,m/masa_npm_R,'.-',label='R')
    ax.plot(H,m_correg_norm_masa,'.-',label='R (s/ laurico)')
    ax.legend(ncol=2,loc='lower right')
    ax.grid()
    ax.set_xlabel('H (G)')
    ax.set_ylabel('m (emu/g)')
    ax.set_title(f)

    axins = ax.inset_axes([0.3, 0.2, 0.69, 0.55])
    axins.axvline(0,0,1,c='k',lw=0.8)
    axins.axhline(0,0,1,c='k',lw=0.8)
    axins.plot(H,m_norm,'.-',label='m norm')
    axins.plot(H_aux,m_aux,'r-',label=f'$\chi$ = {chi:.3e}\nR² = {R2:.3f}')
    axins.set_xlabel('H (G)')
    axins.grid()
    axins.legend(loc='lower right')
    axins.set_xlim(min(H_aux),max(H_aux))
    axins.set_ylim(min(m_aux),max(m_aux))
    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.savefig('VSM_'+f[:-4]+'.png',dpi=300)
    plt.show()

#%% Perpendicular
files=os.listdir(os.path.join(os.getcwd(),'Perpendicular'))
files.sort()
lim_ajuste_P=10

masa_muestra_P=0.28993 #g
masa_npm_P=0.0912*masa_muestra_P #g
masa_laurico_P=(1-0.0912)*masa_muestra_P #g

campos=[]
m_correg_norm=[]
fname=[]
Ms_P=[]

H_aux = np.linspace(-50,50,1000)
campo_lineal=[]
m_interp=[]

pendientes_P=[]
err_pend_P=[]
ordenadas_P=[]
Mr=[]
Hc=[]
angulo_P=[]
for f in files:
    B = np.loadtxt(os.path.join(os.getcwd(),'Perpendicular',f),skiprows=12)
    H = B[:,0]
    m = B[:,1]
    #calculo la contribucion diamagnetica del ac laurico y se la descuento
    contrib_diamag_P = chi_mass_laurico*masa_laurico_P*H
    m_correg = (m-contrib_diamag_P)
    m_correg_norm_masa=m_correg/masa_npm_P
    
    #interpolador=interp1d(H,m_correg_norm_masa,fill_value='extrapolate')
    #m_interp.append(interpolador(H_aux))
    
    m_norm=m_correg_norm_masa/max(m_correg_norm_masa) #Normalizo momento magnetico por valor maximo
    
    m_correg_norm.append(m_norm)
    campos.append(np.array(H))
    fname.append(f.split('_')[-1].split('.')[0])
    Ms_P.append(max(m_correg_norm_masa))
    angulo_P.append(float(f.split('_')[-1].split('.')[0]))
    #Ajuste lineal
    (chi,n),pcov=curve_fit(lineal, H[np.nonzero(abs(H)<lim_ajuste_P)],m_norm[np.nonzero(abs(H)<lim_ajuste_P)])  
    err_m=pcov[0][0]
    pendientes_P.append(chi)
    err_pend_P.append(err_m)
    ordenadas_P.append(n)
    
    H_recortado = H[np.nonzero(abs(H)<=lim_ajuste_P)]
    m_recortado = m_correg_norm_masa[np.nonzero(abs(H)<=lim_ajuste_P)]
    
    H_aux = np.linspace(-lim_ajuste_P,lim_ajuste_P,5000)
    m_aux = lineal(H_aux,chi,n)
    R2= r2_score(m_norm[np.nonzero(abs(H)<=lim_ajuste_P)],lineal(H[np.nonzero(abs(H)<=lim_ajuste_P)],chi,n))
    indx_H = np.nonzero(H_aux>=0)[0][0]
    indx_M = np.nonzero(m_aux>=0)[0][0]
    Hc.append(-H_aux[indx_M])
    Mr.append(m_aux[indx_H])
    print('*'*50)
    print(f.split('_')[0], f.split('_')[-1])
    print('Susceptibilidad =',f'{chi:.3e}','+/-',f'{err_m:.3e}')
    print(f'Mag Remanente = {m_aux[indx_H]:.3e}')
    print(f'Campo Coercitivo = {H_aux[indx_M]:.3e} A/m')
    print(f'R² = {R2:.3f}')
    

    fig,ax=plt.subplots(constrained_layout=True)
    ax.plot(H,m/masa_npm_P,'.-',label='P')
    ax.plot(H,m_correg_norm_masa,'.-',label='P (s/ laurico)')
    ax.legend(ncol=2,loc='lower right')
    ax.grid()
    ax.set_xlabel('H (G)')
    ax.set_ylabel('m (emu/g)')
    ax.set_title(f)
    
    axins = ax.inset_axes([0.3, 0.2, 0.69, 0.55])
    axins.axvline(0,0,1,c='k',lw=0.8)
    axins.axhline(0,0,1,c='k',lw=0.8)
    axins.plot(H,m_norm,'.-',label='m norm')
    axins.plot(H_aux,m_aux,'r-',label=f'$\chi$ = {chi:.3e}\nR² = {R2:.3f}')
    axins.set_xlabel('H (G)')
    axins.grid()
    axins.legend(loc='lower right')
    axins.set_xlim(min(H_aux),max(H_aux))
    axins.set_ylim(min(m_aux),max(m_aux))
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    plt.savefig('VSM_'+f[:-4]+'.png',dpi=300)
    plt.show()
#%% Axial
files=os.listdir(os.path.join(os.getcwd(),'Axial'))
lim_ajuste_A=10

files.sort()

masa_muestra_A=0.28993 #g
masa_npm_A=0.0912*masa_muestra_A #g
masa_laurico_A=(1-0.0912)*masa_muestra_A #g
campos=[]
m_correg_norm=[]
fname=[]
Ms_A=[]

H_aux = np.linspace(-50,50,1000)
campo_lineal=[]
m_interp=[]

pendientes_A=[]
err_pend_A=[]
ordenadas_A=[]
Mr=[]
Hc=[]
angulo_A=[]


for f in files:
    B = np.loadtxt(os.path.join(os.getcwd(),'Axial',f),skiprows=12)
    H = B[:,0]
    m = B[:,1]
    #calculo la contribucion diamagnetica del ac laurico y se la descuento
    contrib_diamag_A = chi_mass_laurico*masa_laurico_A*H
    m_correg = (m-contrib_diamag_A)
    m_correg_norm_masa=m_correg/masa_npm_A
    
    #interpolador=interp1d(H,m_correg_norm_masa,fill_value='extrapolate')
    #m_interp.append(interpolador(H_aux))
    
    m_norm=m_correg_norm_masa/max(m_correg_norm_masa) #Normalizo momento magnetico por valor maximo
    
    m_correg_norm.append(m_norm)
    campos.append(np.array(H))
    fname.append(f.split('_')[-1].split('.')[0])
    Ms_A.append(max(m_correg_norm_masa))
    angulo_A.append(float(f.split('_')[-1].split('.')[0]))
    #Ajuste lineal
    (chi,n),pcov=curve_fit(lineal, H[np.nonzero(abs(H)<lim_ajuste_A)],m_norm[np.nonzero(abs(H)<lim_ajuste_A)])  
    err_m=pcov[0][0]
    pendientes_A.append(chi)
    err_pend_A.append(err_m)
    ordenadas_A.append(n)
    
    H_recortado = H[np.nonzero(abs(H)<=lim_ajuste_A)]
    m_recortado = m_correg_norm_masa[np.nonzero(abs(H)<=lim_ajuste_A)]
    
    H_aux = np.linspace(-lim_ajuste_A,lim_ajuste_A,5000)
    m_aux = lineal(H_aux,chi,n)
    R2= r2_score(m_norm[np.nonzero(abs(H)<=lim_ajuste_A)],lineal(H[np.nonzero(abs(H)<=lim_ajuste_A)],chi,n))
    indx_H = np.nonzero(H_aux>=0)[0][0]
    indx_M = np.nonzero(m_aux>=0)[0][0]
    Hc.append(-H_aux[indx_M])
    Mr.append(m_aux[indx_H])
    print('*'*50)
    print(f.split('_')[0], f.split('_')[-1])
    print('Susceptibilidad =',f'{chi:.3e}','+/-',f'{err_m:.3e}')
    print(f'Mag Remanente = {m_aux[indx_H]:.3e}')
    print(f'Campo Coercitivo = {H_aux[indx_M]:.3e} A/m')
    print(f'R² = {R2:.3f}')
    

    fig,ax=plt.subplots(constrained_layout=True)
    ax.plot(H,m/masa_npm_A,'.-',label='A')
    ax.plot(H,m_correg_norm_masa,'.-',label='A (s/ laurico)')
    ax.legend(ncol=2,loc='lower right')
    ax.grid()
    ax.set_xlabel('H (G)')
    ax.set_ylabel('m (emu/g)')
    ax.set_title(f)
    
    axins = ax.inset_axes([0.3, 0.2, 0.69, 0.55])
    axins.axvline(0,0,1,c='k',lw=0.8)
    axins.axhline(0,0,1,c='k',lw=0.8)
    axins.plot(H,m_norm,'.-',label='m norm')
    axins.plot(H_aux,m_aux,'r-',label=f'$\chi$ = {chi:.3e}\nR² = {R2:.3f}')
    axins.set_xlabel('H (G)')
    axins.grid()
    axins.legend(loc='lower right')
    axins.set_xlim(min(H_aux),max(H_aux))
    axins.set_ylim(min(m_aux),max(m_aux))
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    plt.savefig('VSM_'+f[:-4]+'.png',dpi=300)
    plt.show()
    
#%%SUCEPTIBILIDAD vs ANGULO VERTICAL 
#plt.close('all')
fig,ax=plt.subplots(nrows=1,figsize=(9,4),sharex=True,constrained_layout=True)
ax.errorbar(x=angulo_P,y=pendientes_P,yerr=err_pend_P, fmt='.-', capsize=5,label=f'Perpendicular')
ax.errorbar(x=angulo_A,y=pendientes_A,yerr=err_pend_A, fmt='.-', capsize=5,label=f'Axial')
ax.errorbar(x=angulo_R,y=pendientes_R,yerr=err_pend_R, fmt='.-', capsize=5,label=f'Random')

#ax.axhline(pendientes_Rd,0,1,c='k')
ax.grid()
ax.legend()
ax.set_ylabel('$\chi$')
ax.set_title('Susceptibilidad vs angulo')

xticks_values = angulo_R
xticks_labels = [str(i) for i in xticks_values]
plt.xticks(xticks_values, xticks_labels,rotation=45)

# ax1.plot(rot_all_V,Ms,'o-',label='M$_s$')
# ax1.set_ylabel('M$_s$')
# ax1.grid(True)
# ax1.legend()
plt.xlabel('Ángulo (º)')
# delta_V=(max(pendientes_V)- min(pendientes_V))/max(pendientes_V)
# print(delta_V)
plt.savefig('Susceptibilidad_vs_angulo.png',dpi=300,facecolor='w')

# %%
