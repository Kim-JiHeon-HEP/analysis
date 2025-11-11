import uproot as up
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import vector

tree = up.open('/u/user/yeobi97/SE_UserHome/jeontampeu/DY_genlevel_em.root')['LHEF']


pt=tree['Particle.PT'].array()
eta=tree['Particle.Eta'].array()
phi=tree['Particle.Phi'].array()
status=tree['Particle.Status'].array()
pid=tree['Particle.PID'].array()

mask_e = (status == 1) & (abs(pid) == 11)
mask_mu = (status == 1) & (abs(pid) == 13)

PT_e = pt[mask_e]
PT_mu = pt[mask_mu]

Eta_e = eta[mask_e]
Eta_mu = eta[mask_mu]

Phi_e = phi[mask_e]
Phi_mu = phi[mask_mu]

#print(PT_e)
#print(PT_mu)

#print (ak.num(PT_e))

enentmask_e = (ak.num(PT_e) == 2)
enentmask_mu = (ak.num(PT_mu) == 2)

f_PT_e = PT_e[enentmask_e]
f_Eta_e = Eta_e[enentmask_e]
f_Phi_e = Phi_e[enentmask_e]

f_PT_mu = PT_mu[enentmask_mu]
f_Eta_mu = Eta_mu[enentmask_mu]
f_Phi_mu = Phi_mu[enentmask_mu]

#print (f_PT_e)
#print(f_Eta_e)

#print (f_PT_mu)

IM_Z_e_list = []


for i in range(len(f_PT_e)) : 
    vec_em = vector.obj(
           pt=f_PT_e[i,0], 
           eta=f_Eta_e[i,0], 
           phi=f_Phi_e[i,0], 
           mass=0.000511)
    vec_ep = vector.obj(
           pt=f_PT_e[i,1], 
           eta=f_Eta_e[i,1], 
           phi=f_Phi_e[i,1], 
           mass= 0.000511)

    vec_Z_e = vec_em + vec_ep
    IM_Z_e = vec_Z_e.mass

    IM_Z_e_list.append(IM_Z_e)

    if i%1000==0:
            print(i)
    if i%100000==99999:
            break
    

IM_Z_mu_list = []


for j in range(len(f_PT_mu)) : 
    vec_mu_m = vector.obj(
           pt=f_PT_mu[j,0], 
           eta=f_Eta_mu[j,0], 
           phi=f_Phi_mu[j,0], 
           mass=0.1057)
    vec_mu_p = vector.obj(
           pt=f_PT_mu[j,1], 
           eta=f_Eta_mu[j,1], 
           phi=f_Phi_mu[j,1], 
           mass= 0.1057)

    vec_Z_mu = vec_mu_m + vec_mu_p
    IM_Z_mu = vec_Z_mu.mass

    IM_Z_mu_list.append(IM_Z_mu)

    if j%1000==0:
            print(j)
    if j%100000==99999:
            break
    

plt.figure(figsize=(8,6))
plt.hist(IM_Z_e_list, bins=100 ,range=(0,100))
plt.title(r'Drell-Yan: $M(e^{+}e^{-})$')
plt.xlabel('Mass [Gev]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--' , alpha=0.5)
plt.savefig('DY_Mass_ee',dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.hist(IM_Z_mu_list, bins=100 ,range=(0,100))
plt.title(r'Drell-Yan: $M(\mu^{+}\mu^{-})$')
plt.xlabel('Mass [Gev]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--' , alpha=0.5)
plt.savefig('DY_Mass_mumu',dpi=300)
plt.close()