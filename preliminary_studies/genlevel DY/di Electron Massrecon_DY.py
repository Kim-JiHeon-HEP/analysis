import uproot as up
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import vector

tree = up.open('/u/user/yeobi97/SE_UserHome/jeontampeu/DY_genlevel.root')['LHEF']

pt=tree['Particle.PT'].array()
eta=tree['Particle.Eta'].array()
phi=tree['Particle.Phi'].array()
#mass=tree['Particle.M'].array()
status=tree['Particle.Status'].array()
pid=tree['Particle.PID'].array()

#print(tree.keys())


mask_e = (status == 1) & (abs(pid) == 11) 




PT_e = pt[mask_e]
Eta_e = eta[mask_e]
Phi_e = phi[mask_e]



IM_Z_list = []


for i in range(len(PT_e)) : 
    vec_em = vector.obj(pt=PT_e[i,0], eta=Eta_e[i,0], phi=Phi_e[i,0], mass=0.000511)
    vec_ep = vector.obj(pt=PT_e[i,1], eta= Eta_e[i,1], phi= Phi_e[i,1], mass= 0.000511)

    vec_Z = vec_em + vec_ep
    IM_Z = vec_Z.mass

    IM_Z_list.append(IM_Z)

    if i%1000==0:
            print(i)
    if i%100000==99999:
            break






plt.figure(figsize=(8,6))
plt.hist(IM_Z_list, bins=100 ,range=(0,100))
plt.title('Z Mass Reconstruction')
plt.xlabel('Z Mass [Gev]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--' , alpha=0.5)
plt.savefig('Z Mass Reconstruction',dpi=300)
plt.close()
