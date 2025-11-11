import uproot as up
import awkward as aw
import matplotlib.pyplot as plt
import numpy as np

tree = up.open('/u/user/yeobi97/SE_UserHome/jeontampeu/DY_genlevel.root')['LHEF']

pt=tree['Particle.PT'].array()
status=tree['Particle.Status'].array()
pid=tree['Particle.PID'].array()



mask_e = (status == 1) & (abs(pid) == 11)
mask_mu = (status ==1) & (abs(pid) == 13)

PT_e = pt[mask_e]
PT_mu = pt[mask_mu]

#aw.num 으로 awkward array 의 각 행에 몇개의 원소가 있는지 = 이벤트의 입자수 확인

Num_e = aw.num(PT_e)
Num_mu = aw.num(PT_mu)

min_e = aw.min(Num_e)
max_e = aw.max(Num_e)

min_mu = aw.min(Num_mu)
max_mu = aw.max(Num_mu)

#numver of particles 의 x축 범위를 잡기 위해 확인.
print(min_e)  
print(max_e)
print(min_mu)
print(max_mu)

#muon 이 없음을 확인 했음

plt.figure(figsize=(8,6))
plt.hist(Num_e, bins=100, range=(0,5), color = 'blue')
plt.title('genlevel DY Num_e')
plt.xlabel('Num_e')
plt.ylabel('Events')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('genlevel DY Num_e',dpi=300)
plt.close()