import uproot as up
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

tree = up.open('/u/user/yeobi97/SE_UserHome/jeontampeu/DelPy_Di_boson_ZZ_run_99.root')
Tree= tree['Delphes;1']

Branch_e = Tree['Electron']
Branch_mu = Tree['MuonTight']

PT_e = Branch_e['Electron.PT'].array()
PT_mu = Branch_mu['MuonTight.PT'].array()


Num_e = ak.num(PT_e)
Num_mu = ak.num(PT_mu)
Num_particles = Num_e + Num_mu


min_e = ak.min(Num_e)
max_e = ak.max(Num_e)

min_mu = ak.min(Num_mu)
max_mu = ak.max(Num_mu)

#numver of particles 의 x축 범위를 잡기 위해 확인.
#print(min_e)  
#print(max_e)
#print(min_mu)
#print(max_mu)

#.png 가 기본값인 것 같음

plt.figure(figsize=(8,6))
plt.hist(Num_e, bins=100, range=(0,10), color = 'blue')
plt.title('ZZ Num_e')
plt.xlabel('Num_e')
plt.ylabel('Events')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ZZ Num_e',dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.hist(Num_mu, bins=100, range=(0,10), color = 'blue')
plt.title('ZZ Num_mu')
plt.xlabel('Num_mu')
plt.ylabel('Events')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ZZ Num_mu',dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.hist(Num_particles, bins=100, range=(0,10), color = 'blue')
plt.title('ZZ Num_particles')
plt.xlabel('Num_particles')
plt.ylabel('Events')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ZZ Num_particles',dpi=300)
plt.close()

#0이 많아서 무언가 잘못됐다고 생각했으나 Num_particles 의 0은 더 작은것 보니
#뮤온이 0 인 이벤트에선 전자만 전자가 0인 이벤트에선 뮤온만 나온 이벤트인 것 같음.