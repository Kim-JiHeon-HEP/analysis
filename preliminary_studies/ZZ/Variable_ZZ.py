import uproot as up
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

tree = up.open('/u/user/yeobi97/SE_UserHome/jeontampeu/DelPy_Di_boson_ZZ_run_99.root')

#print(tree.keys())
#출력 결과 : ['ProcessID0;1', 'Delphes;1']

Tree= tree['Delphes;1']

#print (Tree.keys())
# tree 안에 데이터들이 어떻게 저장되는지 branch 확인
#CHS란 Charged Hadron Subtraction 으로 원하지 않는 충돌들이 겹쳐서 기록되는 현상인 파일업 효과를 줄이기 위한 기술 중 하나이다.
#즉 오염이 더 보정된 데이터들이다. 

#PT_e = Tree['ElectronCHS.PT'].array()
#PT_e = Tree['Electron.PT'].array()
#print(PT_e)

#but 데이터를 확인해보니 컷이 안된 것 같아서 그냥 ELectron, muon branch 를 사용했다.


Branch_e = Tree['Electron']
Branch_mu = Tree['MuonTight']

#print (Branch_e.keys())
#!!!중간 변수는 안만드는게 좋은지 여쭤보기!!!

PT_e = Branch_e['Electron.PT'].array()
PT_mu = Tree['MuonTight.PT'].array()

Eta_e = Branch_e['Electron.Eta'].array()
Eta_mu = Branch_mu['MuonTight.Eta'].array()

Phi_e = Branch_e['Electron.Phi'].array()
Phi_mu = Branch_mu['MuonTight.Phi'].array()

#print(PT_e)


#Mass = Tree['Particle/Particle.Mass'].array()

#print(PT_e)
#print(PT_mu)

PT_e_flat = ak.flatten(PT_e)
Eta_e_flat = ak.flatten(Eta_e)
Phi_e_flat = ak.flatten(Phi_e)

PT_mu_flat = ak.flatten(PT_mu)
Eta_mu_flat = ak.flatten(Eta_mu)
Phi_mu_flat = ak.flatten(Phi_mu)

#Mass_flat = ak.flatten(Mass)



plt.figure(figsize=(8,6))
plt.hist(PT_e_flat, bins=100, range=(0,200), color = 'blue')
plt.title('ZZ PT_e')
plt.xlabel('PT_e [Gev]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ZZ PT_e.png',dpi=300)
plt.close()


plt.figure(figsize=(8,6))
plt.hist(Eta_e_flat, bins=100, range=(-6,6), color = 'red')
plt.title('ZZ Eta_e')
plt.xlabel('Eta_e')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ZZ Eta_e.png',dpi=300)
plt.close()


plt.figure(figsize=(8,6))
plt.hist(Phi_e_flat, bins=100, range=(-np.pi,np.pi), color = 'green')
plt.title('ZZ Phi_e')
plt.xlabel('Phi_e [rad]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ZZ phi_e.png',dpi=300)
plt.close()


plt.figure(figsize=(8,6))
plt.hist(PT_mu_flat, bins=100, range=(0,200), color = 'blue')
plt.title('ZZ PT_mu')
plt.xlabel('PT_mu [Gev]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ZZ PT_mu.png',dpi=300)
plt.close()


plt.figure(figsize=(8,6))
plt.hist(Eta_mu_flat, bins=100, range=(-6,6), color = 'red')
plt.title('ZZ Eta_mu')
plt.xlabel('Eta_mu')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ZZ Eta_mu.png',dpi=300)
plt.close()


plt.figure(figsize=(8,6))
plt.hist(Phi_mu_flat, bins=100, range=(-np.pi,np.pi), color = 'green')
plt.title('ZZ Phi_mu')
plt.xlabel('Phi_mu [rad]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ZZ phi_mu.png',dpi=300)
plt.close()