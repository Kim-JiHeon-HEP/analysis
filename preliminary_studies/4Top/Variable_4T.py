import uproot as up
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

tree = up.open('/u/user/yeobi97/SE_UserHome/jeontampeu/4top_ex1_delphes_9.root')

#print(tree.keys())

Tree = tree['Delphes;1']

branch_j = Tree['Jet']

#print(Tree.keys())

#print(branch_j.keys())

pt_j = Tree['Jet.PT'].arrays()
eta_j = Tree['Jet.Eta'].arrays()
phi_j = branch_j['Jet.Phi'].arrays()
mass_j = branch_j['Jet.Mass'].arrays()

#print(pt_j)
#print(eta_j)
#print(phi_j)
#print(mass_j)
#입자들과 다르게 jet은 딕셔너리라는 구조를 포함함

PT_j = pt_j['Jet.PT']
Eta_j = eta_j['Jet.Eta']
Phi_j = phi_j['Jet.Phi']
Mass_j = mass_j['Jet.Mass']

# print (PT_j)
# print (Eta_j)
# print (Phi_j)
# print (Mass_j)

PT_j_flat = ak.flatten(PT_j)
Eta_j_flat = ak.flatten(Eta_j)
Phi_j_flat = ak.flatten(Phi_j)
Mass_j_flat = ak.flatten(Mass_j)

min_Pt = ak.min(PT_j_flat)
max_Pt = ak.max(PT_j_flat)

#print(min_Pt)
#print(max_Pt)
# 0~1600


min_Mass = ak.min(Mass_j_flat)
max_Mass = ak.max(Mass_j_flat)

#print(min_Mass)
#print(max_Mass)
# -3 ~ 271


plt.figure(figsize=(8,6))
plt.hist(PT_j_flat, bins=100, range=(0,1600), color = 'blue')
plt.title('4T PT_j')
plt.xlabel('PT_j [Gev]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('4T PT_j.png',dpi=300)
plt.close()


plt.figure(figsize=(8,6))
plt.hist(Eta_j_flat, bins=100, range=(-6,6), color = 'red')
plt.title('4T Eta_j')
plt.xlabel('Eta_j')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('4T Eta_j.png',dpi=300)
plt.close()


plt.figure(figsize=(8,6))
plt.hist(Phi_j_flat, bins=100, range=(-np.pi,np.pi), color = 'green')
plt.title('4T Phi_j')
plt.xlabel('Phi_j [rad]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('4T phi_j.png',dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.hist(Mass_j_flat, bins=100, range=(-3,300), color = 'green')
plt.title('4T Mass_j')
plt.xlabel('Mass_j [rad]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('4T Mass_j.png',dpi=300)
plt.close()