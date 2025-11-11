import uproot as up
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

tree = up.open('/u/user/yeobi97/SE_UserHome/jeontampeu/4top_ex1_delphes_9.root')

#print(tree.keys())

Tree = tree['Delphes;1']

branch_j = Tree['Jet']

pt_j = Tree['Jet.PT'].arrays()

PT_j = pt_j['Jet.PT']

Num_j = ak.num(PT_j)

min_j = ak.min(Num_j)
max_j = ak.max(Num_j)

#print(min_j)
#print(max_j)


plt.figure(figsize=(8,6))
plt.hist(Num_j, bins=100, range=(0,25), color = 'blue')
plt.title('4T Num_j')
plt.xlabel('Num_j')
plt.ylabel('Events')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('4T Num_j',dpi=300)
plt.close()