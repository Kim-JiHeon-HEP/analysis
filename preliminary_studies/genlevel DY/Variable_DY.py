import uproot as up
import awkward as aw
import matplotlib.pyplot as plt
import numpy as np

tree = up.open('/u/user/yeobi97/SE_UserHome/jeontampeu/DY_genlevel.root')['LHEF']

pt=tree['Particle.PT'].array()
eta=tree['Particle.Eta'].array()
phi=tree['Particle.Phi'].array()
status=tree['Particle.Status'].array()
pid=tree['Particle.PID'].array()



mask_e = (status == 1) & (abs(pid) == 11)

PT_e = pt[mask_e]
ETA_e = eta[mask_e]
PHI_e = phi[mask_e]

#plt.hist 는 입력값을 1차원으로 받기 때문에 
#flatten 으로 [[...],[...]] 을 [......] 로 만들어주기

PT_e_flat = aw.flatten(PT_e)
ETA_e_flat = aw.flatten(ETA_e)
PHI_e_flat = aw.flatten(PHI_e)

#tier3 에 ssh로 접속하는 환경에선 GUI 창이 내 맥북에 뜨는 것이 아니기 때문에
#show가 아닌 savefig 를 사용해서 내 맥북으로 다운받아서 확인하는 것이다.
#나머지 히스토그램 그리는 코드는 matlab 이랑 비슷하다.


#Z Boson 의 질량이 91.2 GeV 이기 때문에 그것에 반의 값을 PT로
# 나눠 가지기 때문에 절반 정도인 46Gev에서 피크를 보일 것으로 예상
# 범위를 0,100 으로 잡았다
 
 
plt.figure(figsize=(8,6))
plt.hist(PT_e_flat, bins=100, range=(0,100), color = 'blue')
plt.title('genlevel DY PT_e')
plt.xlabel('PT_e [Gev]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('genlevel DY PT_e.png',dpi=300)
plt.close()

#ETA 는 빔과 이루는 각도의 tan와 관계된 값으로
#0도에서 180도까지의 범위인 -6,6으로  잡았다.

plt.figure(figsize=(8,6))
plt.hist(ETA_e_flat, bins=100, range=(-6,6), color = 'red')
plt.title('genlevel DY ETA_e')
plt.xlabel('ETA_e')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('genlevel DY ETA_e.png',dpi=300)
plt.close()

#PHI 는 360도의 값을 가질 수 있기 때문에 numpy 를 이용해
#-pi , pi 까지 범위를 잡았다.

plt.figure(figsize=(8,6))
plt.hist(PHI_e_flat, bins=100, range=(-np.pi,np.pi), color = 'green')
plt.title('genlevel DY PHI_e')
plt.xlabel('PHI_e [rad]')
plt.ylabel('Entries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('genlevel DY PHI_e.png',dpi=300)
plt.close()