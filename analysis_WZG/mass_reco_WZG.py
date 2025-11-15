import uproot as up
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import vector
import glob
import os

vector.register_awkward()

base_dir = "/u/user/ab0633/WZG_ProcessedData_20_20"
channels = ['eee', 'eem', 'emm', 'mmm']
processes = ['WG', 'WW', 'WWW', 'WWZ', 'WZ', 'WZZ', 'ZG', 'ZZ', 'ZZZ', 'signal', 'ttbarG']


def load_data (base_dir, process, channels) :

    proc_data = {}

    for channel in channels :

        pattern = f'WZG_{process}_*_{channel}.npz'

        file_path = os.path.join(base_dir, process, pattern)

        # print(file_path)

        files = glob.glob(file_path)

        if not files:
            print(f"No files : {process}/{channel}")
            continue
    
        arrays = {}
        
        for filepath in files :
            with np.load(filepath) as loaded :
                for key in loaded.files :
                    if key not in arrays :
                        arrays[key] = []
                    arrays[key].append(loaded[key])

        channel_data = {}

        for key, array_list in arrays.items() :
            channel_data[key] = np.concatenate(array_list, axis=0)

        proc_data[channel] = channel_data

    return proc_data


 
all_data = {}

for proc in processes :
    all_data[proc] = load_data(base_dir, proc, channels)



#Cross-section normalization

metadata = {
    'signal' : {'xsec' : 1.730 * 10**(-3), 'n_gen' : 9899604,}, 
    'WG'     : {'xsec' : 23.080, 'n_gen' : 10010000,},
    'ZG'     : {'xsec' : 4.977, 'n_gen' : 10010000,},
    'WW'     : {'xsec' : 3.356, 'n_gen' : 10010000,},
    'WZ'     : {'xsec' : 3.983* 10**(-1), 'n_gen' : 10010000,},
    'ZZ'     : {'xsec' : 4.642* 10**(-2), 'n_gen' : 10010000,},
    'WWW'    : {'xsec' : 1.335* 10**(-3), 'n_gen' : 10003323,},
    'WWZ'    : {'xsec' : 3.067* 10**(-4), 'n_gen' : 9995610,},
    'WZZ'    : {'xsec' : 2.989* 10**(-5), 'n_gen' : 10010000,},
    'ZZZ'    : {'xsec' : 3.157* 10**(-6), 'n_gen' : 10010000,},
    'ttbarG' : {'xsec' : 2.445, 'n_gen' : 10010000,}
}

#pb^-1
luminosity = 3000000 

n_selected = {}
for proc in processes :
    total = sum(len(all_data[proc][chan]['weight'])
                for chan in all_data[proc])
    n_selected[proc] = total
    print(f"{proc}: {total} events")


weights = {}

for proc in processes:
    xsec = metadata[proc]['xsec']
    n_gen = metadata[proc]['n_gen']
    n_sel = n_selected[proc]

    weights[proc] = (xsec * luminosity) / n_gen
    print(f" {proc} weight : {weights[proc]}")



# 4vector
def create_4vec(data):
  
    vec = {}
    
    vec['lep_z1'] = ak.zip({
        "pt": data['lep_z1_pt'],
        "eta": data['lep_z1_eta'],
        "phi": data['lep_z1_phi'],
        "mass": data['lep_z1_mass'],
    }, with_name="Momentum4D")

    vec['lep_z2'] = ak.zip({
        "pt": data['lep_z2_pt'],
        "eta": data['lep_z2_eta'],
        "phi": data['lep_z2_phi'],
        "mass": data['lep_z2_mass'],
    }, with_name="Momentum4D")

    vec['lep_w'] = ak.zip({
        "pt": data['lep_w_pt'],
        "eta": data['lep_w_eta'],
        "phi": data['lep_w_phi'],
        "mass": data['lep_w_mass'],
    }, with_name="Momentum4D")

    vec['ph'] = ak.zip({
        "pt": data['ph_pt'],
        "eta": data['ph_eta'],
        "phi": data['ph_phi'],
        "mass": ak.zeros_like(data['ph_pt'])
    }, with_name="Momentum4D")

    vec['met'] = ak.zip({
        "pt": data['met_pt'],
        "eta" : ak.zeros_like(data['met_pt']),
        "phi": data['met_phi'],
        "mass": ak.zeros_like(data['met_pt'])
    }, with_name="Momentum4D")
    
    return vec


#signal

signal_vec = {}

for chan in channels:
    signal_vec[chan] = create_4vec(all_data["signal"][chan])


#bkg
bkg_vec = {}

bkg_processes = ['ZZZ','WZZ','WWW','WWZ','WW','ZG','ttbarG','ZZ','WZ']

for proc in bkg_processes:
    bkg_vec[proc] = {}
    for chan in channels:
        bkg_vec[proc][chan] = create_4vec(all_data[proc][chan])



#plot_mass_reconstruction

def plot_mass_reco(signal_mass_reco, signal_weight, bkg_mass_reco, bkg_weight) :

    for chan in channels :

        bkg_mass_reco_list = []
        bkg_weights_list = []
        bkg_labels = []

        for proc in bkg_processes:
            bkg_mass_reco_list.append(bkg_mass_reco[proc][chan])
            bkg_weights_list.append(bkg_weight[proc][chan])
            bkg_labels.append(proc)


        plt.figure(figsize=(8,6))
        plt.yscale('log')
        plt.hist(
            bkg_mass_reco_list,
            bins= np.linspace(0,200, 51),
            weights=bkg_weights_list,
            label=bkg_labels,
            histtype='stepfilled',
            stacked=True
        )
        plt.hist(
            signal_mass_reco[chan],
            weights = signal_weight[chan], 
            bins= np.linspace(0,200, 51),
            histtype = 'step', 
            color = 'black', 
            linewidth =2,
            label = 'signal'
        )
        plt.ylim(0.01, None)
        plt.legend(loc = 'best', fontsize =10)
        plt.title(f'M_T ({chan}_20_20)')
        plt.xlabel('M_T [GeV]')
        plt.savefig(f'M_T_{chan}_20_20',dpi=300)
        plt.close()





#m_z

signal_z_vec = {}
signal_z_mass = {}
signal_weights = {}


for chan in channels:
    signal_z_vec[chan] = signal_vec[chan]['lep_z1'] + signal_vec[chan]['lep_z2']
    signal_z_mass[chan] = signal_z_vec[chan].mass
    signal_weights[chan] = np.full(len(signal_z_mass[chan]), weights['signal'])

bkg_z_vec = {}
bkg_z_mass = {}
bkg_weights = {}

for proc in bkg_processes :
    bkg_z_vec[proc] = {}
    bkg_z_mass[proc] = {}
    bkg_weights[proc] = {}
    for chan in channels :
        bkg_z_vec[proc][chan] = bkg_vec[proc][chan]['lep_z1'] + bkg_vec[proc][chan]['lep_z2']
        bkg_z_mass[proc][chan] = bkg_z_vec[proc][chan].mass
        bkg_weights[proc][chan] = np.full(len(bkg_z_mass[proc][chan]), weights[proc])


#plot_mass_reco(signal_z_mass, signal_weights, bkg_z_mass, bkg_weights)

#m_w

signal_w_mass = {}
signal_weights = {}

for chan in channels:
    delta_phi = np.abs(signal_vec[chan]['lep_w']['phi'] - signal_vec[chan]['met']['phi'])
    delta_phi = np.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
    
    signal_w_mass[chan] = np.sqrt(
        2 * signal_vec[chan]['lep_w']['pt'] * signal_vec[chan]['met']['pt'] * 
        (1 - np.cos(delta_phi))
    )
    signal_weights[chan]= np.full(len(signal_w_mass[chan]), weights['signal'])

bkg_w_mass = {}
bkg_weights = {}

for proc in bkg_processes :
    bkg_w_mass[proc] = {}
    bkg_weights[proc] = {}
    for chan in channels :
        delta_phi = np.abs(bkg_vec[proc][chan]['lep_w']['phi'] - bkg_vec[proc][chan]['met']['phi'])
        delta_phi = np.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
        
        bkg_w_mass[proc][chan] = np.sqrt(
            2 * bkg_vec[proc][chan]['lep_w']['pt'] * bkg_vec[proc][chan]['met']['pt'] * 
            (1 - np.cos(delta_phi))
        )
        bkg_weights[proc][chan]= np.full(len(bkg_w_mass[proc][chan]), weights[proc])


#plot_mass_reco(signal_w_mass, signal_weights, bkg_w_mass, bkg_weights)


#m_zg

signal_zg_vec = {}
signal_zg_mass = {}
signal_weights = {}


for chan in channels:
    signal_zg_vec[chan] = signal_vec[chan]['lep_z1'] + signal_vec[chan]['lep_z2'] + signal_vec[chan]['ph']
    signal_zg_mass[chan] = signal_zg_vec[chan].mass
    signal_weights[chan] = np.full(len(signal_zg_mass[chan]), weights['signal'])

bkg_zg_vec = {}
bkg_zg_mass = {}
bkg_weights = {}

for proc in bkg_processes :
    bkg_zg_vec[proc] = {}
    bkg_zg_mass[proc] = {}
    bkg_weights[proc] = {}
    for chan in channels :
        bkg_zg_vec[proc][chan] = bkg_vec[proc][chan]['lep_z1'] + bkg_vec[proc][chan]['lep_z2'] + bkg_vec[proc][chan]['ph']
        bkg_zg_mass[proc][chan] = bkg_zg_vec[proc][chan].mass
        bkg_weights[proc][chan] = np.full(len(bkg_zg_mass[proc][chan]), weights[proc])

#plot_mass_reco (signal_zg_mass, signal_weights, bkg_zg_mass, bkg_weights)

# M_lll

signal_lll_vec = {}
signal_lll_mass = {}
signal_weights = {}


for chan in channels:
    signal_lll_vec[chan] = signal_vec[chan]['lep_z1'] + signal_vec[chan]['lep_z2'] + signal_vec[chan]['lep_w']
    signal_lll_mass[chan] = signal_lll_vec[chan].mass
    signal_weights[chan] = np.full(len(signal_lll_mass[chan]), weights['signal'])

bkg_lll_vec = {}
bkg_lll_mass = {}
bkg_weights = {}

for proc in bkg_processes :
    bkg_lll_vec[proc] = {}
    bkg_lll_mass[proc] = {}
    bkg_weights[proc] = {}
    for chan in channels :
        bkg_lll_vec[proc][chan] = bkg_vec[proc][chan]['lep_z1'] + bkg_vec[proc][chan]['lep_z2'] + bkg_vec[proc][chan]['lep_w']
        bkg_lll_mass[proc][chan] = bkg_lll_vec[proc][chan].mass
        bkg_weights[proc][chan] = np.full(len(bkg_lll_mass[proc][chan]), weights[proc])

#plot_mass_reco ( signal_lll_mass, signal_weights, bkg_lll_mass, bkg_weights)


# M_lllg

signal_lllg_vec = {}
signal_lllg_mass = {}
signal_weights = {}


for chan in channels:
    signal_lllg_vec[chan] = signal_vec[chan]['lep_z1'] + signal_vec[chan]['lep_z2'] + signal_vec[chan]['lep_w'] + signal_vec[chan]['ph']
    signal_lllg_mass[chan] = signal_lllg_vec[chan].mass
    signal_weights[chan] = np.full(len(signal_lllg_mass[chan]), weights['signal'])

bkg_lllg_vec = {}
bkg_lllg_mass = {}
bkg_weights = {}

for proc in bkg_processes :
    bkg_lllg_vec[proc] = {}
    bkg_lllg_mass[proc] = {}
    bkg_weights[proc] = {}
    for chan in channels :
        bkg_lllg_vec[proc][chan] = bkg_vec[proc][chan]['lep_z1'] + bkg_vec[proc][chan]['lep_z2'] + bkg_vec[proc][chan]['lep_w'] +  bkg_vec[proc][chan]['ph']
        bkg_lllg_mass[proc][chan] = bkg_lllg_vec[proc][chan].mass
        bkg_weights[proc][chan] = np.full(len(bkg_lllg_mass[proc][chan]), weights[proc])

#plot_mass_reco ( signal_lllg_mass, signal_weights, bkg_lllg_mass, bkg_weights)


signal_w_vec = {}
signal_w_mass = {}
signal_weights = {}


for chan in channels:
    signal_w_vec[chan] = signal_vec[chan]['lep_w'] + signal_vec[chan]['met']
    signal_w_mass[chan] = signal_w_vec[chan].Mt
    signal_weights[chan] = np.full(len(signal_w_mass[chan]), weights['signal'])

bkg_w_vec = {}
bkg_w_mass = {}
bkg_weights = {}

for proc in bkg_processes :
    bkg_w_vec[proc] = {}
    bkg_w_mass[proc] = {}
    bkg_weights[proc] = {}
    for chan in channels :
        bkg_w_vec[proc][chan] = bkg_vec[proc][chan]['lep_w'] + bkg_vec[proc][chan]['met']
        bkg_w_mass[proc][chan] = bkg_w_vec[proc][chan].Mt
        bkg_weights[proc][chan] = np.full(len(bkg_w_mass[proc][chan]), weights[proc])

#plot_mass_reco(signal_w_mass, signal_weights, bkg_w_mass, bkg_weights)

delta_phi = np.abs(signal_vec['eee']['lep_w']['phi'] - signal_vec['eee']['met']['phi'])
delta_phi = np.where(delta_phi > np.pi, 2*np.pi - delta_phi, delta_phi)
m_w_manual = np.sqrt(2 * signal_vec['eee']['lep_w']['pt'] *  signal_vec['eee']['met']['pt'] * (1 - np.cos(delta_phi)))

# 방법 2: .Mt 사용
w_vec = signal_vec['eee']['lep_w'] + signal_vec['eee']['met']
m_w_Mt = w_vec.mt

# 비교
print(f"Manual: {m_w_manual[:10]}")
print(f"Mt method: {m_w_Mt[:10]}")
print(f"Difference: {np.abs(m_w_manual - m_w_Mt)[:10]}")