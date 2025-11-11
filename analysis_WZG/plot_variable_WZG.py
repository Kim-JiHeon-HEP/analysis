import uproot as up
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import vector
import glob
import os

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




#plot low level variables

bkg_processes = ['ZZZ','WZZ','WWW','WWZ','WW','ttbarG','ZG','ZZ','WZ']

plot_pt = ['lep_z1_pt', 'lep_z2_pt', 'lep_w_pt', 'ph_pt', 'met_pt']
plot_eta = ['lep_z1_eta', 'lep_z2_eta', 'lep_w_eta', 'ph_eta']
plot_phi = ['lep_z1_phi', 'lep_z2_phi', 'lep_w_phi', 'ph_phi', 'met_phi']
plot_jet = ['n_jets', 'n_bjets']



for chan in channels :

    for variable in plot_jet :
        
        bkg_data = []
        bkg_weights = []
        bkg_labels = []

        print(f"{variable}")

        for proc in bkg_processes :
            data = all_data[proc][chan][variable]
            weight_arr = np.full(len(data), weights[proc])
            
            bkg_data.append(data)
            bkg_weights.append(weight_arr)
            bkg_labels.append(proc)

        
        
        signal_data = all_data['signal'][chan][variable]
        signal_weights = np.full(len(signal_data), weights['signal'])

        
        plt.figure(figsize=(8,6))
        plt.yscale('log')
        plt.hist(bkg_data, weights=bkg_weights, bins=50, histtype = 'stepfilled',stacked=True, label = bkg_labels )
        plt.hist(signal_data, weights = signal_weights, bins=50,histtype = 'step', color = 'black', linewidth =2, label = 'signal')
        plt.title(f'{variable}_{chan}_ver1')
        plt.xlabel(f'{variable}')
        plt.ylabel('Events')
        plt.ylim(0.01, None)
        plt.xlim(0, 10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc = 'best', fontsize =10)
        plt.savefig(f'{variable}_{chan}_ver1',dpi=300)
        plt.close()


 

