import uproot as up
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import vector
import glob
import os


vector.register_awkward()

path = "/u/user/ab0633/Original"
the_list = glob.glob(path + '/*.root')

branches = [
    "Electron.PT", "Electron.Eta", "Electron.Phi", "Electron.Charge",
    "MuonTight.PT", "MuonTight.Eta", "MuonTight.Phi", "MuonTight.Charge",
    "PhotonTight.PT", "PhotonTight.Eta", "PhotonTight.Phi",
    "PuppiMissingET.MET", "PuppiMissingET.Phi",
    "JetPUPPI.PT", "JetPUPPI.Eta", "JetPUPPI.Phi", "JetPUPPI.BTag",
]
ELECTRON_MASS = 0.000511
MUON_MASS = 0.1057


#dR cleaning
# cartesian 으로 모든 쌍만들기
# dphi가 +pi보다 크면 2*pi를 빼고, -pi보다 작으면 2*pi를 더함
# where(조건, True 일 때 값, False 일 때 값)
# unzip 으로 한 이벤트의 i 번째 끼리 리스트만듬 
def delta_r(obj1, obj2):
    deta = obj1.eta - obj2.eta
    dphi = obj1.phi - obj2.phi
    dphi = ak.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = ak.where(dphi <= -np.pi, dphi + 2*np.pi, dphi)
    return (deta**2 + dphi**2)**0.5


collected_data = {
    "eee": [],
    "eem": [],
    "emm": [],
    "mmm": []
}

first_file = {"eee": True, "eem": True, "emm": True, "mmm": True}

for sample in the_list:
    with up.open(sample) as file:
        tree = file['Delphes;1']
        events = (tree.arrays(branches))


        el = ak.zip({
            "pt": events["Electron.PT"],
            "eta": events["Electron.Eta"],
            "phi": events["Electron.Phi"],
            "charge": events["Electron.Charge"],
            "mass": ak.full_like(events["Electron.PT"], ELECTRON_MASS)
        }, with_name="Momentum4D")



        mu = ak.zip({
            "pt": events["MuonTight.PT"],
            "eta": events["MuonTight.Eta"],
            "phi": events["MuonTight.Phi"],
            "charge": events["MuonTight.Charge"],
            "mass": ak.full_like(events["MuonTight.PT"], MUON_MASS)
        }, with_name="Momentum4D")

        ph = ak.zip({
            "pt": events["PhotonTight.PT"],
            "eta": events["PhotonTight.Eta"],
            "phi": events["PhotonTight.Phi"],
        }, with_name="Momentum4D")

        jet = ak.zip({
            "pt": events["JetPUPPI.PT"],
            "eta": events["JetPUPPI.Eta"],
            "phi": events["JetPUPPI.Phi"],
            "btag": events["JetPUPPI.BTag"],
        }, with_name="Momentum4D")

        met = ak.zip({
            "pt": events["PuppiMissingET.MET"],
            "phi": events["PuppiMissingET.Phi"],
        }, with_name="Momentum4D")

        #print(len(el.pt))

        # object ID

        el_ID = (el.pt > 15) \
                & ((abs(el.eta) < 1.442) | (abs(el.eta) > 1.566)) & (abs(el.eta) < 2.5)

        mu_ID = (mu.pt > 15) \
                & (abs(mu.eta)<2.5)

        ph_ID = (ph.pt > 20) \
                & ((abs(ph.eta) < 1.442) | (abs(ph.eta) > 1.566)) & (abs(ph.eta) < 2.5)

        jet_ID = (jet.pt > 30) \
                & (abs(jet.eta)<2.5)

        el = el[el_ID]
        mu = mu[mu_ID]
        ph = ph[ph_ID]
        good_jet = jet[jet_ID]


        #print(len(emm_mask))




        # dR cleaning

        #electron
        el_ph_pairs = ak.cartesian([el, ph], axis =1, nested=True)
        el_paired, ph_paired_el = ak.unzip(el_ph_pairs)
        dR_el_ph = delta_r(el_paired, ph_paired_el)
        el_iso = ak.all(dR_el_ph > 0.5, axis = -1)

        good_el = el[el_iso]

        #muon
        mu_ph_pairs = ak.cartesian([mu, ph], axis = 1 , nested=True)
        mu_paired, ph_paired_mu = ak.unzip(mu_ph_pairs)
        dR_mu_ph = delta_r(mu_paired, ph_paired_mu)
        mu_iso = ak.all(dR_mu_ph > 0.5, axis = -1)

        good_mu = mu[mu_iso]

        #print(len(good_mu))
        #print(len(good_el))

        #photon
        all_lep = ak.concatenate([el, mu], axis=1)

        lep_ph_pairs = ak.cartesian([all_lep, ph], axis=1, nested=True)
        lep_paired, ph_paired = ak.unzip(lep_ph_pairs)
        dR_lep_ph = delta_r(lep_paired, ph_paired)

        ph_iso = ak.all(dR_lep_ph > 0.5, axis=1)
        good_ph = ph[ph_iso]


        # 3 lepton

        lep = ak.concatenate([good_el, good_mu], axis=1)
        n_lep_mask = (ak.num(lep) == 3)


        # photon
        n_ph_mask = (ak.num(good_ph) == 1)


        # OSSF

        ee_pairs = ak.combinations(good_el, 2, fields=["i0", "i1"])
        os_e_mask = (ee_pairs.i0.charge * ee_pairs.i1.charge) == -1
        os_ee_pairs = ee_pairs[os_e_mask]

        mumu_pairs = ak.combinations(good_mu, 2, fields=["i0", "i1"])
        os_mu_mask = (mumu_pairs.i0.charge * mumu_pairs.i1.charge) == -1
        os_mumu_pairs = mumu_pairs[os_mu_mask]


        ossf_pairs = ak.concatenate([os_ee_pairs, os_mumu_pairs], axis=1)
        ossf_mask = ak.num(ossf_pairs) >= 1


        #Z mass window
        z_candidates = ossf_pairs.i0 + ossf_pairs.i1
        z_cand_mass = z_candidates.mass
        Z_mass = 91.1876
        z_window = abs(z_cand_mass - Z_mass ) < 20
        z_mass_mask = ak.any(z_window, axis=1)


        # lep_z tagging

        z_cand_pairs = ossf_pairs[(z_window)]
        #argmin() 공부 : 최소값의 위치 인덱싱

        z_mass = (z_cand_pairs.i0 + z_cand_pairs.i1).mass
        z_mass_diff = abs(z_mass - Z_mass)

        best_z_index = ak.argmin(z_mass_diff, axis=1)

        #print(best_z_index.type)
        #singletons 공부 : 마지막 차원의 각 요소를 리스트로 감싼다
        #firsts 는 1D 로 만들지만 pairs 의 필드 구조는 그대로 가지고 있음
        lep_pairs_z = ak.firsts(z_cand_pairs[ak.singletons(best_z_index)], axis=1)

        lep_pairs_pt = lep_pairs_z.i0.pt - lep_pairs_z.i1.pt

        lep_z_1 = ak.where(lep_pairs_pt > 0, lep_pairs_z.i0, lep_pairs_z.i1)
        lep_z_2 = ak.where(lep_pairs_pt < 0, lep_pairs_z.i0, lep_pairs_z.i1)



        # lep_w tagging
        w_cand_z1_pairs = ak.cartesian([lep, lep_z_1], axis=1)
        w_cand_1, z1_paired = ak.unzip(w_cand_z1_pairs)

        w_cand_z2_pairs = ak.cartesian([lep, lep_z_2], axis=1)
        w_cand_2, z2_paired = ak.unzip(w_cand_z2_pairs)


        is_z_1 = (w_cand_1.pt == z1_paired.pt) & (w_cand_1.eta == z1_paired.eta)
        is_z_2 = (w_cand_2.pt == z2_paired.pt) & (w_cand_2.eta == z2_paired.eta)


        #z가 없는것도 담기지만 상관없음
        is_from_Z = is_z_1 | is_z_2
        lep_w_all = lep[~is_from_Z]

        # 첫 번째 W lepton (empty 이벤트는 None) firsts 로 꺼내는 나의 논리가 맞는지 검사 받기!!!!!
        lep_w = ak.firsts(lep_w_all, axis=1)



        # lep pt cut
        # lep_z_1.pt >25 만 두면 none 은 자동으로 False 가 안되고 none 으로 남는다.

        lep_pt_mask = (
            ak.fill_none(lep_z_1.pt > 25, False) & 
            ak.fill_none(lep_w.pt > 25, False)
        )


        # met cut

        met_mask = (ak.firsts(met.pt) > 40)


        #bjet veto

        n_jets = ak.num(good_jet)


        #b-tagging은 여러 working point(loose, medium, tight)가 있는데 그 중 medium을 사용

        is_bjet_m = (jet.btag & 2) == 2
        n_bjets = ak.sum(is_bjet_m, axis =1)
        bjet_veto_mask = (n_bjets == 0)



        event_mask = (
            n_lep_mask &
            n_ph_mask &
            ossf_mask &
            z_mass_mask &
            lep_pt_mask &
            met_mask &
            bjet_veto_mask 
        )

        #print(len(event_mask))
        print(len(events[event_mask]))


        # Channel Division
        sel_el = good_el[event_mask]
        sel_mu = good_mu[event_mask]

        n_sel_el = ak.num(sel_el)
        n_sel_mu = ak.num(sel_mu)

        ch_eee = (n_sel_el == 3) & (n_sel_mu == 0)
        ch_eem = (n_sel_el == 2) & (n_sel_mu == 1)
        ch_emm = (n_sel_el == 1) & (n_sel_mu == 2)
        ch_mmm = (n_sel_el == 0) & (n_sel_mu == 3)


        # Saving to .npz
        sel_lep_z_1 = lep_z_1[event_mask]
        sel_lep_z_2 = lep_z_2[event_mask]
        sel_lep_w = lep_w[event_mask]
        sel_met = ak.firsts(met)[event_mask]
        sel_ph = ak.firsts(good_ph[event_mask])
        sel_n_jets = n_jets[event_mask]
        sel_n_bjets = n_bjets[event_mask]

        sel_n_jets = n_jets[event_mask]
        sel_n_bjets = n_bjets[event_mask]

        #가중치 준비
        sel_weight = ak.ones_like(sel_met.pt)

        channel_masks = {
            "eee": ch_eee,
            "eem": ch_eem,
            "emm": ch_emm,
            "mmm": ch_mmm
        }


        for ch_name, ch_mask in channel_masks.items():
            
            # 해당 채널의 이벤트 수 확인
            n_events_in_channel = ak.sum(ch_mask)
            
            # 이벤트가 없으면 NPZ 파일을 만들지 않고 건너뛰기
            if n_events_in_channel == 0:
                print(f"  Channel {ch_name}: 0 events, skipping.")
                continue
                
            # NPZ 파일에 저장할 딕셔너리 생성
            current_data = {
                # Z lep 1
                "lep_z1_pt": ak.to_numpy(sel_lep_z_1[ch_mask].pt),
                "lep_z1_eta": ak.to_numpy(sel_lep_z_1[ch_mask].eta),
                "lep_z1_phi": ak.to_numpy(sel_lep_z_1[ch_mask].phi),
                "lep_z1_mass": ak.to_numpy(sel_lep_z_1[ch_mask].mass),
                
                # Z lep 2
                "lep_z2_pt": ak.to_numpy(sel_lep_z_2[ch_mask].pt),
                "lep_z2_eta": ak.to_numpy(sel_lep_z_2[ch_mask].eta),
                "lep_z2_phi": ak.to_numpy(sel_lep_z_2[ch_mask].phi),
                "lep_z2_mass": ak.to_numpy(sel_lep_z_2[ch_mask].mass),
                
                # W lep
                "lep_w_pt": ak.to_numpy(sel_lep_w[ch_mask].pt),
                "lep_w_eta": ak.to_numpy(sel_lep_w[ch_mask].eta),
                "lep_w_phi": ak.to_numpy(sel_lep_w[ch_mask].phi),
                "lep_w_mass": ak.to_numpy(sel_lep_w[ch_mask].mass),
                "lep_w_charge": ak.to_numpy(sel_lep_w[ch_mask].charge),
                
                # Photon
                "ph_pt": ak.to_numpy(sel_ph[ch_mask].pt),
                "ph_eta": ak.to_numpy(sel_ph[ch_mask].eta),
                "ph_phi": ak.to_numpy(sel_ph[ch_mask].phi),
                
                # MET
                "met_pt": ak.to_numpy(sel_met[ch_mask].pt),
                "met_phi": ak.to_numpy(sel_met[ch_mask].phi),
                
                # jet
                "n_jets": ak.to_numpy(sel_n_jets[ch_mask]),
                "n_bjets": ak.to_numpy(sel_n_bjets[ch_mask]),
                
                # 가중치
                "weight": ak.to_numpy(sel_weight[ch_mask]),
            }
            
            if first_file[ch_name]:
                collected_data[ch_name] = current_data
                first_file[ch_name] = False
            else:
                for key in current_data.keys():
                    collected_data[ch_name][key] = np.concatenate([
                        collected_data[ch_name][key],
                        current_data[key]
                    ])

print("NPZ 파일 저장 시작")

output_directory = "/u/user/ab0633/WZG_ProcessedData/test"
os.makedirs(output_directory, exist_ok=True)

for ch_name in collected_data.keys():
    
    final_data_dict = collected_data[ch_name]
    
    if first_file[ch_name]:
        print(f"  Channel {ch_name}: 수집된 데이터가 없음")
        continue
    
    output_filename = os.path.join(output_directory, f"WZG_signal_{ch_name}.npz")
    
    np.savez_compressed(output_filename, **final_data_dict)

print("저장 완료")
