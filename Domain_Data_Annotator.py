import os
import json
import requests
import itertools
import numpy as np
import multiprocessing
from Bio.PDB import PDBParser

def parse_dommap_ranges(rng_data):
    rng_parse = []
    for rng_list_1 in rng_data.split(","):
        tmp_rng = []
        for rng_list_2 in rng_list_1.split("-"):
            tmp_rng += [int(rng_list_2)]
        rng_parse.append(tmp_rng)
    return rng_parse

def read_dommap_tsv(file_path):
    dom_map = {}
    with open(file_path,"r") as file:
        for line in file:
            if not line.startswith("#"):
                data = line.rsplit("\t")
                uniprot = data[0].split("|")[1]
                dom_rang = parse_dommap_ranges(data[2])
                topo = data[3]

                AXTFFid = data[4:-1]

                # If it is the first time encountering a protein with an unique uniprot...
                if uniprot not in dom_map.keys():
                    # Then attempt to web scrape the InterPro site for this protein...
                    ip_result = add_interpro_ranges(uniprot)
                    if ip_result:
                        # If the web scrape was successful save the InterPro data AND the DomainMapper data...
                        interpro_rang, interpro_topo, interpro_name, interpro_model, pfam_rang, pfam_topo, pfam_name, pfam_model = ip_result
                        dom_map[uniprot] = {"range":[dom_rang], "topology":[topo], "architecture":AXTFFid,
                                            "ip_range":interpro_rang, "ip_topology":interpro_topo, "ip_name":interpro_name, "ip_model":interpro_model,
                                            "pfam_range":pfam_rang, "pfam_topology":pfam_topo, "pfam_name":pfam_name, "pfam_model":pfam_model}
                    else:
                        # Else, if the web scrape was unsuccessful, save only the DomainMapper data.
                        dom_map[uniprot] = {"range":[dom_rang], "topology":[topo], "architecture":AXTFFid}

                # Else, if this is not the first time encountering a protein,
                #   then the web scrape was already performed AND the DomainMapper data was initialized.
                # Therefore, append only the new DomainMapper data.
                else:
                        dom_map[uniprot]["range"].append(dom_rang)
                        dom_map[uniprot]["topology"].append(topo)
                        dom_map[uniprot]["architecture"].append(AXTFFid)
    return dom_map

def add_interpro_ranges(uniprot):
    find_interpro_url = lambda x: "https://www.ebi.ac.uk/interpro/wwwapi//entry/all/protein/reviewed/"+x

    respond = requests.get(find_interpro_url(uniprot), allow_redirects=False)

    try:
        data = json.loads(respond.text)
    except:
        return None

    try:
        ip_dom_rng_list = []
        ip_dom_topo_list = []
        ip_dom_name_list = []
        ip_dom_model_list = []
        pfam_dom_rng_list = []
        pfam_dom_topo_list = []
        pfam_dom_name_list = []
        pfam_dom_model_list = []

        if "results" in data.keys():
            for domain in data["results"]:

                if domain["metadata"]["source_database"] == "cathgene3d":
                    ip_name = domain["metadata"]["name"]
                    for prot in domain["proteins"]:
                        dom_rng = []
                        for prot_loc in prot["entry_protein_locations"]:
                            ip_model = prot_loc["model"]
                            frag_dom = []
                            for frag in prot_loc["fragments"]:
                                frag_dom.append([frag["start"], frag["end"]])
                            dom_rng.append(frag_dom)
                            if len(frag_dom) > 1:
                                ip_dom_topo_list.append("NC")
                            else:
                                ip_dom_topo_list.append("")
                            
                            ip_dom_model_list.append(ip_model)
                            ip_dom_name_list.append(ip_name)

                    ip_dom_rng_list+=(dom_rng)

                if domain["metadata"]["source_database"] == "pfam":
                    pfam_name = domain["metadata"]["name"]
                    for prot in domain["proteins"]:
                        dom_rng = []
                        for prot_loc in prot["entry_protein_locations"]:
                            pfam_model = prot_loc["model"]
                            frag_dom = []
                            for frag in prot_loc["fragments"]:
                                frag_dom.append([frag["start"], frag["end"]])
                            dom_rng.append(frag_dom)
                            if len(frag_dom) > 1:
                                pfam_dom_topo_list.append("NC")
                            else:
                                pfam_dom_topo_list.append("")
                            
                            pfam_dom_model_list.append(pfam_model)
                            pfam_dom_name_list.append(pfam_name)

                    pfam_dom_rng_list+=(dom_rng)
            
            return ip_dom_rng_list, ip_dom_topo_list, ip_dom_name_list, ip_dom_model_list, pfam_dom_rng_list, pfam_dom_topo_list, pfam_dom_name_list, pfam_dom_model_list
        else:
            return None
    except:
        return None

def get_pdb_structure(pdb_file):
    return PDBParser().get_structure(file = pdb_file, id = None)

def get_residues_from_range(pdb_struct, rng_list):
    dom_res = []
    residues = [res for res in pdb_struct.get_residues()]
    for i,dom_rng in enumerate(rng_list):
        seg_res = []
        for s, e in dom_rng:
            for res in residues:
                if res.id[1] >= s and res.id[1] <= e:
                    seg_res.append(res)
        dom_res.append(seg_res) 
    del residues

    return dom_res

def get_midpoint_residue(rng_list):
    mpr = []
    for i,dom_rng in enumerate(rng_list):
        tmp_mpr = []
        for rng in dom_rng:
            s, e = rng
            tmp_mpr.append((e+s)//2)
        mpr.append(tmp_mpr)
    
    return mpr

def get_com(res_list):
    m_sum = 0
    mr_sum = 0
    for res in res_list:
        for atom in res.get_atoms():
            m_sum += atom.mass
            mr_sum += np.multiply(atom.coord,atom.mass)
    
    return mr_sum/m_sum

def get_Rg(res_list):
    rc = get_com(res_list)
    m_sum = 0
    mrrc_sum = 0
    for res in res_list:
        for atom in res.get_atoms():
            m_sum += atom.mass
            mrrc_sum += np.multiply(
                            np.power(
                                np.linalg.norm(
                                    np.subtract(atom.coord,rc))
                            ,2)
                        ,atom.mass)
    
    return np.sqrt(mrrc_sum/m_sum)

if __name__ == "__main__":

    subtract = lambda x: x[1]-x[0]
    euclidean_distance = lambda x: np.linalg.norm(x)
    
    alphafold_path = ""
    find_af_uniprot = lambda x: alphafold_path+x+"-F1-model_v2.pdb"

    yeast_dommap = "" # DomainMapper Output Path

    # change this variable name for your organisms
    if not os.path.exists("dommap_annote.npy"):
        protein_dommap_annotations = read_dommap_tsv(yeast_dommap)
        np.save("dommap_annote.npy", protein_dommap_annotations, allow_pickle=True)
    else:
        protein_dommap_annotations = np.load("dommap_annote.npy",allow_pickle=True)
        protein_dommap_annotations = protein_dommap_annotations.item()


    # non-contiguous domain radius of gyration and center of mass
    nc_dom_rg = []; nc_dom_dcom = []; ib_dom_dcom = []; nc_dom_dmpr = []; ib_dom_dmpr = []
    nc_dom_dcomdmpr = []; ib_dom_dcomdmpr = []

    ip_nc_dom_rg = []; ip_nc_dom_dcom = []; ip_ib_dom_dcom = []; ip_nc_dom_dmpr = []; ip_ib_dom_dmpr = []
    ip_nc_dom_dcomdmpr = []; ip_ib_dom_dcomdmpr = []

    pfam_nc_dom_rg = []

    # contiguous domain radius of gyration and center of mass
    con_dom_rg = []; dec_dom_dcom = []; dec_dom_dmpr = []; dec_dom_dcomdmpr = []

    ip_con_dom_rg = []
    pfam_con_dom_rg = []

    for uniprot in protein_dommap_annotations.keys():

        af_pdb = find_af_uniprot(uniprot)

        if os.path.exists(af_pdb):

            pdb_sruct = get_pdb_structure(af_pdb)

            rng_list = protein_dommap_annotations[uniprot]["range"]
            rng_topo = protein_dommap_annotations[uniprot]["topology"]
            dom_rng_res = get_residues_from_range(pdb_sruct, rng_list)

            # Calculate Radius of Gyration for NC and CON domains as defined by Domain Mapper
            for dom_res, topo in zip(dom_rng_res, rng_topo):
                if topo == "NC":
                    RgN = get_Rg(dom_res)/len(dom_res)
                    N = len(dom_res)
                    nc_dom_rg.append(RgN)
                    if RgN > 0.08 and RgN < 0.09:
                        print(f"{uniprot} DM NC RgN : {RgN}, Len : {N}")
                else:
                    RgN = get_Rg(dom_res)/len(dom_res)
                    N = len(dom_res)
                    con_dom_rg.append(RgN)
                    if RgN > 0.12 and RgN < 0.13:
                        print(f"{uniprot} DM CON RgN : {RgN}, Len : {N}")

            for dom_rng, topo in zip(rng_list, rng_topo):
                if topo == "NC":
                    for rng_idx in range(len(dom_rng)-1):
                        NC_N_RNG = dom_rng[rng_idx]
                        NC_C_RNG = dom_rng[rng_idx+1]
                        IB_RNG = [dom_rng[rng_idx][-1]+1, NC_C_RNG[0]-1]
                        
                        # many redundant function calls for clarity, such the use of indexing
                        NC_N_STRUCT = get_residues_from_range(pdb_sruct, [[NC_N_RNG]])[0]
                        NC_N_COM = get_com(NC_N_STRUCT)
                        NC_C_STRUCT = get_residues_from_range(pdb_sruct, [[NC_C_RNG]])[0]
                        NC_C_COM = get_com(NC_C_STRUCT)
                        IB_STRUCT = get_residues_from_range(pdb_sruct, [[IB_RNG]])[0]
                        IB_COM = get_com(IB_STRUCT)

                        # calc the NC-NC stuff here... I know it is ugly but whatever
                        NC_NC_dMPR = subtract(*get_midpoint_residue([[NC_N_RNG, NC_C_RNG]]))
                        NC_NC_dCOM = euclidean_distance(subtract([NC_N_COM,NC_C_COM]))
                        NC_NC_dCOMdMPR = NC_NC_dCOM/NC_NC_dMPR

                        # calc the NC/IB - IB/NC stuff here
                        NC_IB_dMPR = subtract(*get_midpoint_residue([[NC_N_RNG, IB_RNG]]))
                        NC_IB_dCOM = euclidean_distance(subtract([NC_N_COM,IB_COM]))
                        NC_IB_dCOMdMPR = NC_IB_dCOM/NC_IB_dMPR

                        IB_NC_dMPR = subtract(*get_midpoint_residue([[IB_RNG, NC_C_RNG]]))
                        IB_NC_dCOM = euclidean_distance(subtract([IB_COM,NC_C_COM]))
                        IB_NC_dCOMdMPR = IB_NC_dCOM/IB_NC_dMPR

                        # saving NC_NC data
                        nc_dom_dcom.append(NC_NC_dCOM); nc_dom_dmpr.append(NC_NC_dMPR)
                        nc_dom_dcomdmpr.append(NC_NC_dCOMdMPR)

                        # saving both NC_IB and IB_NC data
                        ib_dom_dcom.append(IB_NC_dCOM); ib_dom_dmpr.append(IB_NC_dMPR)
                        ib_dom_dcom.append(NC_IB_dCOM); ib_dom_dmpr.append(NC_IB_dMPR)
                        ib_dom_dcomdmpr.append(NC_IB_dCOMdMPR)
                        ib_dom_dcomdmpr.append(IB_NC_dCOMdMPR)
                else:
                    if len(dom_rng) < 2: # prevents the inclusion of Circular Permutants in the Decoy analysis
                        # calculate decoy stuff
                        DOM_RNG_N, DOM_RNG_C = dom_rng[0]
                        BUFFER_RNG = 3*(DOM_RNG_C - DOM_RNG_N)//10 # 30% between the domain range
                        DOM_RNG_MID = np.random.randint(DOM_RNG_N+BUFFER_RNG, DOM_RNG_C-BUFFER_RNG)

                        DOM_RNG_NT = [DOM_RNG_N, DOM_RNG_MID]
                        DOM_RNG_CT = [DOM_RNG_MID, DOM_RNG_C]

                        DEC_N_STRUCT = get_residues_from_range(pdb_sruct, [[DOM_RNG_NT]])[0]
                        DEC_N_COM = get_com(DEC_N_STRUCT)
                        DEC_C_STRUCT = get_residues_from_range(pdb_sruct, [[DOM_RNG_CT]])[0]
                        DEC_C_COM = get_com(DEC_C_STRUCT)
                        
                        DEC_NC_dMPR = subtract(*get_midpoint_residue([[DOM_RNG_NT, DOM_RNG_CT]]))
                        DEC_NC_dCOM = euclidean_distance(subtract([DEC_N_COM,DEC_C_COM]))
                        DEC_NC_dCOMdMPR = DEC_NC_dCOM/DEC_NC_dMPR

                        dec_dom_dcom.append(DEC_NC_dCOM)
                        dec_dom_dmpr.append(DEC_NC_dMPR)
                        dec_dom_dcomdmpr.append(DEC_NC_dCOMdMPR)
        
            if "ip_range" in protein_dommap_annotations[uniprot].keys():
                # set up for interpro ranges
                ip_rng_list = protein_dommap_annotations[uniprot]["ip_range"]
                pfam_rng_list = protein_dommap_annotations[uniprot]["pfam_range"]

                ip_rng_topo = protein_dommap_annotations[uniprot]["ip_topology"]
                pfam_rng_topo = protein_dommap_annotations[uniprot]["pfam_topology"]

                ip_dom_rng_res = get_residues_from_range(pdb_sruct, ip_rng_list)
                pfam_dom_rng_res = get_residues_from_range(pdb_sruct, pfam_rng_list)

                # Calculate Radius of Gyration for NC and CON domains as defined by Interpr(Cathgene3D)
                for ip_dom_res,ip_dom_topo in zip(ip_dom_rng_res,ip_rng_topo):
                    if ip_dom_topo == "NC":
                        ip_nc_dom_rg.append(get_Rg(ip_dom_res)/len(ip_dom_res))
                    else:
                        ip_con_dom_rg.append(get_Rg(ip_dom_res)/len(ip_dom_res))

                # Calculate Radius of Gyration for NC and CON domains as defined by PFAM
                for pfam_dom_res,pfam_dom_topo in zip(pfam_dom_rng_res,pfam_rng_topo):
                    if pfam_dom_topo == "NC":
                        pfam_nc_dom_rg.append(get_Rg(pfam_dom_res)/len(pfam_dom_res))
                    else:
                        pfam_con_dom_rg.append(get_Rg(pfam_dom_res)/len(pfam_dom_res))

                # this only needs to happen for cathgene3d
                for dom_rng, topo in zip(ip_rng_list, ip_rng_topo):
                    if topo == "NC":
                        for rng_idx in range(len(dom_rng)-1):
                            NC_N_RNG = dom_rng[rng_idx]
                            NC_C_RNG = dom_rng[rng_idx+1]
                            IB_RNG = [dom_rng[rng_idx][-1]+1, NC_C_RNG[0]-1]
                            
                            # many redundant function calls for clarity, such the use of indexing
                            NC_N_STRUCT = get_residues_from_range(pdb_sruct, [[NC_N_RNG]])[0]
                            NC_N_COM = get_com(NC_N_STRUCT)
                            NC_C_STRUCT = get_residues_from_range(pdb_sruct, [[NC_C_RNG]])[0]
                            NC_C_COM = get_com(NC_C_STRUCT)
                            IB_STRUCT = get_residues_from_range(pdb_sruct, [[IB_RNG]])[0]
                            IB_COM = get_com(IB_STRUCT)

                            # calc the NC-NC stuff here... I know it is ugly but whatever
                            NC_NC_dMPR = subtract(*get_midpoint_residue([[NC_N_RNG, NC_C_RNG]]))
                            NC_NC_dCOM = euclidean_distance(subtract([NC_N_COM,NC_C_COM]))
                            NC_NC_dCOMdMPR = NC_NC_dCOM/NC_NC_dMPR

                            # calc the NC/IB - IB/NC stuff here
                            NC_IB_dMPR = subtract(*get_midpoint_residue([[NC_N_RNG, IB_RNG]]))
                            NC_IB_dCOM = euclidean_distance(subtract([NC_N_COM,IB_COM]))
                            NC_IB_dCOMdMPR = NC_IB_dCOM/NC_IB_dMPR

                            IB_NC_dMPR = subtract(*get_midpoint_residue([[IB_RNG, NC_C_RNG]]))
                            IB_NC_dCOM = euclidean_distance(subtract([IB_COM,NC_C_COM]))
                            IB_NC_dCOMdMPR = IB_NC_dCOM/IB_NC_dMPR

                            # saving NC_NC data
                            ip_nc_dom_dcom.append(NC_NC_dCOM); ip_nc_dom_dmpr.append(NC_NC_dMPR)
                            ip_nc_dom_dcomdmpr.append(NC_NC_dCOMdMPR)

                            # saving both NC_IB and IB_NC data
                            ip_ib_dom_dcom.append(IB_NC_dCOM); ip_ib_dom_dmpr.append(IB_NC_dMPR)
                            ip_ib_dom_dcom.append(NC_IB_dCOM); ip_ib_dom_dmpr.append(NC_IB_dMPR)
                            ip_ib_dom_dcomdmpr.append(NC_IB_dCOMdMPR)
                            ip_ib_dom_dcomdmpr.append(IB_NC_dCOMdMPR)
        else:
            pass
            # print(uniprot, "does not exisit in AlphaFold")

    # saving data for later use
    np.save("yeast_nc_Rg.npy", nc_dom_rg, allow_pickle=True)
    np.save("yeast_con_Rg.npy", con_dom_rg, allow_pickle=True)

    np.save("yeast_nc_dcomdmpr.npy", nc_dom_dcomdmpr, allow_pickle=True)
    np.save("yeast_ib_dcomdmpr.npy", ib_dom_dcomdmpr, allow_pickle=True)
    np.save("yeast_nc_dcom.npy", nc_dom_dcom, allow_pickle=True)
    np.save("yeast_ib_dcom.npy", ib_dom_dcom, allow_pickle=True)
    np.save("yeast_nc_dmpr.npy", nc_dom_dmpr, allow_pickle=True)
    np.save("yeast_ib_dmpr.npy", ib_dom_dmpr, allow_pickle=True)

    np.save("yeast_dec_nc_dcom.npy", dec_dom_dcom, allow_pickle=True)
    np.save("yeast_dec_nc_dmpr.npy", dec_dom_dmpr,allow_pickle=True)
    np.save("yeast_dec_nc_dcomdmpr.npy", dec_dom_dcomdmpr,allow_pickle=True)

    np.save("yeast_cg3d_con_Rg.npy", ip_con_dom_rg, allow_pickle=True)
    np.save("yeast_cg3d_nc_Rg.npy", ip_nc_dom_rg, allow_pickle=True)

    np.save("yeast_cg3d_nc_dcomdmpr.npy", ip_nc_dom_dcomdmpr, allow_pickle=True)
    np.save("yeast_cg3d_ib_dcomdmpr.npy", ip_ib_dom_dcomdmpr, allow_pickle=True)

    np.save("yeast_pfam_con_Rg.npy", pfam_con_dom_rg, allow_pickle=True)
    np.save("yeast_pfam_nc_Rg.npy", pfam_nc_dom_rg, allow_pickle=True)