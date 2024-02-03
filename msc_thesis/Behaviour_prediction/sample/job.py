# Library job

import os
if __name__ == "__main__":
    print("This is library ""job"" containing different ML parameter sets.")
elif __name__ == "sample.job":
    print("Loaded library job")



def ini_data(label):
    if label == 'Matteo_Austenites':
        ini = dict(xls_file=os.path.join('raw_data','Mechnical_properties_austenites_matv_20191012.xlsx'),
                   xls_sheet_nr=0,
                   features_to_delete=['nr', 'alloy', 'N+C', "Rp02%_MPa", 'Rm_MPa', 'A_%', 'Z_%', 'E_GPa', 'G_GPa',
                                       'CTE', 'Notes', 'calc1', 'calc2', 'calc1 v2', 'Ref.', 'is_ready','is_at_perc'],
                   target_feature="Rp02%_MPa",
                   is_at_perc='is_at_perc',
                   use_data='is_ready',
                   use_composition_as_feature=True,
                   use_element_properties_as_feature=True,
                   element_properties_database='magpie',
                   element_properties=["MeltingT", "AtomicWeight", "Electronegativity", "NdValence",
                                       "NdUnfilled", "GSvolume_pa", "GSbandgap", "GSmagmom"])
    elif label == 'Matteo_SFEs':
        ini = dict(xls_file=os.path.join('raw_data','SFE_data_3_FN.xlsx'),
                   xls_sheet_nr=1,
                   features_to_delete=["nr", "is_ready", "method", "is_at_perc", "ref_label", "SFE_Gb_norm",
                                       "std_Gb_norm", "SFE", "std"],
                   target_feature="SFE_Gb_norm",
                   is_at_perc='is_at_perc',
                   use_data='is_ready',
                   use_composition_as_feature=False,
                   use_element_properties_as_feature=True,
                   element_properties_database='magpie',
                   element_properties=["MeltingT", "AtomicWeight", "Electronegativity", "NdValence",
                                       "NdUnfilled", "GSvolume_pa", "GSbandgap", "GSmagmom"])
    else:
        raise NameError("Dataset not found")
    return ini