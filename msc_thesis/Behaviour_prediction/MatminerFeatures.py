if __name__ == '__main__':

    import pandas as pd
    import os                

    api_key = 'qVMjh9NwNhVq0tNzLpSZmjZIHzoSamsA'

    file_name = 'df_all.parquet'
    file_path = os.path.join('msc_thesis', 'data', file_name)

    # Read the parquet file
    df = pd.read_parquet(file_path)

    # Drop unwanted columns 
    df = df.drop(8)
    df = df.drop(['composition', 'Ms', 'Deformation_mechanism', 'Deformation_mechanism_simple', 'LookIntoBehaviour', 'Microstructure'], axis=1)

    from matminer.featurizers.composition import ElementProperty
    from sample import ml
    df = ml.make_composition(df)

    ep_feat = ElementProperty(data_source="pymatgen", features=["X",
                #"row",
                #"group",
                #"block",
                "atomic_mass",
                "atomic_radius",
                #"mendeleev_no",
                #"electrical_resistivity",
                "velocity_of_sound",
                "thermal_conductivity",
                "melting_point",
                #"bulk_modulus",
                #"coefficient_of_linear_thermal_expansion"
                ], stats = ["mean"])
    df = ep_feat.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer

    # Derive additional features from composition
    ep_feat2 = ElementProperty(data_source="deml", features=[#"atom_num",
                #"atom_mass",
                #"row_num",
                #"col_num",
                #"atom_radius",
                "molar_vol",
                "heat_fusion",
                #"melting_point",
                "boiling_point",
                "heat_cap",
                "first_ioniz",
                "electronegativity",
                "electric_pol",
                "GGAU_Etot",
                "mus_fere",
                "FERE correction",], stats = ["mean"])
    df = ep_feat2.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer

    ep_feat3 = ElementProperty(data_source="magpie", features=[#"Number",
                #"MendeleevNumber",
                #"AtomicWeight",
                #"MeltingT",
                #"Column",
                #"Row",
                "CovalentRadius",
                #"Electronegativity",
                "NsValence",
                "NpValence",
                "NdValence",
                "NfValence",
                "NValence",
                "NsUnfilled",
                "NpUnfilled",
                "NdUnfilled",
                "NfUnfilled",
                "NUnfilled",
                "GSvolume_pa",
                "GSbandgap",
                "GSmagmom",
                "SpaceGroupNumber",], stats = ["mean"])
    df = ep_feat3.featurize_dataframe(df, col_id="composition")

    # Drop the non-numerical feature "composition"
    df = df.drop(["composition"], axis=1)
    print(df.head())

    # Save the dataframe to a file
    df.to_pickle('df_matminer.pkl')

