if __name__ == '__main__':

    import pandas as pd
    import os                

    api_key = 'qVMjh9NwNhVq0tNzLpSZmjZIHzoSamsA'

    file_name = 'df_all.parquet'
    file_path = os.path.join('msc_thesis', 'data', file_name)

    # Read the parquet file
    df = pd.read_parquet(file_path)
    df = df.drop(8)
    df = df.drop('composition', axis=1)
    df['alloy'] = df['alloy'].str.replace('-', '')

    #print(df.alloy.iloc[0])
    #alloys = df[['alloy']]

    from matminer.featurizers.composition import ElementFraction, ElementProperty
    #df = StrToComposition().featurize_dataframe(df, "alloy")
    #ef = ElementFraction()
    #X = ef.featurize_dataframe(df, 'alloy')

    from sample import ml
    df = ml.make_composition(df)

    ep_feat = ElementProperty(data_source="pymatgen", features=["bulk_modulus"], stats = ["mean"])
    df = ep_feat.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer

    # %% Derive additional features from composition

    # Drop the non-numerical feature "composition"
    df = df.drop(["composition"], axis=1)


    ep = ElementProperty(data_source="magpie",features=["GSbandgap"],stats=["mean"])
    #df = ep.featurize_dataframe(df, 'Al', 'Cr', 'Fe','Mo', 'Nb', 'O', 'Ta', 'Sn', 'Ti', 'W', 'V', 'Zr')

    from matminer.data_retrieval.retrieve_MP import MPDataRetrieval


