from setup import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # extract the data
    # extract sample ids and locations
    xl_all = pd.ExcelFile('iron_data/chemical_analyses.xlsx')
    df_all = xl_all.parse('Fe contents')
    df_iron = df_all.filter(['Sample','Weight','Fe_kg'], axis=1)
    df_iron.drop(df_iron.index[:36], axis=0, inplace=True)
    df_iron['Fe_kg'] = df_iron['Fe_kg'].astype('float')
    df_iron[["Day", "Level", "TechRep"]] = df_iron['Sample'].str.split('-', expand=True)
    df_iron.drop(['Sample'], axis=1)
    df_clean = pd.DataFrame(columns=['Fe'])
    df_clean['Fe'] = df_iron.groupby(['Level', 'Day'])['Fe_kg'].mean()
    # extract ARGs data

    table_full = {}
    # table_full[0] = xl_args.parse('1A-128A', parse_cols=[2, 3, 5])
    # table_full[1] = xl_args.parse('129A-256A', parse_cols=[2, 3, 5])
    # table_full[2] = xl_args.parse('257A-367A', parse_cols=[2, 3, 5])
    # df = pd.concat(table_full.values(), ignore_index=True)
    # clean afterwards because column parsing did not work
    # df.drop(['Row', 'Column', 'Conc', 'Efficiency', 'Flags'], axis=1, inplace=True)
    # # split Sample names into a number matching Sian's notation and tech replicate number
    # df[["Sample", "TechRep"]] = df['Sample'].str.split('A-', expand=True)
    # df.dropna(subset=['Ct'], inplace=True)
    #
    # # match names of Assays with genes
    # xl_assays = pd.ExcelFile('heap_data/assay_names.xlsx')
    # df_assays = xl_assays.parse()
    # df_assays.drop(['Count', 'Added Info'], axis=1, inplace=True)