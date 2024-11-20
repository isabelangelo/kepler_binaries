import pandas as pd
import numpy as np

########### Table 9 (Kraus KOI sample)
column_names = ['Name','CC','sep_arcsec','Pbin','Psurvey','Pphot','Pastro','Pfield']
kraus_koi_binaries = pd.read_csv(
      './data/literature_data/Kraus_KOI_sample/PaperPbinAll.txt', 
                 delimiter='&', names=column_names)
# add column with separation in mas to match other tables
kraus_koi_binaries['sep_mas'] = kraus_koi_binaries['sep_arcsec']*1000

# extract unresolved binaries
print(len(kraus_koi_binaries), 'total companions from Kraus 2024 Table 9')

kraus_koi_binaries = kraus_koi_binaries.query('Pbin>90')
print(len(kraus_koi_binaries), ' companions with Pbin>90%')

kraus_koi_binaries = kraus_koi_binaries.query('sep_arcsec<0.8')
print(len(kraus_koi_binaries), 'companions have sep_arcsec<0.8arcsec HIRES slit width')
print(len(np.unique(kraus_koi_binaries.Name.to_numpy())), 
      ' unique planet hosts')


# tables with sep/pa information from Kraus KOI sample
########### Table 3 (Keck/NIRC2 imaging)
img_comps_names = ['Name', 'CC#', 'Epoch', 'Filter', 'N', 'sep_mas_with_err', 'pa', \
                  'dmag_with_err', '?', 'PI']
ImgComps = pd.read_csv(
      './data/literature_data/Kraus_KOI_sample/PaperImgComps.txt', 
                       delimiter='&', names=img_comps_names)

psf_comps_names = ['Name', 'CC#', 'Epoch', 'Filter', 'N', 'sep_mas_with_err', 'pa', \
                  'dmag_with_err', '?', '??', 'PI']
PSFComps = pd.read_csv('./data/literature_data/Kraus_KOI_sample/PaperPSFComps.txt', 
                       delimiter='&', names=psf_comps_names)
kraus_koi_table3 = pd.concat((ImgComps, PSFComps[img_comps_names]))
# re-format table 3 to separate sep, dmag from their errors
sep_mas_values = [i.split('$\pm$') for i in kraus_koi_table3['sep_mas_with_err']]
kraus_koi_table3['sep_mas'] = [float(i[0]) for i in sep_mas_values]
kraus_koi_table3['sep_mas_err'] = [float(i[1]) if '*' not in i[1] else np.nan for i in sep_mas_values]
dmag_values = [i.split('$\pm$') for i in kraus_koi_table3['dmag_with_err']]
kraus_koi_table3['dmag'] = [float(i[0]) for i in dmag_values]
kraus_koi_table3['dmag_err'] = [float(i[1]) if '*' not in i[1] else np.nan for i in dmag_values]


########### Table 5 (imaging companions from literature)
lit_comps_names = ['Name','CC#','Telescope','Inst','Filt', 'Epoch','sep_mas','sep_err_mas',\
                  'pa','pa_err', 'dmag','dmag_err','Ref']
kraus_koi_table5 = pd.read_csv('./data/literature_data/Kraus_KOI_sample/PaperLitComps.txt', 
            delimiter='&', names=lit_comps_names)
# re-format table 5 so CC# matches other tables
kraus_koi_table5['CC#'] = [i.replace('0','') for i in kraus_koi_table5['CC#']]


########### Table 6 (companions from Kraus 2016)
# these are delta_Kmag, not dmag, so I'll just be sure to store that accordingly.
kraus_koi_table6 = pd.read_csv(
      './data/literature_data/Kraus_KOI_sample/PaperKraus2016Comps.txt', 
      delimiter=' ')

# fix formatting errors in target names
kraus_koi_binaries['Name'] = [i.replace(' ','') for i in kraus_koi_binaries['Name']]
kraus_koi_table3['Name'] = [i.replace(' ','') for i in kraus_koi_table3['Name']]
kraus_koi_table5['Name'] = [i.replace(' ','') for i in kraus_koi_table5['Name']]

# fix formatting errors in CC numbers
kraus_koi_binaries['CC'] = [i.replace(' ','') for i in kraus_koi_binaries['CC']]
kraus_koi_table3['CC#'] = [i.replace(' ','') for i in kraus_koi_table3['CC#']]
kraus_koi_table5['CC#'] = [i.replace(' ','') for i in kraus_koi_table5['CC#']]
kraus_koi_table3 = kraus_koi_table3.rename(columns={'CC#': 'CC'})
kraus_koi_table5 = kraus_koi_table5.rename(columns={'CC#': 'CC'})
kraus_koi_table6 = kraus_koi_table6.rename(columns={'CC#': 'CC'})


############# store companion separation, dmag for companions in binary sample #############

companion_columns = ['sep_mas', 'dmag', 'dKmag', 'source']
kraus_koi_binaries[companion_columns] = np.nan

for idx, row in kraus_koi_binaries.iterrows():
    comp_query = '(Name==@row.Name) & (CC==@row.CC)'
    comp_df_table3 = kraus_koi_table3.query(comp_query)
    comp_df_table5 = kraus_koi_table5.query(comp_query)
    comp_df_table6 = kraus_koi_table6.query(comp_query)
    
    # Table 3: Keck/NIRC2 companions
    if len(comp_df_table3)>0:
        kraus_koi_binaries.loc[idx, 'sep_mas'] = comp_df_table3.iloc[0].sep_mas
        kraus_koi_binaries.loc[idx, 'dmag'] = comp_df_table3.iloc[0].dmag
        kraus_koi_binaries.loc[idx, 'source'] = 'Keck/NIRC2'
        
    # Table 5: Literature copmanions
    elif len(comp_df_table5)>0:
        kraus_koi_binaries.loc[idx, 'sep_mas'] = comp_df_table5.iloc[0].sep_mas
        kraus_koi_binaries.loc[idx, 'dmag'] = comp_df_table5.iloc[0].dmag
        kraus_koi_binaries.loc[idx, 'source'] = 'Literature'
    
    # Table 6: Kraus 2016 companions
    elif len(comp_df_table6)>0:
        kraus_koi_binaries.loc[idx, 'sep_mas'] = comp_df_table6.iloc[0].sep_mas
        kraus_koi_binaries.loc[idx, 'dKmag'] = comp_df_table6.iloc[0].d_Kmag
        kraus_koi_binaries.loc[idx, 'source'] = 'Kraus 2016'

# write to .csv file
binary_sample_path = './data/literature_data/Kraus_KOI_sample/kraus_koi_unresolved_binaries.csv'
kraus_koi_binaries.to_csv(binary_sample_path)
print('binary sample + separation/dmag information saved to {}'.format(binary_sample_path))

# print names for jump query to get observation IDs
print(np.unique(np.array([i for i in kraus_koi_binaries.Name])))

