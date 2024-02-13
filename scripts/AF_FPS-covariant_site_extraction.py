#!/usr/bin/env python3

####################
# import libraries #
####################

import os
import sys
import pandas as pd
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
import concurrent.futures as cf

from pathlib import Path
from scipy.stats import spearmanr
from natsort import index_natsorted
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests

####################
# define globals #
####################

# Set color values
dutchfield = ["#e60049", "#0bb4ff", "#87bc45", "#ef9b20", "#b33dc6"]

springpastel = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a"] 

gray = 'lightgray'

dutchfield_colordict = {'S6R691V_her2': "#e60049", 'ANAB5F7_basal': "#0bb4ff", '98JKPD8_lumA': "#87bc45", 'PU24GB8_lumB': "#ef9b20", '2GAMBDQ_norm': "#b33dc6"}

gray_colordict = {'S6R691V_her2': gray, 'ANAB5F7_basal': gray, '98JKPD8_lumA': gray, 'PU24GB8_lumB': gray, '2GAMBDQ_norm': gray}

#############################
# define plotting functions #
#############################

def plot_jointplot(dataframe, motif_id, output_path):
	# plot scatter plot of AF_var vs FPS_scaled_var
	g = sns.jointplot(data=dataframe, x='AF', y='FPS_scaled', kind='scatter', hue='sample_id', height=12)
	plt.xlim(-0.1, 1.1)
	g.figure.suptitle(f"Jointpot of AF and scaled FPS values of {motif_id}", fontsize=14)
	# save the plot
	print(f'Saving {motif_id} jointplot of AF and scaled FPS...')
	g.savefig(f'{output_path}/output-data/plots/{motif_id}/{motif_id}_AF_vs_FPS-scaled_jointplot.pdf', dpi=150, bbox_inches="tight")
	# close the plot
	plt.close("all")
	del g
	print('Plot space closed and plots have been saved to file.')


#########################
# define util functions #
#########################

def process_input_tsv(root_dir):
	# Find all *.tsv files in root_dir
	target_dir = Path(root_dir)
	tsv_files = target_dir.glob('*.tsv')
	return tsv_files

def load_datatable(tsv_filepath):
	# import the data
	dt_afps = pd.read_csv(tsv_filepath, sep='\t')
	# extract motif id from filename
	motif_id = os.path.basename(tsv_filepath).replace('_fpscore-af-varsites-combined-matrix-wide.tsv', '')
	print(f'{motif_id} data table has been loaded.')
	
	# copy as a dataframe
	afps_df = dt_afps.filter(regex='_AF$|_fps$|_id$').copy()
	# convert to long format
	afps_df_long = afps_df.melt(id_vars=["region_id"], var_name="variable", value_name="value")
	# split the variable column into sample_id and type columns using reverse split string method, which returns a dataframe of columns based on the number of splits (n=x); this can directly be assigned to new columns in the original dataframe
	afps_df_long[['sample_id', 'type']] = afps_df_long['variable'].str.rsplit('_', n=1, expand=True)
	# drop the redundant 'variable' column
	afps_df_long = afps_df_long.drop(columns=["variable"])
	# now pivot the dataframe to create new columns based on the type column
	afps_df_lpv = afps_df_long.pivot(index=['region_id', 'sample_id'], columns='type', values='value').reset_index()
	# remove the index name and rename the columns to match the type values
	afps_df_lpv = afps_df_lpv.rename_axis(None, axis=1).rename(columns={'fps': 'FPS'})
	# sort the dataframe by region_id naturally
	afps_df_lpv = afps_df_lpv.reindex(index=index_natsorted(afps_df_lpv['region_id']))
	afps_df_lpv = afps_df_lpv.reset_index(drop=True)
	print(f'{motif_id} matrix has been loaded and converted to long format.')
	return dt_afps, motif_id, afps_df_lpv

def scale_merge_data(dt_afps, afps_df_lpv, motif_id, output_path):
	# scale the FPS values to a range of 0-1
	# Initialize a MinMaxScaler
	scaler = MinMaxScaler()
	# copy df
	fps_df_scaled = dt_afps.filter(regex='_fps$|_id$').copy()
	# set the index to 'region_id'
	fps_df_scaled = fps_df_scaled.set_index('region_id')
	# Fit the MinMaxScaler to the 'FPS' column and transform it
	fps_df_scaled = pd.DataFrame(scaler.fit_transform(fps_df_scaled), columns=fps_df_scaled.columns, index=fps_df_scaled.index)
	# rename columns by adding '_scaled' to the column names
	fps_df_scaled = fps_df_scaled.add_suffix('_scaled')
	##### Now convert to long format #####
	# reset index
	fps_df_scaled_long = fps_df_scaled.reset_index()
	# convert to long format
	fps_df_scaled_long = fps_df_scaled_long.melt(id_vars=["region_id"], var_name="variable", value_name="value")
	# split the variable column into sample_id and type columns using reverse split string method, which returns a dataframe of columns based on the number of splits (n=x); this can directly be assigned to new columns in the original dataframe
	# Split the 'variable' column into three parts
	fps_df_scaled_long[['part1', 'part2', 'part3']] = fps_df_scaled_long['variable'].str.rsplit('_', n=2, expand=True)
	# Assign part1 to 'sample_id' and concatenate the other parts to form 'type'
	fps_df_scaled_long['sample_id'] = fps_df_scaled_long['part1']
	fps_df_scaled_long['type'] = fps_df_scaled_long['part2'].str.upper() + '_' + fps_df_scaled_long['part3']
	# Drop the unnecessary columns
	fps_df_scaled_long = fps_df_scaled_long.drop(['variable', 'part1', 'part2', 'part3'], axis=1)
	# now pivot the dataframe to create new columns based on the type column
	fps_df_scaled_lpv = fps_df_scaled_long.pivot(index=['region_id', 'sample_id'], columns='type', values='value').reset_index()
	# remove the index name and rename the columns to match the type values
	fps_df_scaled_lpv = fps_df_scaled_lpv.rename_axis(None, axis=1)
	# sort the dataframe by region_id naturally
	fps_df_scaled_lpv = fps_df_scaled_lpv.reindex(index=index_natsorted(fps_df_scaled_lpv['region_id']))
	fps_df_scaled_lpv = fps_df_scaled_lpv.reset_index(drop=True)

	# merge the AF-FPS and FPS-scaled dataframes on region_id and sample_id
	afps_full_dfl = afps_df_lpv.merge(fps_df_scaled_lpv, on=['region_id', 'sample_id'])
	print('Dataframes have been scaled and merged.')
	# print('Plotting jointplot...')
	# plot_jointplot(afps_full_dfl, motif_id, output_path)
	return fps_df_scaled, fps_df_scaled_lpv, afps_full_dfl

def filter_zero(afps_full_dfl):
	# filter out unique region_id rows that have fps == 0 across the sample_ids and AF == 0
	# group by 'region_id' first  
	merged_filt_dfl = afps_full_dfl.groupby('region_id').filter(lambda x: x['FPS'].sum() > 0 and x['AF'].sum() > 0)
	return merged_filt_dfl

def calculate_variance(dt, fps_df_scaled, motif_id, merged_filt_dfl):
	# calculate the variance of AF and FPS scaled values across sample_ids per region_id
	print(f'Wrangling {motif_id} data for calculations...')
	# extract af columns
	af_df = dt.filter(regex='_AF$|_id$').copy()
	# print af_df length
	print(f'Length of {motif_id} original data table: {len(af_df)}')
	print(f'Subsetting {motif_id} data table with no-zero merged long table...')
	# filter out the rows whose `region_id` is not in the `region_id` column of `merged_filt`; reduce the repeated `region_id` column to unique values in `merged_filt` into an array
	merged_filt_uniq_regid = merged_filt_dfl['region_id'].unique()
	# then using this array, keep only the rows in the af_df that have `region_id` values in the array
	af_df_filt = af_df[af_df['region_id'].isin(merged_filt_uniq_regid)]
	# print af_df_filt length
	print(f'Length of {motif_id} filtered data table: {len(af_df_filt)}')
	# set the index to 'region_id'
	af_df_filt_idx = af_df_filt.set_index('region_id')
	# calculate variance of af values across samples per region_id and add to a new column called 'af_var'
	print(f'Calculating {motif_id} AF variances...')
	af_df_filt_idx['AF_var'] = af_df_filt_idx.var(axis=1)

	# do the same for fps scaled values
	fps_df_scaled_filt = fps_df_scaled[fps_df_scaled.index.isin(merged_filt_uniq_regid)]
	# print fps_df_scaled_filt length
	print(f'Length of {motif_id} filtered FPS-scaled data table: {len(fps_df_scaled_filt)}')
	fps_df_scaled_filt_idx = fps_df_scaled_filt.copy()
	# calculate variance of fps_scaled values across samples per region_id and add to a new column called 'fps_scaled_var'
	print(f'Calculating {motif_id} FPS_scaled variances...')
	fps_df_scaled_filt_idx['FPS_scaled_var'] = fps_df_scaled_filt_idx.var(axis=1)
	return af_df_filt_idx, fps_df_scaled_filt_idx

def merged_stats_df(af_df_filt_idx, fps_df_scaled_filt_idx, merged_filt_dfl):
	# merge the stats columns
	fps_sc_filt_var = fps_df_scaled_filt_idx.filter(regex='_var$|_id$').copy()
	af_filt_var = af_df_filt_idx.filter(regex='_var$|_id$').copy()
	# merge on region_id index from both tables
	merged_var = af_filt_var.merge(fps_sc_filt_var, left_index=True, right_index=True)
	# set index to 'region_id'
	merged_filt_dfl_idx = merged_filt_dfl.set_index('region_id')
	# merge with merged_var
	merged_stat = merged_filt_dfl_idx.merge(merged_var, left_index=True, right_index=True)
	# rearrange columns
	merged_stat = merged_stat[['sample_id', 'AF', 'FPS_scaled', 'AF_var', 'FPS_scaled_var']]
	return merged_stat
	
def get_covariant_sites(merged_stat, motif_id, output_path):
	print('Getting unique region IDs and extracting only AF_var and FPS_scaled_var columns...')
	# subset merged_stat
	merged_stat_vars = merged_stat[['AF_var', 'FPS_scaled_var']].copy().drop_duplicates()
	print(f'Length of {motif_id} merged_stat_vars: {len(merged_stat_vars)}')
	# now calculate the IQR for AF_var and FPS_scaled_var separately
	# calculate IQR for AF_var
	q1_vaf = merged_stat_vars['AF_var'].quantile(0.25)
	q3_vaf = merged_stat_vars['AF_var'].quantile(0.75)
	iqr_vaf = q3_vaf - q1_vaf
	lower_bound_outliers_vaf = q1_vaf - (1.5 * iqr_vaf)
	upper_bound_outliers_vaf = q3_vaf + (1.5 * iqr_vaf)

	# calculate IQR for FPS_scaled_var
	q1_vfps = merged_stat_vars['FPS_scaled_var'].quantile(0.25)
	q3_vfps = merged_stat_vars['FPS_scaled_var'].quantile(0.75)
	iqr_vfps = q3_vfps - q1_vfps
	lower_bound_outliers_vfps = q1_vfps - (1.5 * iqr_vfps)
	upper_bound_outliers_vfps = q3_vfps + (1.5 * iqr_vfps)

	print(f'Outlier bounds for {motif_id} AF variance: {lower_bound_outliers_vaf, upper_bound_outliers_vaf}')

	print(f'Outlier bounds for {motif_id} FPS_scaled variance: {lower_bound_outliers_vfps, upper_bound_outliers_vfps}')

	# using the outlier of only the upper bound, get the region IDs that are outliers in the AF_var and FPS_scaled_var columns

	outlier_af_fps_vars = merged_stat_vars[(merged_stat_vars['AF_var'] > upper_bound_outliers_vaf) & (merged_stat_vars['FPS_scaled_var'] > upper_bound_outliers_vfps)]

	print(f'Length of {motif_id} outlier_af_fps_vars: {len(outlier_af_fps_vars)}')

	# now extract the unique region IDs as a list from the outlier_af_fps_vars
	outliers_list = outlier_af_fps_vars.index.tolist()

	# subset the merged_stat dataframe to get the highly covariant sites (outlier sites)
	covar_sites = merged_stat[merged_stat.index.isin(outliers_list)]
	
	# sort the outlier sites by descending order of AF_var and FPS_scaled_var
	covar_sites_sorted = covar_sites.sort_values(by=['AF_var', 'FPS_scaled_var'], ascending=[False, False])
	print(f'Number of unique {motif_id} covar_sites: {len(covar_sites_sorted)/5}')

	# save to file
	print(f'Saving {motif_id} covariant sites to file...')
	covar_sites.to_csv(f'{output_path}/covariant-sites/{motif_id}_covariant_sites.tsv', sep='\t', index=True)

	return covar_sites_sorted

def test_correlation_spearman(covar_sites_sorted, motif_id, output_path):
	# test for Spearman correlation between AF_var and FPS_scaled_var for each covariant site across sample_ids

	# drop variance columns
	covar_sites_sorted_novars = covar_sites_sorted.drop(columns=['AF_var', 'FPS_scaled_var'])
	# reset index
	covar_sites_sorted_novars = covar_sites_sorted_novars.reset_index()
	# group by region_id and calculate spearman correlation
	correlations = covar_sites_sorted_novars.groupby('region_id').apply(lambda group: spearmanr(group['AF'], group['FPS_scaled']), include_groups=False)
	print(f'Testing for correlation between AF_var and FPS_scaled_var for {motif_id}...')
	# that returned a pandas Series, so convert to dataframe
	correlations_df = pd.DataFrame(correlations, columns=['corr_coeff_and_pvalue'])
	# split the 'corr_coeff_and_pvalue' column into two separate columns using apply() and pd.Series
	correlations_df[['corr_coeff', 'pvalue']] = correlations_df['corr_coeff_and_pvalue'].apply(pd.Series)
	# drop the original column
	correlations_df = correlations_df.drop(columns=['corr_coeff_and_pvalue']).reset_index()
	# sort region_ids naturally
	correlations_df_sorted = correlations_df.reindex(index=index_natsorted(correlations_df['region_id']))
	# reset index
	corr_df_allcovarsites = correlations_df_sorted.reset_index(drop=True)

	# save to file
	print(f'Saving {motif_id} correlation test results to file...')
	corr_df_allcovarsites.to_csv(f'{output_path}/correlation-tests/{motif_id}_correlation_test_results.tsv', sep='\t', index=False)

	return corr_df_allcovarsites

def correct_for_fdr(corr_df_allcovarsites, motif_id, output_path):
	# perform FDR correction on the p-values
	print(f'Performing FDR correction on {motif_id} p-values...')
	# extract the p-values
	pvalues = corr_df_allcovarsites['pvalue']
	# perform FDR correction
	fdr_corrected = multipletests(pvalues, alpha=0.05, method='fdr_bh')
	# add the corrected p-values to the dataframe
	corr_df_allcovarsites['adj_pvalues'] = fdr_corrected[1]

	# save to file
	print(f'Saving {motif_id} FDR corrected p-values to file...')
	corr_df_allcovarsites.to_csv(f'{output_path}/correlation-tests/{motif_id}_correlation_test_results_fdr-corrected.tsv', sep='\t', index=False)

	# filter for significant correlations
	significant_corr = corr_df_allcovarsites[corr_df_allcovarsites['adj_pvalues'] < 0.05]
	print(f'Number of significant correlations for {motif_id}: {len(significant_corr)}')
	# save to file
	print(f'Saving {motif_id} significant correlations to file...')
	significant_corr.to_csv(f'{output_path}/correlation-tests/{motif_id}_correlation_test_restuls_significant.tsv', sep='\t', index=False)

def process_data(tsv_filepath, output_path):
	# load the data
	dt_afps, motif_id, afps_df_lpv = load_datatable(tsv_filepath)
	# scale and merge the data
	fps_df_scaled, _, afps_full_dfl = scale_merge_data(dt_afps, afps_df_lpv, motif_id, output_path)
	# filter out unique region_id rows that have fps == 0 across the sample_ids and AF == 0
	merged_filt_dfl = filter_zero(afps_full_dfl)
	# calculate variance of AF and FPS scaled values across sample_ids per region_id
	af_df_filt_idx, fps_df_scaled_filt_idx = calculate_variance(dt_afps, fps_df_scaled, motif_id, merged_filt_dfl)
	# merge the stats columns
	merged_stat = merged_stats_df(af_df_filt_idx, fps_df_scaled_filt_idx, merged_filt_dfl)
	# get covariant sites
	covar_sites_sorted = get_covariant_sites(merged_stat, motif_id, output_path)
	# test for correlation between AF_var and FPS_scaled_var for each covariant site across sample_ids
	corr_df_allcovarsites = test_correlation_spearman(covar_sites_sorted, motif_id, output_path)
	# perform FDR correction on the p-values
	correct_for_fdr(corr_df_allcovarsites, motif_id, output_path)
	print(f'Processing of {motif_id} data is complete.')


##################
# load arguments #
##################
# check for the required arguments
if len(sys.argv) < 3:
	print(f'ERROR: Missing required arguments!')
	print(f'USAGE: python3 AF_FPS_covariant_site_extraction.py <directory where the motif matrix tsv files are stored> <top directory for output files>')
	sys.exit(1)
else:
    root_dir = sys.argv[1]
    output_dir = sys.argv[2]

# ##################
# # define globals #
# ##################

if __name__ == '__main__':
	inputs = process_input_tsv(root_dir)
	# uncomment this to run serially
	# for target_file in inputs:
	# 	process_data(target_file, output_dir, 'iqr', True)

	# uncomment this to run in parallel
	with cf.ProcessPoolExecutor(max_workers=8) as executor:
		executor.map(process_data, inputs, it.repeat(output_dir))

	print ("Pipeline finished! All footprint matrices have been processed.")
