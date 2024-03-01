#!/usr/bin/env python3

####################
# import libraries #
####################

import os
import sys
import logging
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
	logging.info(f'Saving {motif_id} jointplot of AF and scaled FPS...')
	g.savefig(f'{output_path}/output-data/plots/{motif_id}/{motif_id}_AF_vs_FPS-scaled_jointplot.pdf', dpi=150, bbox_inches="tight")
	# close the plot
	plt.close("all")
	del g
	logging.info('Plot space closed and plots have been saved to file.')


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
	logging.info(f'{motif_id} data table has been loaded.')
	
	# copy as a dataframe
	afps_df = dt_afps.filter(regex='_AF$|_fps$|_id$').copy()
	

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
	logging.info(f'Processing of {motif_id} data is complete.')


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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
