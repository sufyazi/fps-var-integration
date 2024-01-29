#!/usr/bin/env python3

####################
# import libraries #
####################
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

##################
# load arguments #
##################
# check for the required arguments
if len(sys.argv) < 2:
	print(f'ERROR: Missing required arguments!')
	print(f'USAGE: python3 AF_FPS_calculate-adjusted-multitest.py <root_dir> <output_path>')
	sys.exit(1)
else:
	root_dir = sys.argv[1] # where the motif ID table directories are
	output_path = sys.argv[2]

##################
# define globals #
##################

if __name__ == '__main__':
    
	# initialize a dictionary to store the statistics
	statistic_motifs = {}

	# initialize a list to store the pvalues
	pvalues_list = []

	# Find all motif directories in the root directory
	target_dir = Path(root_dir)
	print(f'Currently processing motifs in target directory: {target_dir}')
	files = target_dir.glob('*_AF-FPS_region_contingency_table.tsv')
	for i, m in enumerate(files):
		# get the filename without the suffix
		filename = m.stem

		# get motif id
		motif_id = filename.replace('_AF-FPS_region_contingency_table', '')
		print(f'Processing motif {i+1}: {motif_id}...')

		# load the contingency table
		df = pd.read_csv(m, sep="\t", index_col=0)
		print(df.head())

		# calculate the fisher exact test
		ber = stats.fisher_exact(df)
		statistic, pvalue = ber.statistic, ber.pvalue
		print(f'Fisher exact test for {motif_id}: {statistic};{pvalue}')

		# add the statistic and pvalue to the dictionary
		statistic_motifs[motif_id] = [statistic, pvalue]

		# also add the pvalue to the list
		pvalues_list.append(pvalue)

	# calculate the adjusted pvalues
	print(f'Calculating adjusted pvalues...')
	np_pvalues = np.array(pvalues_list)
	adjusted_pvalues = stats.false_discovery_control(np_pvalues, method='bh')

	# initialize a dictionary to store the adjusted pvalues with motif ids
	adjusted_pvalues_dict = {}

	for i, motif_id in enumerate(statistic_motifs.keys()):
		# get the adjusted pvalue
		adjusted_pvalue = adjusted_pvalues[i]
		print(f'Adjusted pvalue for {motif_id}: {adjusted_pvalue}')

		# add the adjusted pvalue to the dictionary
		adjusted_pvalues_dict[motif_id] = adjusted_pvalue
	
	# save the dictionary of adjusted pvalues to file
	print(f'Finished processing all motifs in target directory: {target_dir}')
	print(f'Saving the dictionary of ALL adjusted pvalues to file...')
	with open(f'{output_path}/AF-FPS_adjusted_pvalues_dictionary_ALL.tsv', 'w') as f:
		for key in adjusted_pvalues_dict.keys():
			f.write("%s\t%s\n"%(key,adjusted_pvalues_dict[key]))
	
	# now filter the adjusted pvalues to only include those that are significant
	# initialize a dictionary to store the significantly mutated motifs
	significant_motifs = {}

	for i, motif_id in enumerate(adjusted_pvalues_dict.keys()):
		# get the adjusted pvalue
		adjusted_pvalue = adjusted_pvalues_dict[motif_id]

		if adjusted_pvalue < 0.05:
			print(f'Adjusted pvalue for {motif_id} is significant. Adding to the dictionary of significantly mutated motifs...')
			significant_motifs[motif_id] = adjusted_pvalue
		else:
			print(f'Adjusted pvalue for {motif_id} is not significant. Skipping...')

	# save the dictionary of significant motifs to file
	print(f'Finished processing all motifs in target directory: {target_dir}')
	print(f'Saving the dictionary of significant motifs to file...')
	with open(f'{output_path}/AF-FPS_adjusted_pvalues_dictionary_SIG.tsv', 'w') as f:
		for key in significant_motifs.keys():
			f.write("%s\t%s\n"%(key,significant_motifs[key]))
	print('Done!')

	