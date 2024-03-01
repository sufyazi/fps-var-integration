#!/usr/bin/env python3

####################
# import libraries #
####################
import os
import sys
import pandas as pd
from natsort import order_by_index, index_natsorted
####################

# check if the input directory (correlation-tests sig tsvs) and output directory are provided
if len(sys.argv) != 3:
    print("Usage: python3 AF_FPS-covariant_site_count_merge.py <input_dir> <output_dir>")
    sys.exit(1)

# set the input directory (where the correlation-tests directory is located)
input_dir = sys.argv[1]

# grab all the files with the name *significant.tsv
files = [f for f in os.listdir(input_dir) if f.endswith("sig.tsv")]
print(len(files))

# initialize the master df
master_df = pd.DataFrame()

# load each file into a df, if the df is empty, skip it
for index, file in enumerate(files):
    # get the file path
    filepath = os.path.join(input_dir, file)
    # extract motif id from filename
    motif_id = file.replace('_correlation_test_results_fdr-corrected_sig.tsv', '')
    print(f"Processing file no. {index+1} of {len(files)} files...")
    print(f"Motif ID: {motif_id}")
    # load the file into a df
    print(f"Loading {file} into a df...")
    df = pd.read_csv(filepath, sep="\t")
    # if the df is empty, skip it
    if df.empty:
        print(f"{file} no. {index+1} is empty, skipping...")
        continue
    else:
        # rename the original index column
        print(f"Renaming original index column...")
        df = df.rename(columns={'Unnamed: 0': 'orig_index'})
        # rename pvalue column
        print(f"Renaming pvalue column...")
        df = df.rename(columns={'pvalue': 'pvalues'})
        # add the motif id to the df
        print(f"Adding motif id to the df...")
        df["motif_id"] = motif_id
        # then move the motif id to the second column
        df = df[['orig_index', 'motif_id', 'region_id', 'corr_coeff', 'pvalues', 'adj_pvalues']]
        print(df.head())

        # merge the current df with the master df
        print(f"Now merging {file} df with the master df...")
        master_df = pd.concat([master_df, df], ignore_index=True)

# print the master df
print(master_df.head())
print(master_df.shape)

# sort the master df by motif id and then by region id, preserving the sorting of motif id
print("Sorting the master df by motif id and region id...")

# Sort by 'motif_id' first
master_df = master_df.sort_values(by=['motif_id', 'region_id'])

# master_df['region_id'] = master_df['region_id'].astype(str)
# master_df = master_df.iloc[order_by_index(master_df.index, index_natsorted(master_df['region_id']))]

# Reset the index
master_df = master_df.reset_index(drop=True)

print(master_df.head())

# save the master df to a file
print("Saving the master df to a file...")
output_dir = sys.argv[2]
output_file = f"{output_dir}/AF_FPS-covariant_sites_significant.combined.tsv"

master_df.to_csv(output_file, sep="\t", index=False)



        