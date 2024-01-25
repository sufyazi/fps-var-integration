# TF Footprinting & Noncoding Mutation Data Integration
Suffian Azizan

*Note: This is a Github README doc that is dynamically generated from
the `qmd` notebook.*

## Transcription Factor (TF) Footprinting Scores

TF footprints are scored using
[TOBIAS](https://github.molgen.mpg.de/pages/loosolab/www/software/TOBIAS/)
program from Looso Lab, Max Plank Institute. The TOBIAS pipeline was run
on individual samples so TOBIAS’s internal intersample normalization was
not applied. The raw footprint scores were collated for all samples in
the pan-cancer study and combined into a large dense matrix containing
unique footprint sites of a particular motif of interest across all
samples. As we have 1360 motifs of interest in this study, we have 1360
large matrices to process.

For demonstration purposes, a subset of TF footprint data on the open
chromatin regions in only breast cancer samples from TCGA database was
used here to save storage space and lower processing overhead.
Additionally, only 1 TF motif will be presented here in the analysis
workflow.

### Loading the footprint matrix

First, load up required Python packages.

``` python
import os
import textwrap
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from natsort import index_natsorted
```

Now, we can import the matrix into a Pandas dataframe. The matrix is
stored in a tab-delimited text file (`.tsv`). The first and second
column contains the genomic coordinates of the footprint site, the third
column contains the footprint score, and the third column contains the
sample ID. The first row contains the TF motif name. The data is loaded
into a Pandas dataframe and the first 5 rows are displayed.

``` python
# import the data
filepath = '../demo-data/E2F2_E2F2_HUMAN.H11MO.0.B_BRCA-subtype-vcf-filtered-matrix.txt'
matrix_afps = pd.read_csv(filepath, sep='\t')
matrix_afps.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | TFBS_chr | TFBS_start | TFBS_end | TFBS_strand | TFBS_score | 98JKPD8_LumA_score | ANAB5F7_Basal_score | S6R691V_Her2_score | PU24GB8_LumB_score | 2GAMBDQ_Normal-like_score |
|-----|----------|------------|----------|-------------|------------|--------------------|---------------------|--------------------|--------------------|---------------------------|
| 0   | chr1     | 10628      | 10638    | \+          | 7.40189    | 0.00000            | 0.00000             | 0.00000            | 0.00000            | 0.00000                   |
| 1   | chr1     | 181224     | 181234   | \+          | 7.99866    | 0.00000            | 0.00000             | 0.00000            | 0.00000            | 0.00000                   |
| 2   | chr1     | 779214     | 779224   | \-          | 7.79647    | 0.00000            | 0.00000             | 0.00000            | 0.00000            | 0.00000                   |
| 3   | chr1     | 998754     | 998764   | \+          | 8.56029    | 0.13760            | 0.14035             | 0.12842            | 0.14500            | 0.10510                   |
| 4   | chr1     | 998768     | 998778   | \+          | 8.56029    | 0.18224            | 0.16708             | 0.16911            | 0.18302            | 0.12759                   |

</div>

``` python
# extract motif id from filename
motif_id = os.path.basename(filepath).replace('_fpscore-af-varsites-combined-matrix-wide.tsv', '')
motif_id
```

    'E2F2_E2F2_HUMAN.H11MO.0.B_BRCA-subtype-vcf-filtered-matrix.txt'
