# Preprocessing TF and Mutation Data
Suffian Azizan

# Background

This repo documents a crucial step in my current pan-cancer study that
makes use of restricted ATAC-seq and DNAse-seq datasets generated from
healthy and cancerous human tissue samples available in the public
databases (i.e. TCGA and BLUEPRINT). I collated raw data from these
databases to generate TF footprinting data from the open chromatin
regions. The TF footprinting data is then combined with the noncoding
mutation data from the same samples obtained via variant calling to
identify TF binding sites (TFBS) that carry variant alleles (potentially
somatic mutations) that may modulate TF footprint scores (proxy for TF
binding activity).

# Data Preprocessing

TF footprints are determined and scored using the
[TOBIAS](https://github.molgen.mpg.de/pages/loosolab/www/software/TOBIAS/)
program published by Looso Lab (Max Plank Institute). A customized
TOBIAS pipeline was run on individual samples so TOBIAS’s internal
intersample normalization was not applied. The raw footprint scores were
collated for all samples in this pan-cancer study and combined into a
large dense matrix containing unique footprint sites of a particular
motif of interest across all samples. As we have 1360 motifs of interest
in this study, we have 1360 large data tables to process.

For the purpose of demonstration, a subset of TF footprint data of the
open chromatin regions in only breast cancer samples from TCGA database
was used here to save storage space and lower processing overhead.
Additionally, only 1 TF motif will be presented here in the analysis
workflow.

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

## Loading up the footprint data table

Now, we can import the data tables as Pandas dataframes. The data are
stored in tab-delimited text files (`.tsv`). The first column defines
the chromosomal location, the second and third column contain the
genomic coordinates of the TF footprint, the fourth column retains the
strandedness of the TF binding site (TFBS), the fifth column contains
the TFBS score (similarity score with the tested motif PWM), and the
rest of the columns carry the actual TOBIAS-calculated TF footprint
scores for individual samples.

The data is loaded into a Pandas dataframe and the first 5 rows are
displayed.

``` python
# import the data
filepath = '../demo-data/E2F2_E2F2_HUMAN.H11MO.0.B_BRCA-subtype-vcf-filtered-matrix.txt'
df_fpscore = pd.read_csv(filepath, sep='\t')
```

**Aside**: specify a formatter function to wrap long text in the data
tables.

``` python
#replace underscore with whitespace
func_underscore_replace = lambda x: x.replace("_", " ")
#wrap text
def func_wrap(x):
    if isinstance(x, str):
        return textwrap.fill(x, width=10)
    else:
        return x
```

<style type="text/css">
</style>
<table id="T_af364" data-quarto-disable-processing="true">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_af364_level0_col0" class="col_heading level0 col0" >TFBS_chr</th>
      <th id="T_af364_level0_col1" class="col_heading level0 col1" >TFBS_start</th>
      <th id="T_af364_level0_col2" class="col_heading level0 col2" >TFBS_end</th>
      <th id="T_af364_level0_col3" class="col_heading level0 col3" >TFBS_strand</th>
      <th id="T_af364_level0_col4" class="col_heading level0 col4" >TFBS_score</th>
      <th id="T_af364_level0_col5" class="col_heading level0 col5" >98JKPD8_LumA_score</th>
      <th id="T_af364_level0_col6" class="col_heading level0 col6" >ANAB5F7_Basal_score</th>
      <th id="T_af364_level0_col7" class="col_heading level0 col7" >S6R691V_Her2_score</th>
      <th id="T_af364_level0_col8" class="col_heading level0 col8" >PU24GB8_LumB_score</th>
      <th id="T_af364_level0_col9" class="col_heading level0 col9" >2GAMBDQ_Normal-like_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_af364_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_af364_row0_col0" class="data row0 col0" >chr1</td>
      <td id="T_af364_row0_col1" class="data row0 col1" >10628</td>
      <td id="T_af364_row0_col2" class="data row0 col2" >10638</td>
      <td id="T_af364_row0_col3" class="data row0 col3" >+</td>
      <td id="T_af364_row0_col4" class="data row0 col4" >7.401890</td>
      <td id="T_af364_row0_col5" class="data row0 col5" >0.000000</td>
      <td id="T_af364_row0_col6" class="data row0 col6" >0.000000</td>
      <td id="T_af364_row0_col7" class="data row0 col7" >0.000000</td>
      <td id="T_af364_row0_col8" class="data row0 col8" >0.000000</td>
      <td id="T_af364_row0_col9" class="data row0 col9" >0.000000</td>
    </tr>
    <tr>
      <th id="T_af364_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_af364_row1_col0" class="data row1 col0" >chr1</td>
      <td id="T_af364_row1_col1" class="data row1 col1" >181224</td>
      <td id="T_af364_row1_col2" class="data row1 col2" >181234</td>
      <td id="T_af364_row1_col3" class="data row1 col3" >+</td>
      <td id="T_af364_row1_col4" class="data row1 col4" >7.998660</td>
      <td id="T_af364_row1_col5" class="data row1 col5" >0.000000</td>
      <td id="T_af364_row1_col6" class="data row1 col6" >0.000000</td>
      <td id="T_af364_row1_col7" class="data row1 col7" >0.000000</td>
      <td id="T_af364_row1_col8" class="data row1 col8" >0.000000</td>
      <td id="T_af364_row1_col9" class="data row1 col9" >0.000000</td>
    </tr>
    <tr>
      <th id="T_af364_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_af364_row2_col0" class="data row2 col0" >chr1</td>
      <td id="T_af364_row2_col1" class="data row2 col1" >779214</td>
      <td id="T_af364_row2_col2" class="data row2 col2" >779224</td>
      <td id="T_af364_row2_col3" class="data row2 col3" >-</td>
      <td id="T_af364_row2_col4" class="data row2 col4" >7.796470</td>
      <td id="T_af364_row2_col5" class="data row2 col5" >0.000000</td>
      <td id="T_af364_row2_col6" class="data row2 col6" >0.000000</td>
      <td id="T_af364_row2_col7" class="data row2 col7" >0.000000</td>
      <td id="T_af364_row2_col8" class="data row2 col8" >0.000000</td>
      <td id="T_af364_row2_col9" class="data row2 col9" >0.000000</td>
    </tr>
    <tr>
      <th id="T_af364_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_af364_row3_col0" class="data row3 col0" >chr1</td>
      <td id="T_af364_row3_col1" class="data row3 col1" >998754</td>
      <td id="T_af364_row3_col2" class="data row3 col2" >998764</td>
      <td id="T_af364_row3_col3" class="data row3 col3" >+</td>
      <td id="T_af364_row3_col4" class="data row3 col4" >8.560290</td>
      <td id="T_af364_row3_col5" class="data row3 col5" >0.137600</td>
      <td id="T_af364_row3_col6" class="data row3 col6" >0.140350</td>
      <td id="T_af364_row3_col7" class="data row3 col7" >0.128420</td>
      <td id="T_af364_row3_col8" class="data row3 col8" >0.145000</td>
      <td id="T_af364_row3_col9" class="data row3 col9" >0.105100</td>
    </tr>
  </tbody>
</table>

Rename the `_score` column to `_fps` to avoid confusion with the
mutation score column in the mutation data table later, and drop the
`TFBS_strand` and `TFBS_score` columns as they are not needed for now.

``` python
# drop the column "TFBS_strand" and "TFBS_score"
df_fpscore = df_fpscore.drop(columns=["TFBS_strand", "TFBS_score"])
# rename columns in the dataframe
df_fpscore = df_fpscore.rename(columns={"TFBS_chr": "Chromosome", "TFBS_start": "Start", "TFBS_end": "End", "2GAMBDQ_Normal-like_score": "2GAMBDQ_Norm_fps"})
# for all column names that end with the string 'score', replace the string with 'fps'
df_fpscore = df_fpscore.rename(columns=lambda x: x.replace('score', 'fps') if x.endswith('score') else x)
```

<style type="text/css">
</style>
<table id="T_e3ecf" data-quarto-disable-processing="true">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e3ecf_level0_col0" class="col_heading level0 col0" >Chromosome</th>
      <th id="T_e3ecf_level0_col1" class="col_heading level0 col1" >Start</th>
      <th id="T_e3ecf_level0_col2" class="col_heading level0 col2" >End</th>
      <th id="T_e3ecf_level0_col3" class="col_heading level0 col3" >98JKPD8_LumA_fps</th>
      <th id="T_e3ecf_level0_col4" class="col_heading level0 col4" >ANAB5F7_Basal_fps</th>
      <th id="T_e3ecf_level0_col5" class="col_heading level0 col5" >S6R691V_Her2_fps</th>
      <th id="T_e3ecf_level0_col6" class="col_heading level0 col6" >PU24GB8_LumB_fps</th>
      <th id="T_e3ecf_level0_col7" class="col_heading level0 col7" >2GAMBDQ_Norm_fps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e3ecf_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_e3ecf_row0_col0" class="data row0 col0" >chr1</td>
      <td id="T_e3ecf_row0_col1" class="data row0 col1" >10628</td>
      <td id="T_e3ecf_row0_col2" class="data row0 col2" >10638</td>
      <td id="T_e3ecf_row0_col3" class="data row0 col3" >0.000000</td>
      <td id="T_e3ecf_row0_col4" class="data row0 col4" >0.000000</td>
      <td id="T_e3ecf_row0_col5" class="data row0 col5" >0.000000</td>
      <td id="T_e3ecf_row0_col6" class="data row0 col6" >0.000000</td>
      <td id="T_e3ecf_row0_col7" class="data row0 col7" >0.000000</td>
    </tr>
    <tr>
      <th id="T_e3ecf_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_e3ecf_row1_col0" class="data row1 col0" >chr1</td>
      <td id="T_e3ecf_row1_col1" class="data row1 col1" >181224</td>
      <td id="T_e3ecf_row1_col2" class="data row1 col2" >181234</td>
      <td id="T_e3ecf_row1_col3" class="data row1 col3" >0.000000</td>
      <td id="T_e3ecf_row1_col4" class="data row1 col4" >0.000000</td>
      <td id="T_e3ecf_row1_col5" class="data row1 col5" >0.000000</td>
      <td id="T_e3ecf_row1_col6" class="data row1 col6" >0.000000</td>
      <td id="T_e3ecf_row1_col7" class="data row1 col7" >0.000000</td>
    </tr>
    <tr>
      <th id="T_e3ecf_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_e3ecf_row2_col0" class="data row2 col0" >chr1</td>
      <td id="T_e3ecf_row2_col1" class="data row2 col1" >779214</td>
      <td id="T_e3ecf_row2_col2" class="data row2 col2" >779224</td>
      <td id="T_e3ecf_row2_col3" class="data row2 col3" >0.000000</td>
      <td id="T_e3ecf_row2_col4" class="data row2 col4" >0.000000</td>
      <td id="T_e3ecf_row2_col5" class="data row2 col5" >0.000000</td>
      <td id="T_e3ecf_row2_col6" class="data row2 col6" >0.000000</td>
      <td id="T_e3ecf_row2_col7" class="data row2 col7" >0.000000</td>
    </tr>
    <tr>
      <th id="T_e3ecf_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_e3ecf_row3_col0" class="data row3 col0" >chr1</td>
      <td id="T_e3ecf_row3_col1" class="data row3 col1" >998754</td>
      <td id="T_e3ecf_row3_col2" class="data row3 col2" >998764</td>
      <td id="T_e3ecf_row3_col3" class="data row3 col3" >0.137600</td>
      <td id="T_e3ecf_row3_col4" class="data row3 col4" >0.140350</td>
      <td id="T_e3ecf_row3_col5" class="data row3 col5" >0.128420</td>
      <td id="T_e3ecf_row3_col6" class="data row3 col6" >0.145000</td>
      <td id="T_e3ecf_row3_col7" class="data row3 col7" >0.105100</td>
    </tr>
    <tr>
      <th id="T_e3ecf_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_e3ecf_row4_col0" class="data row4 col0" >chr1</td>
      <td id="T_e3ecf_row4_col1" class="data row4 col1" >998768</td>
      <td id="T_e3ecf_row4_col2" class="data row4 col2" >998778</td>
      <td id="T_e3ecf_row4_col3" class="data row4 col3" >0.182240</td>
      <td id="T_e3ecf_row4_col4" class="data row4 col4" >0.167080</td>
      <td id="T_e3ecf_row4_col5" class="data row4 col5" >0.169110</td>
      <td id="T_e3ecf_row4_col6" class="data row4 col6" >0.183020</td>
      <td id="T_e3ecf_row4_col7" class="data row4 col7" >0.127590</td>
    </tr>
    <tr>
      <th id="T_e3ecf_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_e3ecf_row5_col0" class="data row5 col0" >chr1</td>
      <td id="T_e3ecf_row5_col1" class="data row5 col1" >1019693</td>
      <td id="T_e3ecf_row5_col2" class="data row5 col2" >1019703</td>
      <td id="T_e3ecf_row5_col3" class="data row5 col3" >0.622590</td>
      <td id="T_e3ecf_row5_col4" class="data row5 col4" >0.379450</td>
      <td id="T_e3ecf_row5_col5" class="data row5 col5" >0.489290</td>
      <td id="T_e3ecf_row5_col6" class="data row5 col6" >0.644200</td>
      <td id="T_e3ecf_row5_col7" class="data row5 col7" >0.492220</td>
    </tr>
    <tr>
      <th id="T_e3ecf_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_e3ecf_row6_col0" class="data row6 col0" >chr1</td>
      <td id="T_e3ecf_row6_col1" class="data row6 col1" >1041096</td>
      <td id="T_e3ecf_row6_col2" class="data row6 col2" >1041106</td>
      <td id="T_e3ecf_row6_col3" class="data row6 col3" >0.096050</td>
      <td id="T_e3ecf_row6_col4" class="data row6 col4" >0.097560</td>
      <td id="T_e3ecf_row6_col5" class="data row6 col5" >0.090640</td>
      <td id="T_e3ecf_row6_col6" class="data row6 col6" >0.127590</td>
      <td id="T_e3ecf_row6_col7" class="data row6 col7" >0.110160</td>
    </tr>
    <tr>
      <th id="T_e3ecf_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_e3ecf_row7_col0" class="data row7 col0" >chr1</td>
      <td id="T_e3ecf_row7_col1" class="data row7 col1" >1164827</td>
      <td id="T_e3ecf_row7_col2" class="data row7 col2" >1164837</td>
      <td id="T_e3ecf_row7_col3" class="data row7 col3" >0.110270</td>
      <td id="T_e3ecf_row7_col4" class="data row7 col4" >0.149950</td>
      <td id="T_e3ecf_row7_col5" class="data row7 col5" >0.073960</td>
      <td id="T_e3ecf_row7_col6" class="data row7 col6" >0.067850</td>
      <td id="T_e3ecf_row7_col7" class="data row7 col7" >0.080790</td>
    </tr>
    <tr>
      <th id="T_e3ecf_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_e3ecf_row8_col0" class="data row8 col0" >chr1</td>
      <td id="T_e3ecf_row8_col1" class="data row8 col1" >1206766</td>
      <td id="T_e3ecf_row8_col2" class="data row8 col2" >1206776</td>
      <td id="T_e3ecf_row8_col3" class="data row8 col3" >0.067360</td>
      <td id="T_e3ecf_row8_col4" class="data row8 col4" >0.063970</td>
      <td id="T_e3ecf_row8_col5" class="data row8 col5" >0.041980</td>
      <td id="T_e3ecf_row8_col6" class="data row8 col6" >0.069200</td>
      <td id="T_e3ecf_row8_col7" class="data row8 col7" >0.072820</td>
    </tr>
    <tr>
      <th id="T_e3ecf_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_e3ecf_row9_col0" class="data row9 col0" >chr1</td>
      <td id="T_e3ecf_row9_col1" class="data row9 col1" >1309989</td>
      <td id="T_e3ecf_row9_col2" class="data row9 col2" >1309999</td>
      <td id="T_e3ecf_row9_col3" class="data row9 col3" >0.123050</td>
      <td id="T_e3ecf_row9_col4" class="data row9 col4" >0.144050</td>
      <td id="T_e3ecf_row9_col5" class="data row9 col5" >0.141260</td>
      <td id="T_e3ecf_row9_col6" class="data row9 col6" >0.124540</td>
      <td id="T_e3ecf_row9_col7" class="data row9 col7" >0.096460</td>
    </tr>
  </tbody>
</table>

## Loading up the mutation data table

This mutation data is generated from the output of `bcftools` variant
calling pipeline. First, load up an example data file to see how the
data is structured.

``` python
vcfpath = '../demo-data/2GAMBDQ_E2F2_E2F2_HUMAN.H11MO.0.B_AF-per-site-with-indels.txt'
# load up the vcf file with indels and multiallelic sites split into separate rows
df_vcf = pd.read_csv(vcfpath, sep="\t")
```

<style type="text/css">
</style>
<table id="T_3e8fc" data-quarto-disable-processing="true">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_3e8fc_level0_col0" class="col_heading level0 col0" >#[1]CHROM</th>
      <th id="T_3e8fc_level0_col1" class="col_heading level0 col1" >[2]POS</th>
      <th id="T_3e8fc_level0_col2" class="col_heading level0 col2" >[3]REF</th>
      <th id="T_3e8fc_level0_col3" class="col_heading level0 col3" >[4]ALT</th>
      <th id="T_3e8fc_level0_col4" class="col_heading level0 col4" >[5]AF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3e8fc_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_3e8fc_row0_col0" class="data row0 col0" >chr1</td>
      <td id="T_3e8fc_row0_col1" class="data row0 col1" >10629</td>
      <td id="T_3e8fc_row0_col2" class="data row0 col2" >GGCGCGC</td>
      <td id="T_3e8fc_row0_col3" class="data row0 col3" >GGCGC</td>
      <td id="T_3e8fc_row0_col4" class="data row0 col4" >1.000000</td>
    </tr>
    <tr>
      <th id="T_3e8fc_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_3e8fc_row1_col0" class="data row1 col0" >chr1</td>
      <td id="T_3e8fc_row1_col1" class="data row1 col1" >998764</td>
      <td id="T_3e8fc_row1_col2" class="data row1 col2" >C</td>
      <td id="T_3e8fc_row1_col3" class="data row1 col3" >G</td>
      <td id="T_3e8fc_row1_col4" class="data row1 col4" >0.500000</td>
    </tr>
    <tr>
      <th id="T_3e8fc_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_3e8fc_row2_col0" class="data row2 col0" >chr1</td>
      <td id="T_3e8fc_row2_col1" class="data row2 col1" >998764</td>
      <td id="T_3e8fc_row2_col2" class="data row2 col2" >CGGAGGG</td>
      <td id="T_3e8fc_row2_col3" class="data row2 col3" >CGGAGGGGAGGG</td>
      <td id="T_3e8fc_row2_col4" class="data row2 col4" >0.312500</td>
    </tr>
    <tr>
      <th id="T_3e8fc_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_3e8fc_row3_col0" class="data row3 col0" >chr1</td>
      <td id="T_3e8fc_row3_col1" class="data row3 col1" >1041101</td>
      <td id="T_3e8fc_row3_col2" class="data row3 col2" >CGGAGCGGGGCGGGAGCGGGGCGGGAGCGGGG</td>
      <td id="T_3e8fc_row3_col3" class="data row3 col3" >CGGAGCGGGGCGGGAGCGGGG</td>
      <td id="T_3e8fc_row3_col4" class="data row3 col4" >0.500000</td>
    </tr>
    <tr>
      <th id="T_3e8fc_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_3e8fc_row4_col0" class="data row4 col0" >chr1</td>
      <td id="T_3e8fc_row4_col1" class="data row4 col1" >1164837</td>
      <td id="T_3e8fc_row4_col2" class="data row4 col2" >C</td>
      <td id="T_3e8fc_row4_col3" class="data row4 col3" >T</td>
      <td id="T_3e8fc_row4_col4" class="data row4 col4" >0.812500</td>
    </tr>
  </tbody>
</table>

The file above corresponds to just one of the sample IDs in this
pan-cancer study. To load up all the mutation data for all samples, we
need to load up all the files in the directory. Let’s put the dataframes
in one dictionary object using a loop.

``` python
# create a vcf load function for the query vcfs
def load_vcf(vcf_path):
    # load up the vcf file with indels and multiallelic sites split into separate rows
    df_vcf = pd.read_csv(vcf_path, sep="\t")
    # rename columns in the dataframe
    df_vcf = df_vcf.rename(columns={"#[1]CHROM": "Chromosome", "[2]POS": "Start", "[3]REF": "ref_allele", "[4]ALT": "alt_allele", "[5]AF": "AF"})
    # add a column next to the "start" column called "end" with the same value as the "start" column
    df_vcf.insert(2, "End", df_vcf["Start"])
    return df_vcf

# now put the paths in a list
paths = [
    "../demo-data/2GAMBDQ_E2F2_E2F2_HUMAN.H11MO.0.B_AF-per-site-with-indels.txt",
    "../demo-data/98JKPD8_E2F2_E2F2_HUMAN.H11MO.0.B_AF-per-site-with-indels.txt",
    "../demo-data/ANAB5F7_E2F2_E2F2_HUMAN.H11MO.0.B_AF-per-site-with-indels.txt",
    "../demo-data/PU24GB8_E2F2_E2F2_HUMAN.H11MO.0.B_AF-per-site-with-indels.txt",
    "../demo-data/S6R691V_E2F2_E2F2_HUMAN.H11MO.0.B_AF-per-site-with-indels.txt"
]

# create a list of IDs
ids = [ "ANAB5F7_basal", "98JKPD8_lumA", "PU24GB8_lumB", "S6R691V_her2", "2GAMBDQ_norm"]

# create a pair dictionary using nested dict comprehension
# for each id in the list of ids, iterate through the list of paths and check if the id is in the path; this means there is no need to order the list of ids according to the order of the paths

path_id_dict = {id: load_vcf(path) for id in ids for path in paths if id.split("_")[0] in path}
```

## Using PyRanges for dataframe merging

As we are dealing with genomic regions where the data is related to
interval values (start and end coordinates) spanning across two columns,
it is not possible to do dataframe overlap or join using Pandas. We will
need to use a specialized Python package called [PyRanges]() to handle
genomic coordinates.

Import the package and then first convert the footprint dataframe into a
PyRanges object.

``` python
import pyranges as pr

gr_fpscore = pr.PyRanges(df_fpscore)
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

|     | Chromosome | Start  | End    | ... | S6R691V_Her2_fps | PU24GB8_LumB_fps | 2GAMBDQ_Norm_fps |
|-----|------------|--------|--------|-----|------------------|------------------|------------------|
| 0   | chr1       | 10628  | 10638  | ... | 0.00000          | 0.00000          | 0.00000          |
| 1   | chr1       | 181224 | 181234 | ... | 0.00000          | 0.00000          | 0.00000          |
| 2   | chr1       | 779214 | 779224 | ... | 0.00000          | 0.00000          | 0.00000          |
| 3   | chr1       | 998754 | 998764 | ... | 0.12842          | 0.14500          | 0.10510          |
| 4   | chr1       | 998768 | 998778 | ... | 0.16911          | 0.18302          | 0.12759          |

<p>5 rows × 8 columns</p>
</div>

Do the same for the mutation dataframes in the dictionary. Loop through
it and save them in a new dictionary.

``` python
# load up vcf_dfs into pyranges 
grs = {}
for name,vcf in path_id_dict.items():
    gr_vcf = pr.PyRanges(vcf)
    grs[name] = gr_vcf

print(grs["ANAB5F7_basal"].head(n=10))
```

    +--------------+-----------+-----------+-------+
    | Chromosome   | Start     | End       | +3    |
    | (category)   | (int64)   | (int64)   | ...   |
    |--------------+-----------+-----------+-------|
    | chr1         | 10629     | 10629     | ...   |
    | chr1         | 181234    | 181234    | ...   |
    | chr1         | 779216    | 779216    | ...   |
    | chr1         | 998764    | 998764    | ...   |
    | ...          | ...       | ...       | ...   |
    | chr1         | 1019700   | 1019700   | ...   |
    | chr1         | 1041101   | 1041101   | ...   |
    | chr1         | 1164837   | 1164837   | ...   |
    | chr1         | 1206770   | 1206770   | ...   |
    +--------------+-----------+-----------+-------+
    Unstranded PyRanges object has 10 rows and 6 columns from 1 chromosomes.
    For printing, the PyRanges was sorted on Chromosome.
    3 hidden columns: ref_allele, alt_allele, AF

## Merging the PyRanges objects

Now, we can merge the PyRanges objects using the `join` function. The
`how` argument is set to `left` to retain all the rows in the left
dataframe (i.e. the footprint dataframe) and the `suffix` argument is
set to `_[sample ID]_varsite_pos` to add a suffix to the columns in the
right dataframe (i.e. the mutation dataframe) to avoid column name
clashes.

``` python
count = 0
for key, val in grs.items():
    
    if count == 0:
        overlap = gr_fpscore.join(val, how='left', suffix=f"_{key}_varsite_pos", preserve_order = True)
    else:
        overlap = filtered_gr.join(val, how='left', suffix=f"_{key}_varsite_pos", preserve_order = True)
    
    # drop the column "End" column
    overlap = overlap.drop([f"End_{key}_varsite_pos"])

    # cluster the pyRanges object by genomic range; overlapping regions wil share the same id. This will add a new column called "Cluster"
    overlap = overlap.cluster(slack=-1)

    # cast back into a dataframe and filter by the AF column's max value (by Cluster); this returns a filtered dataframe
    filtered_df = overlap.df.loc[overlap.df.groupby('Cluster')['AF'].idxmax()]
    
    # cast back into a dataframe and rename metadata columns
    filtered_df = filtered_df.rename(columns={f"Start_{key}_varsite_pos": f"{key}_varsite_pos", "ref_allele": f"{key}_REF", "alt_allele": f"{key}_ALT", "AF": f"{key}_AF"})
    
    # replace all the -1 values in column 'Start_varsites', 'ref_allele' and 'alt_allele', and AF with 0
    # Define a dictionary mapping column names to values to replace
    replace_dict = {f"{key}_varsite_pos": {-1: None}, f"{key}_REF": {str(-1): None}, f"{key}_ALT": {str(-1): None}, f"{key}_AF": {-1: 0}}
    filtered_df = filtered_df.replace(replace_dict)

    # drop cluster column
    filtered_df = filtered_df.drop(columns=["Cluster"])

    # cast back into pyrange object
    filtered_gr = pr.PyRanges(filtered_df)

    # increment count
    count += 1
```

Note that during the overlap process, the use of `cluster` function is
to ensure that overlapping regions will share the same ID. This is
important as we will need to filter the overlapping regions by the
maximum allele frequency (AF) value so that only 1) unique chromosome
regions are returned, and 2) regions with multiallelic sites, only the
site with the highest AF value is returned.

Now, we can clean up the PyRanges object and convert it back to a Pandas
dataframe. This file will be the basis for all downstream analyses.

``` python
final_df = filtered_gr.df

# create a column called 'region_id'
final_df["region_id"] = final_df["Chromosome"].astype(str) + ":" + final_df["Start"].astype(str) + "-" + final_df["End"].astype(str)

# for all column name ending with the string '_fps', split the string, take the second element, change the first letter in the string to lowercase, and reconstruct the original string with the new first letter
final_df = final_df.rename(columns=lambda x: x.split('_')[0] + '_' + x.split('_')[1][0].lower() + x.split('_')[1][1:] + '_fps' if x.endswith('_fps') else x)
```

A slice of the final dataframe is shown below.

<style type="text/css">
</style>
<table id="T_8c0c1" data-quarto-disable-processing="true">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_8c0c1_level0_col0" class="col_heading level0 col0" >Chromosome</th>
      <th id="T_8c0c1_level0_col1" class="col_heading level0 col1" >Start</th>
      <th id="T_8c0c1_level0_col2" class="col_heading level0 col2" >End</th>
      <th id="T_8c0c1_level0_col3" class="col_heading level0 col3" >98JKPD8_lumA_fps</th>
      <th id="T_8c0c1_level0_col4" class="col_heading level0 col4" >ANAB5F7_basal_fps</th>
      <th id="T_8c0c1_level0_col5" class="col_heading level0 col5" >S6R691V_her2_fps</th>
      <th id="T_8c0c1_level0_col6" class="col_heading level0 col6" >PU24GB8_lumB_fps</th>
      <th id="T_8c0c1_level0_col7" class="col_heading level0 col7" >2GAMBDQ_norm_fps</th>
      <th id="T_8c0c1_level0_col8" class="col_heading level0 col8" >ANAB5F7_basal_varsite_pos</th>
      <th id="T_8c0c1_level0_col9" class="col_heading level0 col9" >ANAB5F7_basal_REF</th>
      <th id="T_8c0c1_level0_col10" class="col_heading level0 col10" >ANAB5F7_basal_ALT</th>
      <th id="T_8c0c1_level0_col11" class="col_heading level0 col11" >ANAB5F7_basal_AF</th>
      <th id="T_8c0c1_level0_col12" class="col_heading level0 col12" >98JKPD8_lumA_varsite_pos</th>
      <th id="T_8c0c1_level0_col13" class="col_heading level0 col13" >98JKPD8_lumA_REF</th>
      <th id="T_8c0c1_level0_col14" class="col_heading level0 col14" >98JKPD8_lumA_ALT</th>
      <th id="T_8c0c1_level0_col15" class="col_heading level0 col15" >98JKPD8_lumA_AF</th>
      <th id="T_8c0c1_level0_col16" class="col_heading level0 col16" >PU24GB8_lumB_varsite_pos</th>
      <th id="T_8c0c1_level0_col17" class="col_heading level0 col17" >PU24GB8_lumB_REF</th>
      <th id="T_8c0c1_level0_col18" class="col_heading level0 col18" >PU24GB8_lumB_ALT</th>
      <th id="T_8c0c1_level0_col19" class="col_heading level0 col19" >PU24GB8_lumB_AF</th>
      <th id="T_8c0c1_level0_col20" class="col_heading level0 col20" >S6R691V_her2_varsite_pos</th>
      <th id="T_8c0c1_level0_col21" class="col_heading level0 col21" >S6R691V_her2_REF</th>
      <th id="T_8c0c1_level0_col22" class="col_heading level0 col22" >S6R691V_her2_ALT</th>
      <th id="T_8c0c1_level0_col23" class="col_heading level0 col23" >S6R691V_her2_AF</th>
      <th id="T_8c0c1_level0_col24" class="col_heading level0 col24" >2GAMBDQ_norm_varsite_pos</th>
      <th id="T_8c0c1_level0_col25" class="col_heading level0 col25" >2GAMBDQ_norm_REF</th>
      <th id="T_8c0c1_level0_col26" class="col_heading level0 col26" >2GAMBDQ_norm_ALT</th>
      <th id="T_8c0c1_level0_col27" class="col_heading level0 col27" >2GAMBDQ_norm_AF</th>
      <th id="T_8c0c1_level0_col28" class="col_heading level0 col28" >region_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_8c0c1_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_8c0c1_row0_col0" class="data row0 col0" >chr1</td>
      <td id="T_8c0c1_row0_col1" class="data row0 col1" >10628</td>
      <td id="T_8c0c1_row0_col2" class="data row0 col2" >10638</td>
      <td id="T_8c0c1_row0_col3" class="data row0 col3" >0.000000</td>
      <td id="T_8c0c1_row0_col4" class="data row0 col4" >0.000000</td>
      <td id="T_8c0c1_row0_col5" class="data row0 col5" >0.000000</td>
      <td id="T_8c0c1_row0_col6" class="data row0 col6" >0.000000</td>
      <td id="T_8c0c1_row0_col7" class="data row0 col7" >0.000000</td>
      <td id="T_8c0c1_row0_col8" class="data row0 col8" >10629</td>
      <td id="T_8c0c1_row0_col9" class="data row0 col9" >GGCGCGC</td>
      <td id="T_8c0c1_row0_col10" class="data row0 col10" >GGCGC</td>
      <td id="T_8c0c1_row0_col11" class="data row0 col11" >1.000000</td>
      <td id="T_8c0c1_row0_col12" class="data row0 col12" >10629</td>
      <td id="T_8c0c1_row0_col13" class="data row0 col13" >GGCGCGC</td>
      <td id="T_8c0c1_row0_col14" class="data row0 col14" >GGCGC</td>
      <td id="T_8c0c1_row0_col15" class="data row0 col15" >0.958333</td>
      <td id="T_8c0c1_row0_col16" class="data row0 col16" >10629</td>
      <td id="T_8c0c1_row0_col17" class="data row0 col17" >GGCGCGC</td>
      <td id="T_8c0c1_row0_col18" class="data row0 col18" >GGCGC</td>
      <td id="T_8c0c1_row0_col19" class="data row0 col19" >1.000000</td>
      <td id="T_8c0c1_row0_col20" class="data row0 col20" >10629</td>
      <td id="T_8c0c1_row0_col21" class="data row0 col21" >GGCGCGC</td>
      <td id="T_8c0c1_row0_col22" class="data row0 col22" >GGCGC</td>
      <td id="T_8c0c1_row0_col23" class="data row0 col23" >1.000000</td>
      <td id="T_8c0c1_row0_col24" class="data row0 col24" >10629</td>
      <td id="T_8c0c1_row0_col25" class="data row0 col25" >GGCGCGC</td>
      <td id="T_8c0c1_row0_col26" class="data row0 col26" >GGCGC</td>
      <td id="T_8c0c1_row0_col27" class="data row0 col27" >1.000000</td>
      <td id="T_8c0c1_row0_col28" class="data row0 col28" >chr1:10628-10638</td>
    </tr>
    <tr>
      <th id="T_8c0c1_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_8c0c1_row1_col0" class="data row1 col0" >chr1</td>
      <td id="T_8c0c1_row1_col1" class="data row1 col1" >181224</td>
      <td id="T_8c0c1_row1_col2" class="data row1 col2" >181234</td>
      <td id="T_8c0c1_row1_col3" class="data row1 col3" >0.000000</td>
      <td id="T_8c0c1_row1_col4" class="data row1 col4" >0.000000</td>
      <td id="T_8c0c1_row1_col5" class="data row1 col5" >0.000000</td>
      <td id="T_8c0c1_row1_col6" class="data row1 col6" >0.000000</td>
      <td id="T_8c0c1_row1_col7" class="data row1 col7" >0.000000</td>
      <td id="T_8c0c1_row1_col8" class="data row1 col8" >None</td>
      <td id="T_8c0c1_row1_col9" class="data row1 col9" >None</td>
      <td id="T_8c0c1_row1_col10" class="data row1 col10" >None</td>
      <td id="T_8c0c1_row1_col11" class="data row1 col11" >0.000000</td>
      <td id="T_8c0c1_row1_col12" class="data row1 col12" >None</td>
      <td id="T_8c0c1_row1_col13" class="data row1 col13" >None</td>
      <td id="T_8c0c1_row1_col14" class="data row1 col14" >None</td>
      <td id="T_8c0c1_row1_col15" class="data row1 col15" >0.000000</td>
      <td id="T_8c0c1_row1_col16" class="data row1 col16" >None</td>
      <td id="T_8c0c1_row1_col17" class="data row1 col17" >None</td>
      <td id="T_8c0c1_row1_col18" class="data row1 col18" >None</td>
      <td id="T_8c0c1_row1_col19" class="data row1 col19" >0.000000</td>
      <td id="T_8c0c1_row1_col20" class="data row1 col20" >None</td>
      <td id="T_8c0c1_row1_col21" class="data row1 col21" >None</td>
      <td id="T_8c0c1_row1_col22" class="data row1 col22" >None</td>
      <td id="T_8c0c1_row1_col23" class="data row1 col23" >0.000000</td>
      <td id="T_8c0c1_row1_col24" class="data row1 col24" >None</td>
      <td id="T_8c0c1_row1_col25" class="data row1 col25" >None</td>
      <td id="T_8c0c1_row1_col26" class="data row1 col26" >None</td>
      <td id="T_8c0c1_row1_col27" class="data row1 col27" >0.000000</td>
      <td id="T_8c0c1_row1_col28" class="data row1 col28" >chr1:181224-181234</td>
    </tr>
    <tr>
      <th id="T_8c0c1_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_8c0c1_row2_col0" class="data row2 col0" >chr1</td>
      <td id="T_8c0c1_row2_col1" class="data row2 col1" >779214</td>
      <td id="T_8c0c1_row2_col2" class="data row2 col2" >779224</td>
      <td id="T_8c0c1_row2_col3" class="data row2 col3" >0.000000</td>
      <td id="T_8c0c1_row2_col4" class="data row2 col4" >0.000000</td>
      <td id="T_8c0c1_row2_col5" class="data row2 col5" >0.000000</td>
      <td id="T_8c0c1_row2_col6" class="data row2 col6" >0.000000</td>
      <td id="T_8c0c1_row2_col7" class="data row2 col7" >0.000000</td>
      <td id="T_8c0c1_row2_col8" class="data row2 col8" >779216</td>
      <td id="T_8c0c1_row2_col9" class="data row2 col9" >T</td>
      <td id="T_8c0c1_row2_col10" class="data row2 col10" >C</td>
      <td id="T_8c0c1_row2_col11" class="data row2 col11" >0.029412</td>
      <td id="T_8c0c1_row2_col12" class="data row2 col12" >779216</td>
      <td id="T_8c0c1_row2_col13" class="data row2 col13" >T</td>
      <td id="T_8c0c1_row2_col14" class="data row2 col14" >C</td>
      <td id="T_8c0c1_row2_col15" class="data row2 col15" >0.125000</td>
      <td id="T_8c0c1_row2_col16" class="data row2 col16" >779216</td>
      <td id="T_8c0c1_row2_col17" class="data row2 col17" >T</td>
      <td id="T_8c0c1_row2_col18" class="data row2 col18" >C</td>
      <td id="T_8c0c1_row2_col19" class="data row2 col19" >0.026316</td>
      <td id="T_8c0c1_row2_col20" class="data row2 col20" >779216</td>
      <td id="T_8c0c1_row2_col21" class="data row2 col21" >T</td>
      <td id="T_8c0c1_row2_col22" class="data row2 col22" >C</td>
      <td id="T_8c0c1_row2_col23" class="data row2 col23" >0.058824</td>
      <td id="T_8c0c1_row2_col24" class="data row2 col24" >None</td>
      <td id="T_8c0c1_row2_col25" class="data row2 col25" >None</td>
      <td id="T_8c0c1_row2_col26" class="data row2 col26" >None</td>
      <td id="T_8c0c1_row2_col27" class="data row2 col27" >0.000000</td>
      <td id="T_8c0c1_row2_col28" class="data row2 col28" >chr1:779214-779224</td>
    </tr>
    <tr>
      <th id="T_8c0c1_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_8c0c1_row3_col0" class="data row3 col0" >chr1</td>
      <td id="T_8c0c1_row3_col1" class="data row3 col1" >998754</td>
      <td id="T_8c0c1_row3_col2" class="data row3 col2" >998764</td>
      <td id="T_8c0c1_row3_col3" class="data row3 col3" >0.137600</td>
      <td id="T_8c0c1_row3_col4" class="data row3 col4" >0.140350</td>
      <td id="T_8c0c1_row3_col5" class="data row3 col5" >0.128420</td>
      <td id="T_8c0c1_row3_col6" class="data row3 col6" >0.145000</td>
      <td id="T_8c0c1_row3_col7" class="data row3 col7" >0.105100</td>
      <td id="T_8c0c1_row3_col8" class="data row3 col8" >None</td>
      <td id="T_8c0c1_row3_col9" class="data row3 col9" >None</td>
      <td id="T_8c0c1_row3_col10" class="data row3 col10" >None</td>
      <td id="T_8c0c1_row3_col11" class="data row3 col11" >0.000000</td>
      <td id="T_8c0c1_row3_col12" class="data row3 col12" >None</td>
      <td id="T_8c0c1_row3_col13" class="data row3 col13" >None</td>
      <td id="T_8c0c1_row3_col14" class="data row3 col14" >None</td>
      <td id="T_8c0c1_row3_col15" class="data row3 col15" >0.000000</td>
      <td id="T_8c0c1_row3_col16" class="data row3 col16" >None</td>
      <td id="T_8c0c1_row3_col17" class="data row3 col17" >None</td>
      <td id="T_8c0c1_row3_col18" class="data row3 col18" >None</td>
      <td id="T_8c0c1_row3_col19" class="data row3 col19" >0.000000</td>
      <td id="T_8c0c1_row3_col20" class="data row3 col20" >None</td>
      <td id="T_8c0c1_row3_col21" class="data row3 col21" >None</td>
      <td id="T_8c0c1_row3_col22" class="data row3 col22" >None</td>
      <td id="T_8c0c1_row3_col23" class="data row3 col23" >0.000000</td>
      <td id="T_8c0c1_row3_col24" class="data row3 col24" >None</td>
      <td id="T_8c0c1_row3_col25" class="data row3 col25" >None</td>
      <td id="T_8c0c1_row3_col26" class="data row3 col26" >None</td>
      <td id="T_8c0c1_row3_col27" class="data row3 col27" >0.000000</td>
      <td id="T_8c0c1_row3_col28" class="data row3 col28" >chr1:998754-998764</td>
    </tr>
    <tr>
      <th id="T_8c0c1_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_8c0c1_row4_col0" class="data row4 col0" >chr1</td>
      <td id="T_8c0c1_row4_col1" class="data row4 col1" >998768</td>
      <td id="T_8c0c1_row4_col2" class="data row4 col2" >998778</td>
      <td id="T_8c0c1_row4_col3" class="data row4 col3" >0.182240</td>
      <td id="T_8c0c1_row4_col4" class="data row4 col4" >0.167080</td>
      <td id="T_8c0c1_row4_col5" class="data row4 col5" >0.169110</td>
      <td id="T_8c0c1_row4_col6" class="data row4 col6" >0.183020</td>
      <td id="T_8c0c1_row4_col7" class="data row4 col7" >0.127590</td>
      <td id="T_8c0c1_row4_col8" class="data row4 col8" >None</td>
      <td id="T_8c0c1_row4_col9" class="data row4 col9" >None</td>
      <td id="T_8c0c1_row4_col10" class="data row4 col10" >None</td>
      <td id="T_8c0c1_row4_col11" class="data row4 col11" >0.000000</td>
      <td id="T_8c0c1_row4_col12" class="data row4 col12" >None</td>
      <td id="T_8c0c1_row4_col13" class="data row4 col13" >None</td>
      <td id="T_8c0c1_row4_col14" class="data row4 col14" >None</td>
      <td id="T_8c0c1_row4_col15" class="data row4 col15" >0.000000</td>
      <td id="T_8c0c1_row4_col16" class="data row4 col16" >None</td>
      <td id="T_8c0c1_row4_col17" class="data row4 col17" >None</td>
      <td id="T_8c0c1_row4_col18" class="data row4 col18" >None</td>
      <td id="T_8c0c1_row4_col19" class="data row4 col19" >0.000000</td>
      <td id="T_8c0c1_row4_col20" class="data row4 col20" >None</td>
      <td id="T_8c0c1_row4_col21" class="data row4 col21" >None</td>
      <td id="T_8c0c1_row4_col22" class="data row4 col22" >None</td>
      <td id="T_8c0c1_row4_col23" class="data row4 col23" >0.000000</td>
      <td id="T_8c0c1_row4_col24" class="data row4 col24" >None</td>
      <td id="T_8c0c1_row4_col25" class="data row4 col25" >None</td>
      <td id="T_8c0c1_row4_col26" class="data row4 col26" >None</td>
      <td id="T_8c0c1_row4_col27" class="data row4 col27" >0.000000</td>
      <td id="T_8c0c1_row4_col28" class="data row4 col28" >chr1:998768-998778</td>
    </tr>
  </tbody>
</table>

The final dataframe here is saved as a tab-delimited text file in
`demo-data/` with the filename suffix
“`_fpscore-af-varsites-combined-matrix-wide.tsv`”.

# Merged Dataframe Data Analysis

Now that the data from the footprinting of TFs (**NOTE**: we are only
using E2F2 TF footprinting data here) and the mutation data overlapping
these footprints (obtained post-variant caling) are combined into a
single dataframe, we can start to do some analysis.

First load up the merged dataframe.

``` python
# import the data
filepath = '../demo-data/E2F2_E2F2_HUMAN.H11MO.0.B_fpscore-af-varsites-combined-matrix-wide.tsv'
afps_df = pd.read_csv(filepath, sep='\t')
# extract motif id from filename
motif_id = os.path.basename(filepath).replace('_fpscore-af-varsites-combined-matrix-wide.tsv', '')
print(f"The motif ID of the current TF data: {motif_id} \n")
```

    The motif ID of the current TF data: E2F2_E2F2_HUMAN.H11MO.0.B 

<style type="text/css">
</style>
<table id="T_9be5d" data-quarto-disable-processing="true">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9be5d_level0_col0" class="col_heading level0 col0" >Chromosome</th>
      <th id="T_9be5d_level0_col1" class="col_heading level0 col1" >Start</th>
      <th id="T_9be5d_level0_col2" class="col_heading level0 col2" >End</th>
      <th id="T_9be5d_level0_col3" class="col_heading level0 col3" >98JKPD8_lumA_fps</th>
      <th id="T_9be5d_level0_col4" class="col_heading level0 col4" >ANAB5F7_basal_fps</th>
      <th id="T_9be5d_level0_col5" class="col_heading level0 col5" >S6R691V_her2_fps</th>
      <th id="T_9be5d_level0_col6" class="col_heading level0 col6" >PU24GB8_lumB_fps</th>
      <th id="T_9be5d_level0_col7" class="col_heading level0 col7" >2GAMBDQ_norm_fps</th>
      <th id="T_9be5d_level0_col8" class="col_heading level0 col8" >2GAMBDQ_norm_varsite_pos</th>
      <th id="T_9be5d_level0_col9" class="col_heading level0 col9" >2GAMBDQ_norm_ref_allele</th>
      <th id="T_9be5d_level0_col10" class="col_heading level0 col10" >2GAMBDQ_norm_alt_allele</th>
      <th id="T_9be5d_level0_col11" class="col_heading level0 col11" >2GAMBDQ_norm_AF</th>
      <th id="T_9be5d_level0_col12" class="col_heading level0 col12" >98JKPD8_lumA_varsite_pos</th>
      <th id="T_9be5d_level0_col13" class="col_heading level0 col13" >98JKPD8_lumA_ref_allele</th>
      <th id="T_9be5d_level0_col14" class="col_heading level0 col14" >98JKPD8_lumA_alt_allele</th>
      <th id="T_9be5d_level0_col15" class="col_heading level0 col15" >98JKPD8_lumA_AF</th>
      <th id="T_9be5d_level0_col16" class="col_heading level0 col16" >ANAB5F7_basal_varsite_pos</th>
      <th id="T_9be5d_level0_col17" class="col_heading level0 col17" >ANAB5F7_basal_ref_allele</th>
      <th id="T_9be5d_level0_col18" class="col_heading level0 col18" >ANAB5F7_basal_alt_allele</th>
      <th id="T_9be5d_level0_col19" class="col_heading level0 col19" >ANAB5F7_basal_AF</th>
      <th id="T_9be5d_level0_col20" class="col_heading level0 col20" >PU24GB8_lumB_varsite_pos</th>
      <th id="T_9be5d_level0_col21" class="col_heading level0 col21" >PU24GB8_lumB_ref_allele</th>
      <th id="T_9be5d_level0_col22" class="col_heading level0 col22" >PU24GB8_lumB_alt_allele</th>
      <th id="T_9be5d_level0_col23" class="col_heading level0 col23" >PU24GB8_lumB_AF</th>
      <th id="T_9be5d_level0_col24" class="col_heading level0 col24" >S6R691V_her2_varsite_pos</th>
      <th id="T_9be5d_level0_col25" class="col_heading level0 col25" >S6R691V_her2_ref_allele</th>
      <th id="T_9be5d_level0_col26" class="col_heading level0 col26" >S6R691V_her2_alt_allele</th>
      <th id="T_9be5d_level0_col27" class="col_heading level0 col27" >S6R691V_her2_AF</th>
      <th id="T_9be5d_level0_col28" class="col_heading level0 col28" >region_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9be5d_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_9be5d_row0_col0" class="data row0 col0" >chr1</td>
      <td id="T_9be5d_row0_col1" class="data row0 col1" >10628</td>
      <td id="T_9be5d_row0_col2" class="data row0 col2" >10638</td>
      <td id="T_9be5d_row0_col3" class="data row0 col3" >0.000000</td>
      <td id="T_9be5d_row0_col4" class="data row0 col4" >0.000000</td>
      <td id="T_9be5d_row0_col5" class="data row0 col5" >0.000000</td>
      <td id="T_9be5d_row0_col6" class="data row0 col6" >0.000000</td>
      <td id="T_9be5d_row0_col7" class="data row0 col7" >0.000000</td>
      <td id="T_9be5d_row0_col8" class="data row0 col8" >10629.000000</td>
      <td id="T_9be5d_row0_col9" class="data row0 col9" >GGCGCGC</td>
      <td id="T_9be5d_row0_col10" class="data row0 col10" >GGCGC</td>
      <td id="T_9be5d_row0_col11" class="data row0 col11" >1.000000</td>
      <td id="T_9be5d_row0_col12" class="data row0 col12" >10629.000000</td>
      <td id="T_9be5d_row0_col13" class="data row0 col13" >GGCGCGC</td>
      <td id="T_9be5d_row0_col14" class="data row0 col14" >GGCGC</td>
      <td id="T_9be5d_row0_col15" class="data row0 col15" >0.958333</td>
      <td id="T_9be5d_row0_col16" class="data row0 col16" >10629.000000</td>
      <td id="T_9be5d_row0_col17" class="data row0 col17" >GGCGCGC</td>
      <td id="T_9be5d_row0_col18" class="data row0 col18" >GGCGC</td>
      <td id="T_9be5d_row0_col19" class="data row0 col19" >1.000000</td>
      <td id="T_9be5d_row0_col20" class="data row0 col20" >10629.000000</td>
      <td id="T_9be5d_row0_col21" class="data row0 col21" >GGCGCGC</td>
      <td id="T_9be5d_row0_col22" class="data row0 col22" >GGCGC</td>
      <td id="T_9be5d_row0_col23" class="data row0 col23" >1.000000</td>
      <td id="T_9be5d_row0_col24" class="data row0 col24" >10629.000000</td>
      <td id="T_9be5d_row0_col25" class="data row0 col25" >GGCGCGC</td>
      <td id="T_9be5d_row0_col26" class="data row0 col26" >GGCGC</td>
      <td id="T_9be5d_row0_col27" class="data row0 col27" >1.000000</td>
      <td id="T_9be5d_row0_col28" class="data row0 col28" >chr1:10628-10638</td>
    </tr>
    <tr>
      <th id="T_9be5d_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_9be5d_row1_col0" class="data row1 col0" >chr1</td>
      <td id="T_9be5d_row1_col1" class="data row1 col1" >181224</td>
      <td id="T_9be5d_row1_col2" class="data row1 col2" >181234</td>
      <td id="T_9be5d_row1_col3" class="data row1 col3" >0.000000</td>
      <td id="T_9be5d_row1_col4" class="data row1 col4" >0.000000</td>
      <td id="T_9be5d_row1_col5" class="data row1 col5" >0.000000</td>
      <td id="T_9be5d_row1_col6" class="data row1 col6" >0.000000</td>
      <td id="T_9be5d_row1_col7" class="data row1 col7" >0.000000</td>
      <td id="T_9be5d_row1_col8" class="data row1 col8" >nan</td>
      <td id="T_9be5d_row1_col9" class="data row1 col9" >nan</td>
      <td id="T_9be5d_row1_col10" class="data row1 col10" >nan</td>
      <td id="T_9be5d_row1_col11" class="data row1 col11" >0.000000</td>
      <td id="T_9be5d_row1_col12" class="data row1 col12" >nan</td>
      <td id="T_9be5d_row1_col13" class="data row1 col13" >nan</td>
      <td id="T_9be5d_row1_col14" class="data row1 col14" >nan</td>
      <td id="T_9be5d_row1_col15" class="data row1 col15" >0.000000</td>
      <td id="T_9be5d_row1_col16" class="data row1 col16" >nan</td>
      <td id="T_9be5d_row1_col17" class="data row1 col17" >nan</td>
      <td id="T_9be5d_row1_col18" class="data row1 col18" >nan</td>
      <td id="T_9be5d_row1_col19" class="data row1 col19" >0.000000</td>
      <td id="T_9be5d_row1_col20" class="data row1 col20" >nan</td>
      <td id="T_9be5d_row1_col21" class="data row1 col21" >nan</td>
      <td id="T_9be5d_row1_col22" class="data row1 col22" >nan</td>
      <td id="T_9be5d_row1_col23" class="data row1 col23" >0.000000</td>
      <td id="T_9be5d_row1_col24" class="data row1 col24" >nan</td>
      <td id="T_9be5d_row1_col25" class="data row1 col25" >nan</td>
      <td id="T_9be5d_row1_col26" class="data row1 col26" >nan</td>
      <td id="T_9be5d_row1_col27" class="data row1 col27" >0.000000</td>
      <td id="T_9be5d_row1_col28" class="data row1 col28" >chr1:181224-181234</td>
    </tr>
    <tr>
      <th id="T_9be5d_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_9be5d_row2_col0" class="data row2 col0" >chr1</td>
      <td id="T_9be5d_row2_col1" class="data row2 col1" >779214</td>
      <td id="T_9be5d_row2_col2" class="data row2 col2" >779224</td>
      <td id="T_9be5d_row2_col3" class="data row2 col3" >0.000000</td>
      <td id="T_9be5d_row2_col4" class="data row2 col4" >0.000000</td>
      <td id="T_9be5d_row2_col5" class="data row2 col5" >0.000000</td>
      <td id="T_9be5d_row2_col6" class="data row2 col6" >0.000000</td>
      <td id="T_9be5d_row2_col7" class="data row2 col7" >0.000000</td>
      <td id="T_9be5d_row2_col8" class="data row2 col8" >nan</td>
      <td id="T_9be5d_row2_col9" class="data row2 col9" >nan</td>
      <td id="T_9be5d_row2_col10" class="data row2 col10" >nan</td>
      <td id="T_9be5d_row2_col11" class="data row2 col11" >0.000000</td>
      <td id="T_9be5d_row2_col12" class="data row2 col12" >779216.000000</td>
      <td id="T_9be5d_row2_col13" class="data row2 col13" >T</td>
      <td id="T_9be5d_row2_col14" class="data row2 col14" >C</td>
      <td id="T_9be5d_row2_col15" class="data row2 col15" >0.125000</td>
      <td id="T_9be5d_row2_col16" class="data row2 col16" >779216.000000</td>
      <td id="T_9be5d_row2_col17" class="data row2 col17" >T</td>
      <td id="T_9be5d_row2_col18" class="data row2 col18" >C</td>
      <td id="T_9be5d_row2_col19" class="data row2 col19" >0.029412</td>
      <td id="T_9be5d_row2_col20" class="data row2 col20" >779216.000000</td>
      <td id="T_9be5d_row2_col21" class="data row2 col21" >T</td>
      <td id="T_9be5d_row2_col22" class="data row2 col22" >C</td>
      <td id="T_9be5d_row2_col23" class="data row2 col23" >0.026316</td>
      <td id="T_9be5d_row2_col24" class="data row2 col24" >779216.000000</td>
      <td id="T_9be5d_row2_col25" class="data row2 col25" >T</td>
      <td id="T_9be5d_row2_col26" class="data row2 col26" >C</td>
      <td id="T_9be5d_row2_col27" class="data row2 col27" >0.058824</td>
      <td id="T_9be5d_row2_col28" class="data row2 col28" >chr1:779214-779224</td>
    </tr>
    <tr>
      <th id="T_9be5d_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_9be5d_row3_col0" class="data row3 col0" >chr1</td>
      <td id="T_9be5d_row3_col1" class="data row3 col1" >998754</td>
      <td id="T_9be5d_row3_col2" class="data row3 col2" >998764</td>
      <td id="T_9be5d_row3_col3" class="data row3 col3" >0.137600</td>
      <td id="T_9be5d_row3_col4" class="data row3 col4" >0.140350</td>
      <td id="T_9be5d_row3_col5" class="data row3 col5" >0.128420</td>
      <td id="T_9be5d_row3_col6" class="data row3 col6" >0.145000</td>
      <td id="T_9be5d_row3_col7" class="data row3 col7" >0.105100</td>
      <td id="T_9be5d_row3_col8" class="data row3 col8" >nan</td>
      <td id="T_9be5d_row3_col9" class="data row3 col9" >nan</td>
      <td id="T_9be5d_row3_col10" class="data row3 col10" >nan</td>
      <td id="T_9be5d_row3_col11" class="data row3 col11" >0.000000</td>
      <td id="T_9be5d_row3_col12" class="data row3 col12" >nan</td>
      <td id="T_9be5d_row3_col13" class="data row3 col13" >nan</td>
      <td id="T_9be5d_row3_col14" class="data row3 col14" >nan</td>
      <td id="T_9be5d_row3_col15" class="data row3 col15" >0.000000</td>
      <td id="T_9be5d_row3_col16" class="data row3 col16" >nan</td>
      <td id="T_9be5d_row3_col17" class="data row3 col17" >nan</td>
      <td id="T_9be5d_row3_col18" class="data row3 col18" >nan</td>
      <td id="T_9be5d_row3_col19" class="data row3 col19" >0.000000</td>
      <td id="T_9be5d_row3_col20" class="data row3 col20" >nan</td>
      <td id="T_9be5d_row3_col21" class="data row3 col21" >nan</td>
      <td id="T_9be5d_row3_col22" class="data row3 col22" >nan</td>
      <td id="T_9be5d_row3_col23" class="data row3 col23" >0.000000</td>
      <td id="T_9be5d_row3_col24" class="data row3 col24" >nan</td>
      <td id="T_9be5d_row3_col25" class="data row3 col25" >nan</td>
      <td id="T_9be5d_row3_col26" class="data row3 col26" >nan</td>
      <td id="T_9be5d_row3_col27" class="data row3 col27" >0.000000</td>
      <td id="T_9be5d_row3_col28" class="data row3 col28" >chr1:998754-998764</td>
    </tr>
    <tr>
      <th id="T_9be5d_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_9be5d_row4_col0" class="data row4 col0" >chr1</td>
      <td id="T_9be5d_row4_col1" class="data row4 col1" >998768</td>
      <td id="T_9be5d_row4_col2" class="data row4 col2" >998778</td>
      <td id="T_9be5d_row4_col3" class="data row4 col3" >0.182240</td>
      <td id="T_9be5d_row4_col4" class="data row4 col4" >0.167080</td>
      <td id="T_9be5d_row4_col5" class="data row4 col5" >0.169110</td>
      <td id="T_9be5d_row4_col6" class="data row4 col6" >0.183020</td>
      <td id="T_9be5d_row4_col7" class="data row4 col7" >0.127590</td>
      <td id="T_9be5d_row4_col8" class="data row4 col8" >nan</td>
      <td id="T_9be5d_row4_col9" class="data row4 col9" >nan</td>
      <td id="T_9be5d_row4_col10" class="data row4 col10" >nan</td>
      <td id="T_9be5d_row4_col11" class="data row4 col11" >0.000000</td>
      <td id="T_9be5d_row4_col12" class="data row4 col12" >nan</td>
      <td id="T_9be5d_row4_col13" class="data row4 col13" >nan</td>
      <td id="T_9be5d_row4_col14" class="data row4 col14" >nan</td>
      <td id="T_9be5d_row4_col15" class="data row4 col15" >0.000000</td>
      <td id="T_9be5d_row4_col16" class="data row4 col16" >nan</td>
      <td id="T_9be5d_row4_col17" class="data row4 col17" >nan</td>
      <td id="T_9be5d_row4_col18" class="data row4 col18" >nan</td>
      <td id="T_9be5d_row4_col19" class="data row4 col19" >0.000000</td>
      <td id="T_9be5d_row4_col20" class="data row4 col20" >nan</td>
      <td id="T_9be5d_row4_col21" class="data row4 col21" >nan</td>
      <td id="T_9be5d_row4_col22" class="data row4 col22" >nan</td>
      <td id="T_9be5d_row4_col23" class="data row4 col23" >0.000000</td>
      <td id="T_9be5d_row4_col24" class="data row4 col24" >nan</td>
      <td id="T_9be5d_row4_col25" class="data row4 col25" >nan</td>
      <td id="T_9be5d_row4_col26" class="data row4 col26" >nan</td>
      <td id="T_9be5d_row4_col27" class="data row4 col27" >0.000000</td>
      <td id="T_9be5d_row4_col28" class="data row4 col28" >chr1:998768-998778</td>
    </tr>
  </tbody>
</table>

Filter the loaded table to include only the `_AF` and `_fps` columns, as
well as the `region_id` column to get a matrix of TF footprint scores
and allele frequencies of the variant sites overlapping the TF footprint
sites.

``` python
afps_matrix = afps_df.filter(regex='_AF$|_fps$|_id$').copy()
```

<style type="text/css">
</style>
<table id="T_f7e7f" data-quarto-disable-processing="true">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f7e7f_level0_col0" class="col_heading level0 col0" >98JKPD8_lumA_fps</th>
      <th id="T_f7e7f_level0_col1" class="col_heading level0 col1" >ANAB5F7_basal_fps</th>
      <th id="T_f7e7f_level0_col2" class="col_heading level0 col2" >S6R691V_her2_fps</th>
      <th id="T_f7e7f_level0_col3" class="col_heading level0 col3" >PU24GB8_lumB_fps</th>
      <th id="T_f7e7f_level0_col4" class="col_heading level0 col4" >2GAMBDQ_norm_fps</th>
      <th id="T_f7e7f_level0_col5" class="col_heading level0 col5" >2GAMBDQ_norm_AF</th>
      <th id="T_f7e7f_level0_col6" class="col_heading level0 col6" >98JKPD8_lumA_AF</th>
      <th id="T_f7e7f_level0_col7" class="col_heading level0 col7" >ANAB5F7_basal_AF</th>
      <th id="T_f7e7f_level0_col8" class="col_heading level0 col8" >PU24GB8_lumB_AF</th>
      <th id="T_f7e7f_level0_col9" class="col_heading level0 col9" >S6R691V_her2_AF</th>
      <th id="T_f7e7f_level0_col10" class="col_heading level0 col10" >region_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f7e7f_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_f7e7f_row0_col0" class="data row0 col0" >0.000000</td>
      <td id="T_f7e7f_row0_col1" class="data row0 col1" >0.000000</td>
      <td id="T_f7e7f_row0_col2" class="data row0 col2" >0.000000</td>
      <td id="T_f7e7f_row0_col3" class="data row0 col3" >0.000000</td>
      <td id="T_f7e7f_row0_col4" class="data row0 col4" >0.000000</td>
      <td id="T_f7e7f_row0_col5" class="data row0 col5" >1.000000</td>
      <td id="T_f7e7f_row0_col6" class="data row0 col6" >0.958333</td>
      <td id="T_f7e7f_row0_col7" class="data row0 col7" >1.000000</td>
      <td id="T_f7e7f_row0_col8" class="data row0 col8" >1.000000</td>
      <td id="T_f7e7f_row0_col9" class="data row0 col9" >1.000000</td>
      <td id="T_f7e7f_row0_col10" class="data row0 col10" >chr1:10628-10638</td>
    </tr>
    <tr>
      <th id="T_f7e7f_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_f7e7f_row1_col0" class="data row1 col0" >0.000000</td>
      <td id="T_f7e7f_row1_col1" class="data row1 col1" >0.000000</td>
      <td id="T_f7e7f_row1_col2" class="data row1 col2" >0.000000</td>
      <td id="T_f7e7f_row1_col3" class="data row1 col3" >0.000000</td>
      <td id="T_f7e7f_row1_col4" class="data row1 col4" >0.000000</td>
      <td id="T_f7e7f_row1_col5" class="data row1 col5" >0.000000</td>
      <td id="T_f7e7f_row1_col6" class="data row1 col6" >0.000000</td>
      <td id="T_f7e7f_row1_col7" class="data row1 col7" >0.000000</td>
      <td id="T_f7e7f_row1_col8" class="data row1 col8" >0.000000</td>
      <td id="T_f7e7f_row1_col9" class="data row1 col9" >0.000000</td>
      <td id="T_f7e7f_row1_col10" class="data row1 col10" >chr1:181224-181234</td>
    </tr>
    <tr>
      <th id="T_f7e7f_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_f7e7f_row2_col0" class="data row2 col0" >0.000000</td>
      <td id="T_f7e7f_row2_col1" class="data row2 col1" >0.000000</td>
      <td id="T_f7e7f_row2_col2" class="data row2 col2" >0.000000</td>
      <td id="T_f7e7f_row2_col3" class="data row2 col3" >0.000000</td>
      <td id="T_f7e7f_row2_col4" class="data row2 col4" >0.000000</td>
      <td id="T_f7e7f_row2_col5" class="data row2 col5" >0.000000</td>
      <td id="T_f7e7f_row2_col6" class="data row2 col6" >0.125000</td>
      <td id="T_f7e7f_row2_col7" class="data row2 col7" >0.029412</td>
      <td id="T_f7e7f_row2_col8" class="data row2 col8" >0.026316</td>
      <td id="T_f7e7f_row2_col9" class="data row2 col9" >0.058824</td>
      <td id="T_f7e7f_row2_col10" class="data row2 col10" >chr1:779214-779224</td>
    </tr>
    <tr>
      <th id="T_f7e7f_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_f7e7f_row3_col0" class="data row3 col0" >0.137600</td>
      <td id="T_f7e7f_row3_col1" class="data row3 col1" >0.140350</td>
      <td id="T_f7e7f_row3_col2" class="data row3 col2" >0.128420</td>
      <td id="T_f7e7f_row3_col3" class="data row3 col3" >0.145000</td>
      <td id="T_f7e7f_row3_col4" class="data row3 col4" >0.105100</td>
      <td id="T_f7e7f_row3_col5" class="data row3 col5" >0.000000</td>
      <td id="T_f7e7f_row3_col6" class="data row3 col6" >0.000000</td>
      <td id="T_f7e7f_row3_col7" class="data row3 col7" >0.000000</td>
      <td id="T_f7e7f_row3_col8" class="data row3 col8" >0.000000</td>
      <td id="T_f7e7f_row3_col9" class="data row3 col9" >0.000000</td>
      <td id="T_f7e7f_row3_col10" class="data row3 col10" >chr1:998754-998764</td>
    </tr>
    <tr>
      <th id="T_f7e7f_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_f7e7f_row4_col0" class="data row4 col0" >0.182240</td>
      <td id="T_f7e7f_row4_col1" class="data row4 col1" >0.167080</td>
      <td id="T_f7e7f_row4_col2" class="data row4 col2" >0.169110</td>
      <td id="T_f7e7f_row4_col3" class="data row4 col3" >0.183020</td>
      <td id="T_f7e7f_row4_col4" class="data row4 col4" >0.127590</td>
      <td id="T_f7e7f_row4_col5" class="data row4 col5" >0.000000</td>
      <td id="T_f7e7f_row4_col6" class="data row4 col6" >0.000000</td>
      <td id="T_f7e7f_row4_col7" class="data row4 col7" >0.000000</td>
      <td id="T_f7e7f_row4_col8" class="data row4 col8" >0.000000</td>
      <td id="T_f7e7f_row4_col9" class="data row4 col9" >0.000000</td>
      <td id="T_f7e7f_row4_col10" class="data row4 col10" >chr1:998768-998778</td>
    </tr>
  </tbody>
</table>

This matrix is in the wide format so it should be converted into a long
format for easier wrangling.

``` python
# convert to long format
afps_mtx_long = afps_matrix.melt(id_vars=["region_id"], var_name="variable", value_name="value")

# split the variable column into sample_id and type columns using reverse split string method, which returns a dataframe of columns based on the number of splits (n=x); this can directly be assigned to new columns in the original dataframe
afps_mtx_long[['sample_id', 'type']] = afps_mtx_long['variable'].str.rsplit('_', n=1, expand=True)

# drop the redundant 'variable' column
afps_mtx_long = afps_mtx_long.drop(columns=["variable"])

# now pivot the dataframe to create new columns based on the type column
afps_mtx_lpv = afps_mtx_long.pivot(index=['region_id', 'sample_id'], columns='type', values='value').reset_index()

# remove the index name and rename the columns to match the type values
afps_mtx_lpv = afps_mtx_lpv.rename_axis(None, axis=1).rename(columns={'fps': 'FPS'})

# sort the dataframe by region_id naturally
afps_mtx_lpv = afps_mtx_lpv.reindex(index=index_natsorted(afps_mtx_lpv['region_id']))
afps_mtx_lpv = afps_mtx_lpv.reset_index(drop=True)
```

<style type="text/css">
</style>
<table id="T_94cf7" data-quarto-disable-processing="true">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_94cf7_level0_col0" class="col_heading level0 col0" >region_id</th>
      <th id="T_94cf7_level0_col1" class="col_heading level0 col1" >sample_id</th>
      <th id="T_94cf7_level0_col2" class="col_heading level0 col2" >AF</th>
      <th id="T_94cf7_level0_col3" class="col_heading level0 col3" >FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_94cf7_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_94cf7_row0_col0" class="data row0 col0" >chr1:10628-10638</td>
      <td id="T_94cf7_row0_col1" class="data row0 col1" >2GAMBDQ_norm</td>
      <td id="T_94cf7_row0_col2" class="data row0 col2" >1.000000</td>
      <td id="T_94cf7_row0_col3" class="data row0 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_94cf7_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_94cf7_row1_col0" class="data row1 col0" >chr1:10628-10638</td>
      <td id="T_94cf7_row1_col1" class="data row1 col1" >98JKPD8_lumA</td>
      <td id="T_94cf7_row1_col2" class="data row1 col2" >0.958333</td>
      <td id="T_94cf7_row1_col3" class="data row1 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_94cf7_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_94cf7_row2_col0" class="data row2 col0" >chr1:10628-10638</td>
      <td id="T_94cf7_row2_col1" class="data row2 col1" >ANAB5F7_basal</td>
      <td id="T_94cf7_row2_col2" class="data row2 col2" >1.000000</td>
      <td id="T_94cf7_row2_col3" class="data row2 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_94cf7_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_94cf7_row3_col0" class="data row3 col0" >chr1:10628-10638</td>
      <td id="T_94cf7_row3_col1" class="data row3 col1" >PU24GB8_lumB</td>
      <td id="T_94cf7_row3_col2" class="data row3 col2" >1.000000</td>
      <td id="T_94cf7_row3_col3" class="data row3 col3" >0.000000</td>
    </tr>
    <tr>
      <th id="T_94cf7_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_94cf7_row4_col0" class="data row4 col0" >chr1:10628-10638</td>
      <td id="T_94cf7_row4_col1" class="data row4 col1" >S6R691V_her2</td>
      <td id="T_94cf7_row4_col2" class="data row4 col2" >1.000000</td>
      <td id="T_94cf7_row4_col3" class="data row4 col3" >0.000000</td>
    </tr>
  </tbody>
</table>

    Number of rows in the wide form: 2972
    Number of rows in the long form: 14860

## Scaling the TF footprint scores

As the TF footprint scores are not normalized across samples, has a
range from 0 to +Inf, and the fact that the mutation data come in the
form of allelic frequency (AF) probabilistic values (i.e. values between
0 and 1), the TF footprint scores should be scaled between 0 to 1.

``` python
# use MinMaxScaler to scale the raw fps values to range between 0 and 1
from sklearn.preprocessing import MinMaxScaler
# scale the FPS values to a range of 0-1
# Initialize a MinMaxScaler
scaler = MinMaxScaler()

# copy df
fps_df_scaled = afps_matrix.filter(regex='_fps$|_id$').copy()

# set the index to 'region_id'
fps_df_scaled = fps_df_scaled.set_index('region_id')

# Fit the MinMaxScaler to the 'FPS' column and transform it
fps_df_scaled = pd.DataFrame(scaler.fit_transform(fps_df_scaled), columns=fps_df_scaled.columns, index=fps_df_scaled.index)

# rename columns by adding '_scaled' to the column names
fps_df_scaled = fps_df_scaled.add_suffix('_scaled')
```

<style type="text/css">
</style>
<table id="T_89dd8" data-quarto-disable-processing="true">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_89dd8_level0_col0" class="col_heading level0 col0" >98JKPD8_lumA_fps_scaled</th>
      <th id="T_89dd8_level0_col1" class="col_heading level0 col1" >ANAB5F7_basal_fps_scaled</th>
      <th id="T_89dd8_level0_col2" class="col_heading level0 col2" >S6R691V_her2_fps_scaled</th>
      <th id="T_89dd8_level0_col3" class="col_heading level0 col3" >PU24GB8_lumB_fps_scaled</th>
      <th id="T_89dd8_level0_col4" class="col_heading level0 col4" >2GAMBDQ_norm_fps_scaled</th>
    </tr>
    <tr>
      <th class="index_name level0" >region_id</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_89dd8_level0_row0" class="row_heading level0 row0" >chr1:10628-10638</th>
      <td id="T_89dd8_row0_col0" class="data row0 col0" >0.000000</td>
      <td id="T_89dd8_row0_col1" class="data row0 col1" >0.000000</td>
      <td id="T_89dd8_row0_col2" class="data row0 col2" >0.000000</td>
      <td id="T_89dd8_row0_col3" class="data row0 col3" >0.000000</td>
      <td id="T_89dd8_row0_col4" class="data row0 col4" >0.000000</td>
    </tr>
    <tr>
      <th id="T_89dd8_level0_row1" class="row_heading level0 row1" >chr1:181224-181234</th>
      <td id="T_89dd8_row1_col0" class="data row1 col0" >0.000000</td>
      <td id="T_89dd8_row1_col1" class="data row1 col1" >0.000000</td>
      <td id="T_89dd8_row1_col2" class="data row1 col2" >0.000000</td>
      <td id="T_89dd8_row1_col3" class="data row1 col3" >0.000000</td>
      <td id="T_89dd8_row1_col4" class="data row1 col4" >0.000000</td>
    </tr>
    <tr>
      <th id="T_89dd8_level0_row2" class="row_heading level0 row2" >chr1:779214-779224</th>
      <td id="T_89dd8_row2_col0" class="data row2 col0" >0.000000</td>
      <td id="T_89dd8_row2_col1" class="data row2 col1" >0.000000</td>
      <td id="T_89dd8_row2_col2" class="data row2 col2" >0.000000</td>
      <td id="T_89dd8_row2_col3" class="data row2 col3" >0.000000</td>
      <td id="T_89dd8_row2_col4" class="data row2 col4" >0.000000</td>
    </tr>
    <tr>
      <th id="T_89dd8_level0_row3" class="row_heading level0 row3" >chr1:998754-998764</th>
      <td id="T_89dd8_row3_col0" class="data row3 col0" >0.034220</td>
      <td id="T_89dd8_row3_col1" class="data row3 col1" >0.034358</td>
      <td id="T_89dd8_row3_col2" class="data row3 col2" >0.040032</td>
      <td id="T_89dd8_row3_col3" class="data row3 col3" >0.041878</td>
      <td id="T_89dd8_row3_col4" class="data row3 col4" >0.027917</td>
    </tr>
    <tr>
      <th id="T_89dd8_level0_row4" class="row_heading level0 row4" >chr1:998768-998778</th>
      <td id="T_89dd8_row4_col0" class="data row4 col0" >0.045321</td>
      <td id="T_89dd8_row4_col1" class="data row4 col1" >0.040901</td>
      <td id="T_89dd8_row4_col2" class="data row4 col2" >0.052716</td>
      <td id="T_89dd8_row4_col3" class="data row4 col3" >0.052858</td>
      <td id="T_89dd8_row4_col4" class="data row4 col4" >0.033891</td>
    </tr>
  </tbody>
</table>

Now, convert the scaled dataframe into a long format.

``` python
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
```

<style type="text/css">
#T_125fc th {
  font-size: 10pt;
}
#T_125fc_row0_col0, #T_125fc_row0_col1, #T_125fc_row0_col2, #T_125fc_row1_col0, #T_125fc_row1_col1, #T_125fc_row1_col2, #T_125fc_row2_col0, #T_125fc_row2_col1, #T_125fc_row2_col2, #T_125fc_row3_col0, #T_125fc_row3_col1, #T_125fc_row3_col2, #T_125fc_row4_col0, #T_125fc_row4_col1, #T_125fc_row4_col2 {
  font-size: 10pt;
}
</style>

|     | region id        | sample id     | FPS scaled |
|-----|------------------|---------------|------------|
| 0   | chr1:10628-10638 | 2GAMBDQ_norm  | 0.000000   |
| 1   | chr1:10628-10638 | 98JKPD8_lumA  | 0.000000   |
| 2   | chr1:10628-10638 | ANAB5F7_basal | 0.000000   |
| 3   | chr1:10628-10638 | PU24GB8_lumB  | 0.000000   |
| 4   | chr1:10628-10638 | S6R691V_her2  | 0.000000   |

    Number of rows in the scaled matrix: 2972
    Number of rows in the scaled matrix in the long form: 14860

The distribution of the unscaled and scaled FPS datasets can be plotted
using Seaborn’s `displot` function.

![](data-integration-nb_files/figure-commonmark/cell-30-output-1.png)
