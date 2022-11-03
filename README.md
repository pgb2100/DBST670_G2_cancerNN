# DBST670_G2_cancerNN
## Neural Network for Drug Efficiency Against Tumors
We will use data from http://cancerrgene.org
The drug with the lowest IC50 index is the drug that has the highest efficiency.The concentration of a drug or inhibitor needed to inhibit a biological process or response by 50%.IC50 is commonly used as a measure of drug potency in whole cell assays. IC50 assays are also used for screening in target-based drug discovery campaigns.
## python Libraries Needed
 - numpy
 - matplotlib
 - pandas
 - tensorflow
 - keras
 - scikit-learn
 - seaborn
 - matplotlib
 - joblib

## R Libraries Needed
 - caret

### Run the  Python script
ex) python3 NN_Drug.py dataset_gene_expression.csv

### Run the R scrip
ex) Rscript CancerNN.R ComboData.csv

### Example data set
#### gene_expression.csv
| gene_1 | gene_2 | gene_3 | ... | gene_7998 | gene_7999 | gene_8000 | Cancer_Type |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0 |0| 0 |	... | 9.932659 | 6.928584 | 2.088413 | KIRC |
| 0 |0| 0 |	... | 9.872796 | 5.039231 | 2.448002 | KIRC |
| ... | ... | ... |	... | ... | ... | ... | ... |
| 0 | 0.639232 | 0 | ... | 10.649382 | 5.282158 | 0.639232 | BRCA0 |
| 0 | 0 | 0 | ... | 10.397717 | 7.55 | 0.926379 | COAD |

#### dataset_file.csv
|  | Drug Name | Drug ID | Cell Line Name | Cosmic ID | TCGA Classification | Tissue | Tissue Sub-type | IC50 | AUC | Max Conc | RMSE | Z score |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0 | Camptothecin | 1003 | TE-5 | 735784 | ESCA | aero_digestive_tract | oesophagus | -2.555310782 | 0.834075918 | 0.1 | 0.087242117 | -0.161952499 |
| 1 | Camptothecin | 1003 | EC-GI-10 | 753555 | ESCA | aero_digestive_tract | oesophagus | -3.125664052 | 0.804941689 | 0.1 | 0.082367836 | -0.472096346 |
| 2 | Camptothecin | 1003 | HCE-4 | 753559 | ESCA | aero_digestive_tract | oesophagus | -3.536140073 | 0.77867008 | 0.1 | 0.087080221 | -0.695302943 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

#### dataset_gene_expression.csv
 - combind dataset_file.csv and gene_expression.csv
 
| Drug Name | IC50 | Cell Line Name | TCGA Classification | Tissue | Tissue Sub-type | AUC | gene_5 | gene_6 | ... | gene_7992 | gene_7998 | gene_7999 | gene_8000 | Cancer_Type |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0 | | Erlotinib | -3.130315 | NCI-H1648 | LUAD | lung | lung_NSCLC_adenocarcinoma | 0.349972 | 0 | ... | 7.841923 | 10.426789 | 6.26948 | 1.789228 | LUAD |
| 1 | | Erlotinib | 3.661843 | NCI-H1650 | LUAD | lung | lung_NSCLC_adenocarcinoma | 0.983298 | 0 | ... | 8.229261 | 10.309976 | 5.823204 | 2.139208 | LUAD |
| 2 | | Erlotinib | 2.322838 | NCI-H1838 | LUAD | lung | lung_NSCLC_adenocarcinoma | 0.931209 | 0 | ... | 6.562594 | 10.455985 | 5.784609 | 1.37401 | LUAD |
| … | | … | … | … | … | … | … | … | … | ... | … | … | … | … | … |
| 4 | | Erlotinib | 0.95477 | NCI-H1355 | LUAD | lung | lung_NSCLC_adenocarcinoma | 0.852814 | 0 | ... | 7.716785 | 10.308157 | 5.327881 | 1.973905 | LUAD |
| 5 | | Erlotinib | 1.796901 | LXF-289 | LUAD | lung | lung_NSCLC_adenocarcinoma | 0.947678 | 0 | ... | 7.964786 | 9.244109 | 7.1679 | 3.957571 | LUAD |
| 6 | | Erlotinib | 3.145254 | NCI-H23 | LUAD | lung | lung_NSCLC_adenocarcinoma | 0.987954 | 0 | ... | 7.223423 | 10.135991 | 7.894854 | 2.916247 | LUAD |
| 7 | | Erlotinib | 1.134468 | NCI-H322M | LUAD | lung | lung_NSCLC_adenocarcinoma | 0.857633 | 0 | ... | 7.88194 | 10.344584 | 7.232785 | 3.245176 | LUAD |
| 8 | | Erlotinib | -1.097318 | EKVX | LUAD | lung | lung_NSCLC_adenocarcinoma | 0.612032 | 0 | ... | 7.911452 | 10.428119 | 6.373664 | 4.275126 | LUAD |

#### ComboData.csv on Rscript
 - this dataset can find from dataset R_NN Files_20221029.zip(combine Hotspot_Mutations.csv and PANCANCER_IC_Wed.csv through CombineICandGF.py)
 
| Drug ID | Cosmic ID | IC50 | A1CF | ... | ZRANB3 | ZSCAN1 | ZSCAN18 | ZSCAN5B |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1003 | 906800 | 2 | 0 | ... | 0 | 0 | 0 | 0 |
| 1004 | 906800 | 2 | 0 | ... | 0 | 0 | 0 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 1005 | 906800 | 1 | 0 | ... | 0 | 0 | 0 | 0 |
| 1006 | 906800 | 2 | 0 | ... | 0 | 0 | 0 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
