# DBST670_G2_cancerNN
## Neural Network for Drug Efficiency Against Tumors
We will use data from http://cancerrgene.org
The drug with the lowest IC50 index is the drug that has the highest efficiency.The concentration of a drug or inhibitor needed to inhibit a biological process or response by 50%.IC50 is commonly used as a measure of drug potency in whole cell assays. IC50 assays are also used for screening in target-based drug discovery campaigns.
## Libraries Needed
 - numpy
 - matplotlib
 - pandas
 - tensorflow
 - keras
 - scikit-learn
 - seaborn
 - matplotlib
### Run the script
ex) python3 NN_Drug.py dataset_file.csv gene_expression.csv

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
