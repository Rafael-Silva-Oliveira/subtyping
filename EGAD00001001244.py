# %% Imports
from rnanorm import CPM, TPM, TMM
import os.path
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import KaplanMeierFitter
import shutil
from cnmf import cNMF
from tqdm import tqdm
from scipy.stats import zscore
from pydeseq2.ds import DeseqStats
from pydeseq2.dds import DeseqDataSet
from matplotlib.colors import rgb2hex
import itertools
from scipy import stats
import scanpy as sc
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import decoupler as dc
import pandas as pd
import anndata as ad
import re
import numpy as np
from scipy.sparse import csr_matrix
from pybiomart import Server
from scipy.stats import gmean
import json
import joblib


def fpkm_to_tpm(self, fpkm):
    """
    Convert FPKM values to TPM.

    Parameters:
        fpkm (array-like): Array or list of FPKM values.

    Returns:
        np.ndarray: Array of TPM values.
    """
    fpkm = np.array(fpkm)
    return np.exp(np.log1p(fpkm) - np.log1p(np.sum(fpkm)) + np.log(1e6))


# %% Add gene lengths using pybiomart
# Connect to the Ensembl BioMart server
server = Server(host="http://www.ensembl.org")

# Access the specific mart
mart = server["ENSEMBL_MART_ENSEMBL"]

# Retrieve the desired dataset from the mart
dataset = mart["hsapiens_gene_ensembl"]

# Now you can perform queries on the dataset
gene_mapping = dataset.query(
    attributes=["ensembl_gene_id", "hgnc_symbol",
                "start_position", "end_position"]
)

# 3. Compute gene length
gene_mapping["gene_length"] = (
    gene_mapping["Gene end (bp)"] - gene_mapping["Gene start (bp)"] + 1
)

# Remove entries with missing HGNC symbols
gene_mapping = gene_mapping[gene_mapping["HGNC symbol"].notnull()]

# Remove entries with missing HGNC symbols
gene_mapping = gene_mapping[gene_mapping["HGNC symbol"].notnull()]

# Remove duplicate HGNC symbols, keeping the first occurrence
gene_mapping = gene_mapping.drop_duplicates(subset="HGNC symbol", keep="first")

# %%

try:
    _base_dir = Path(__file__).resolve().parent
except NameError:
    _base_dir = Path.cwd()

# The data folder is a subdirectory named 'EGAD00001001244' next to this script.
data_file = _base_dir / "EGAD00001001244" / "data_mrna_seq_fpkm.txt"

count_data = pd.read_csv(data_file, sep="\t")
# %%

count_data.drop(columns=["Entrez_Gene_Id"], inplace=True)

gene_symbols = count_data["Hugo_Symbol"].unique()
symbol_to_ensembl = dict(
    zip(gene_mapping["HGNC symbol"], gene_mapping["Gene stable ID"])
)

symbol_length = dict(
    zip(gene_mapping["HGNC symbol"], gene_mapping["gene_length"]))


# %%
# Convert FPKM to TPM
count_data.set_index("Hugo_Symbol", inplace=True)


# %%
clinical_metadata_path = _base_dir / \
    "EGAD00001001244" / "data_clinical_patient.txt"

clinical_metadata = pd.read_csv(clinical_metadata_path, sep="\t")

# %%
clinical_metadata = clinical_metadata.iloc[4:, :]

# Subset clinical metadata with the samples from count_data
clinical_metadata_subset = clinical_metadata[
    clinical_metadata["#Patient Identifier"].isin(count_data.columns)
].copy()

clinical_metadata_subset.rename(
    columns={
        "#Patient Identifier": "ID_Sample",
        "UICC Tumor Stage": "tnm_staging",
        "Sex": "sex",
        "Diagnosis Age": "age_at_sample",
        "Overall Survival (Months)": "time_since_t0_death",
        "Overall Survival Status": "status_os",
    },
    inplace=True,
)

clinical_metadata_subset["status_os"].replace(
    {"1:DECEASED": 1, "0:LIVING": 0}, inplace=True
)

# Add the staging (< III, LS, IV then ES)
clinical_metadata_subset["staging"] = clinical_metadata_subset["tnm_staging"].apply(
    lambda x: "ES" if x == "IV" else "LS" if x != "nan" else "LS"
)

try:
    clinical_metadata_subset.set_index("ID_Sample", inplace=True)
except Exception as e:
    print(f"Error setting index: {e}")

print(
    f"Number of samples per stage: {clinical_metadata_subset.staging.value_counts()}")

# %%


class NormalizationClass:
    """
    A class for performing normalization methods (CPM, TPM, TMM, log1p, z-score, FPKM-to-TPM) on AnnData objects.
    Stores results in adata.layers.
    """

    def fpkm_to_tpm(self, fpkm):
        """
        Convert FPKM values to TPM.

        Parameters:
            fpkm (array-like): Array or list of FPKM values.

        Returns:
            np.ndarray: Array of TPM values.
        """
        fpkm = np.array(fpkm)
        return np.exp(np.log1p(fpkm) - np.log1p(np.sum(fpkm)) + np.log(1e6))


# %%
normalizer = NormalizationClass()
tpm_data = count_data.apply(normalizer.fpkm_to_tpm, axis=0)
count_data_mean = tpm_data.groupby(level=0).agg(lambda x: gmean(x)).copy()

# %%
adata = ad.AnnData(count_data_mean.T)

adata.obs = clinical_metadata_subset.copy()

adata.var = pd.DataFrame(count_data_mean.index)

adata.var["ensembl_gene_id"] = adata.var["Hugo_Symbol"].map(symbol_to_ensembl)
adata.var["length"] = adata.var["Hugo_Symbol"].map(symbol_length)

adata.var["length"] = adata.var["length"].apply(
    lambda x: int(x) if pd.notnull(x) else x
)

adata.var["length"] = adata.var["length"].fillna(
    int(adata.var["length"].mean()))

adata.layers["tpm_counts"] = adata.X.copy()

adata.layers["log_tpm_counts"] = np.log1p(count_data_mean.T.values).copy()

adata.layers["z_log_tpm_counts"] = zscore(
    adata.layers["log_tpm_counts"], axis=0).copy()

adata.var.set_index("Hugo_Symbol", inplace=True)

# %% Subtyping
############### Subtyping ###########################
# %% Subtyping
# Read markers
new_markers = pd.read_excel(
    r"/mnt/work/RO_src/Projects/Common/data/1-s2.0-S1535610820306620-mmc2.xlsx",
    header=1,
)
markers_list = [gene.upper() for gene in new_markers["Unnamed: 0"].tolist()]
markers_list.append("YAP1")

unique_sclc_markers = get_unique_markers(sclc_stratifications, genes)

# %%
# Load genes 1300 discovery set
final_genes = pd.read_excel(
    r"/mnt/work/RO_src/Projects/Common/data/1-s2.0-S1535610820306620-mmc2.xlsx",
    header=1,
)
final_genes = final_genes["Unnamed: 0"].tolist()
final_genes.append("YAP1")
# final_genes.extend(["ZEB1", "ZEB2", "SNAI1", "TWIST1"])
missing_genes = [gene for gene in final_genes if gene not in adata.var_names]
print("Missing genes:", missing_genes)

final_genes_filtered = [
    gene for gene in final_genes if gene in adata.var_names]
final_adata = adata[:, final_genes_filtered].copy()
final_adata.X = final_adata.layers["tpm_counts"].copy()

# %%
try:
    final_adata.obs.set_index("ID_Sample", inplace=True)
except KeyError:
    print("ID_Sample column not found in final_adata.obs or already set as index")
subtyping = Subtyping(adata=adata, adata_nmf=final_adata)

tpm_counts = pd.DataFrame(
    final_adata.layers["tpm_counts"],
    final_adata.obs_names,
    final_adata.var_names,
)
tpm_counts.index.name = None

# Save as tab delimited file
tpm_counts.to_csv(
    "/mnt/work/RO_src/Projects/Common/data/ValidationDatasets/EGAD00001001244/tpm_counts.csv",
    sep="\t",
    index=True,
)

tpm_counts = pd.read_csv(
    "/mnt/work/RO_src/Projects/Common/data/ValidationDatasets/EGAD00001001244/tpm_counts.csv",
    sep="\t",
    index_col=0,
)

tpm_counts.index  # should be samples
tpm_counts.columns  # should be genes

# %%
subtyping.cnmf(
    k=4,
    layer="tpm_counts",
    n_iter=150,
    tpm_file="/mnt/work/RO_src/Projects/Common/data/ValidationDatasets/EGAD00001001244/tpm_counts.csv",
)
final_adata_nmf = subtyping.adata_nmf

# %%

deg_analysis_new = DEGAnalysis(
    final_adata_nmf,
    design_factor="nmf-group-de-novo",
    layer="tpm_counts",
    output_dir="./deg_analysis_results_de_novo_egad",
)
deg_analysis_new.create_dds()
deg_analysis_new.run_comparisons()
deg_analysis_new.save_results()

deg_analysis_new.create_boxplots(
    [
        "ASCL1",
        "NEUROD1",
        "POU2F3",
        "YAP1",
        "REST",
        "MYC",
        "B2M",
        "CHGA",
        "DLL3",
        "HLA-A",
        "HLA-C",
        "CD274",
    ],
    layer="log_tpm_counts",
    save_path="/mnt/work/RO_src/Projects/Paper_Subtyping/Subtyping/scripts/marker_boxplots_new_egad.png",
)

highlight_dict = {
    "NE": ["ASCL1", "NEUROD1", "INSM1", "DLL3", "CHGA"],
    "nonNE": ["REST", "MYC", "POU2F3", "YAP1"],
    "RR": ["JUN", "STAT1", "ABL1"],
    "RS": ["AR", "PRKCB", "RELA", "SUMO1", "PAK2", "IRF1", "HDAC1"],
    "APM": ["HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP", "B2M"],
    "Checkpt": ["PDCD1", "CD274", "LAG3", "CTLA4", "BTLA", "TIGIT"],
    "B/PC": ["CD79A", "MS4A1", "MZB1", "JCHAIN"],
    "T-eff": ["CD8A", "GZMA", "GZMB", "PRF1", "IFNG", "CXCL9", "CXCL10", "TBX21"],
    "EMT": ["ZEB1", "ZEB2", "SNAI1", "TWIST1"],
}

# Create volcano grid with highlighted genes
deg_analysis_new.create_volcano_grid(highlight_dict=highlight_dict)
results_nmf_new = deg_analysis_new.get_results()
for comparison, res in results_nmf_new.items():
    if "_vs_rest" in comparison:
        print(f"############## {comparison} ##############")
        print(res.results_df[res.results_df.index.isin(key_markers)])

# %%

subtyping.subtype_mapper(mapping_dictionary={1: "A", 2: "N", 3: "I", 4: "P"})

genes_to_plot, gene_categories = subtyping.reorder_samples()

final_adata_nmf.obs.rename(
    columns={
        1: "NMF_A_de_novo",
        2: "NMF_N_de_novo",
        3: "NMF_I_de_novo",
        4: "NMF_P_de_novo",
    },
    inplace=True,
)

adata_nmf_cp = final_adata_nmf.copy()
adata_nmf_cp.obs.rename(
    columns={
        "nmf-group-de-novo": "nmf_group_de_novo",
        "SCLC-Subtype-de-novo": "SCLC_Subtype_de_novo",
    },
    inplace=True,
)

highlight_dict = {
    "NE": ["ASCL1", "NEUROD1", "INSM1", "DLL3", "CHGA"],
    "nonNE": ["REST", "MYC", "POU2F3", "YAP1"],
    "RR": ["JUN", "STAT1", "ABL1"],
    "RS": ["AR", "PRKCB", "RELA", "SUMO1", "PAK2", "IRF1", "HDAC1"],
    "APM": ["HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP", "B2M"],
    "Checkpt": ["PDCD1", "CD274", "LAG3", "CTLA4", "BTLA", "TIGIT"],
    "B/PC": ["CD79A", "MS4A1", "MZB1", "JCHAIN"],
    "T-eff": ["CD8A", "GZMA", "GZMB", "PRF1", "IFNG", "CXCL9", "CXCL10", "TBX21"],
}


final_adata_nmf.obs["nmf-group-de-novo"].value_counts(normalize=True).round(2)

# %% Expression plot


# 1. Genes of interest
genes = ["ASCL1", "NEUROD1", "YAP1", "POU2F3"]

# 2. Extract expression data and add subtype
df = final_adata_nmf.to_df(layer="log_tpm_counts")[genes].copy()
df["Subtype"] = final_adata_nmf.obs["SCLC-Subtype-de-novo"]

# 3. Melt data to long format
melted = df.melt(id_vars="Subtype", var_name="Gene", value_name="Expression")

# 4. Optional: define subtype order and color palette
subtype_order = ["A", "N", "I", "P"]  # Adjust as needed
palette = {
    "A": "#E69F00",  # Amber
    "N": "#56B4E9",  # Sky Blue
    "I": "#009E73",  # Green
    "P": "#D55E00",  # Red
}

# 5. Plot boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=melted,
    x="Gene",
    y="Expression",
    hue="Subtype",
    order=genes,
    hue_order=subtype_order,
    palette=palette,
    showfliers=False,
    linewidth=1.5,
    width=0.6,
    dodge=True,
)

# 6. Overlay individual data points (stripplot)
sns.stripplot(
    data=melted,
    x="Gene",
    y="Expression",
    hue="Subtype",
    order=genes,
    hue_order=subtype_order,
    palette=palette,
    dodge=True,
    size=3,
    jitter=0.2,
    alpha=0.5,
    linewidth=0,
)

# 7. Cleanup plot
plt.title("Expression of SCLC Subtype-Defining Genes", fontsize=14)
plt.ylabel("log(TPM + 1)", fontsize=12)
plt.xlabel("Gene", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()

# 8. Fix duplicate legends due to stripplot
handles, labels = plt.gca().get_legend_handles_labels()
n_subtypes = len(subtype_order)
plt.legend(
    handles[:n_subtypes],
    labels[:n_subtypes],
    title="Subtype",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.show()

# %%
# Radial Plots ###########
adata_analysis_radial = final_adata_nmf.copy()
adata_analysis_radial.obs = adata_analysis_radial.obs[
    [
        "NMF_A_de_novo",
        "NMF_N_de_novo",
        "NMF_I_de_novo",
        "NMF_P_de_novo",
        "nmf-group-de-novo",
    ]
]

# Ensure we're working with raw counts
adata_analysis_radial.X = adata_analysis_radial.layers["log_tpm_counts"].copy()

# Normalize, log-transform, and scale the data
# sc.pp.normalize_total(adata_analysis_radial, target_sum=1e6)
# sc.pp.log1p(adata_analysis_radial)
sc.pp.scale(adata_analysis_radial)
# "EMT": ["ZEB1", "ZEB2", "SNAI1", "TWIST1"],
# Add gene categories
gene_categories = {
    "NE": ["CHGA", "DLL3", "NEUROD1", "INSM1", "ASCL1"],
    "nonNE": ["YAP1", "POU2F3", "MYC", "REST"],
    "T-eff": [
        "CD8A",
        "GZMA",
        "GZMB",
        "PRF1",
        "IFNG",
        "CXCL9",
        "CXCL10",
        "TBX21",
    ],
    "B/PC": ["CD79A", "MS4A1", "MZB1", "JCHAIN"],
    "APM": ["TAP1", "TAP2", "B2M", "HLA-A", "HLA-C"],
    "Checkpt": ["PDCD1", "CD274", "LAG3", "CTLA4", "BTLA", "TIGIT"],
}

# Compute scores for each category
for category, genes in gene_categories.items():
    sc.tl.score_genes(adata_analysis_radial, genes,
                      score_name=f"{category}_score")

# Calculate mean scores for each NMF group
nmf_scores = adata_analysis_radial.obs.groupby("nmf-group-de-novo").mean()

# Normalize scores (Z-score across subsets)
score_columns = [f"{cat}_score" for cat in gene_categories.keys()]
nmf_scores_norm = (
    nmf_scores[score_columns] - nmf_scores[score_columns].mean()
) / nmf_scores[score_columns].std()

# Define NMF labels
nmf_labels = {
    "1": "A",
    "2": "N",
    "3": "I",
    "4": "P",
}
print(nmf_scores_norm.round(2))
# Colors for each NMF group
colors = ["#8A7EB5", "#67C28E", "#E75480", "#FFA500"]
color_map = {
    "1": "#4169E1",  # A - RoyalBlue (cold)
    "2": "#008080",  # N - Teal (cold)
    "3": "#DC143C",  # I-nNE - Crimson (warm)
    "4": "#FF8C00",  # I-NE - DarkOrange (warm)
}
plot_radial_nmf(nmf_scores_norm, nmf_labels, color_map,
                gene_categories, score_columns)

# %%
# # %%
# prepare_for_saving(adata)
# # Save adata
# adata.write(
#     r"/mnt/work/RO_src/Projects/Common/data/ValidationDatasets/EGAD00001001244/EGAD00001001244.h5ad"
# )

# for l in ["log_tpm_counts", "z_log_tpm_counts"]:
#     counts = pd.DataFrame(adata.layers[l], adata.obs_names, adata.var["Hugo_Symbol"])
#     clinical = adata.obs[
#         ["tnm_staging", "sex", "age_at_sample", "status_os", "time_since_t0_death"]
#     ].copy()
#     merged_data = pd.concat(objs=[clinical, counts], axis=1)
#     merged_data.to_csv(
#         f"/mnt/work/RO_src/Projects/Common/data/ValidationDatasets/EGAD00001001244/bm_{l}.csv"
#     )
# # %%

# %%
# Survival

# Extract relevant data from AnnData object
df = final_adata_nmf.obs[
    ["SCLC-Subtype-de-novo", "status_os", "time_since_t0_death"]
].copy()

# Optional: drop rows with missing data
df = df.dropna(subset=["SCLC-Subtype-de-novo",
               "status_os", "time_since_t0_death"])

# Ensure proper types
df["status_os"] = df["status_os"].astype(int)  # 1 = death, 0 = censored
df["time_since_t0_death"] = df["time_since_t0_death"].astype(float)

kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 6))

for subtype in df["SCLC-Subtype-de-novo"].unique():
    sub_df = df[df["SCLC-Subtype-de-novo"] == subtype]
    kmf.fit(
        sub_df["time_since_t0_death"], event_observed=sub_df["status_os"], label=subtype
    )
    kmf.plot_survival_function(ci_show=False)

plt.title("Survival by SCLC Subtype")
plt.xlabel("Time since diagnosis (months)")
plt.ylabel("Survival probability")
plt.legend(title="Subtype")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

results = multivariate_logrank_test(
    df["time_since_t0_death"], df["SCLC-Subtype-de-novo"], df["status_os"]
)

results.print_summary()


# Create dataframe with all covariates
df_full = final_adata_nmf.obs[
    ["SCLC-Subtype-de-novo", "status_os",
        "time_since_t0_death", "age_at_sample", "sex"]
].dropna()
df_full["status_os"] = df_full["status_os"].astype(int)
df_full["time_since_t0_death"] = df_full["time_since_t0_death"].astype(float)

# Convert categorical variables to dummies
df_full = pd.get_dummies(
    df_full, columns=["SCLC-Subtype-de-novo", "sex"], drop_first=True
)

# Fit Cox model
cph = CoxPHFitter()
cph.fit(df_full, duration_col="time_since_t0_death", event_col="status_os")
cph.print_summary()
cph.plot()

# %%
# Read current file

temp_file = pd.read_csv(
    r"/mnt/work/RO_src/Projects/Paper_Biomarkers/Biomarkers/data/files/EGAD00001001244_validation/bm_log_tpm_counts.csv"
)

temp_file_cp = temp_file.copy()
temp_file_cp["staging"] = temp_file_cp["tnm_staging"].apply(
    lambda x: "ES" if x == "IV" else "LS" if x != "nan" else "nan"
)

altered_final_temp_file = temp_file_cp.copy()
col = altered_final_temp_file.pop("staging")
temp_file_new = temp_file.insert(
    temp_file.columns.get_loc("tnm_staging") + 1, col.name, col
).copy()

# %%
