from data_navigator import DataNavigator

# File paths based on the prompt
EXPR_PATH = "/mnt/chromatin/home/cz2294/archive_proj/Allen_SingleCell_2_VIS_MO/expr_new/expr_mat/Tasic2018_MO_VIS_core.individual.expr.mat.txt"
SPLICING_PATH = "/mnt/chromatin/home/cz2294/archive_proj/Allen_SingleCell_2_VIS_MO/AS/cass_mat/MO_VIS_core.individual.cass.mat.txt"
MO_META_PATH = "/mnt/chromatin/home/cz2294/archive_proj/Allen_SingleCell_2_VIS_MO/fastq/meta_data/MO/sample_metadata.csv"
VIS_META_PATH = "/mnt/chromatin/home/cz2294/archive_proj/Allen_SingleCell_2_VIS_MO/fastq/meta_data/VIS/sample_metadata.csv"
CELL_ID_PATH = "/mnt/chromatin/home/cz2294/archive_proj/Allen_SingleCell_2_VIS_MO/conf/MO_VIS_core.txt"

def run_tests():
    print("Initializing DataNavigator...")
    nav = DataNavigator(
        expr_path=EXPR_PATH,
        cass_path=SPLICING_PATH,
        mo_meta_path=MO_META_PATH,
        vis_meta_path=VIS_META_PATH,
        cell_id_path=CELL_ID_PATH
    )

    print("\n--- Testing Metadata ---")
    meta = nav.get_metadata()
    print(f"Total samples in metadata: {len(meta)}")
    print(f"First 3 samples:\n{meta.head(3)[['full_sample_id', 'region', 'class', 'subclass', 'sample_name']]}")

    print("\n--- Testing Extracted Queries ---")
    # Query only L5 PT samples from MO
    matched_ids = nav.filter_samples(region='MO', subclass='L5 PT')
    print(f"Total L5 PT cells in MO: {len(matched_ids)}")
    if matched_ids:
        print(f"Sample matches (first 5): {matched_ids[:5]}")

        # Fetch expression for those samples
        expr_df = nav.get_expression(sample_ids=matched_ids)
        print(f"\nExpression matrix shape for these samples: {expr_df.shape}")
        
        # Fetch splicing for those samples
        splicing_df = nav.get_splicing(sample_ids=matched_ids)
        print(f"Splicing matrix shape for these samples: {splicing_df.shape}")
        
    print("\n--- Testing AnnData Creation ---")
    try:
        adata = nav.create_anndata(modality='expression')
        print(f"Successfully created AnnData from expression!")
        print(adata)
    except ImportError as e:
        print(f"Skipping AnnData test: {e}")

if __name__ == "__main__":
    run_tests()
