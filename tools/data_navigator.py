import pandas as pd
import numpy as np

class DataNavigator:
    """
    DataNavigator provides a flexible and easy-to-use framework for linking single-cell RNA-seq 
    metadata with gene expression and splicing matrices.
    """
    
    def __init__(self, expr_path=None, cass_path=None, mo_meta_path=None, vis_meta_path=None, cell_id_path=None):
        """
        Initialize the DataNavigator with paths to relevant datasets.
        
        Args:
            expr_path (str): Path to the gene expression matrix (.txt format).
            cass_path (str): Path to the targeted splicing matrix (.txt format).
            mo_meta_path (str): Path to the Motor cortex (MO) metadata CSV.
            vis_meta_path (str): Path to the Visual cortex (VIS) metadata CSV.
            cell_id_path (str): Path to the core cell identity mapping file.
        """
        self.expr_path = expr_path
        self.cass_path = cass_path
        self.mo_meta_path = mo_meta_path
        self.vis_meta_path = vis_meta_path
        self.cell_id_path = cell_id_path
        
        self.metadata = None
        self.expression = None
        self.splicing = None
        
        # Eagerly load metadata if all paths are provided
        if self.mo_meta_path and self.vis_meta_path and self.cell_id_path:
            self._load_metadata()
            
    def _load_metadata(self):
        """Internal method to load and unify metadata from VIS and MO cortices."""
        # Read MO metadata and append region
        mo_meta = pd.read_csv(self.mo_meta_path, encoding='latin1')
        mo_meta['region'] = 'MO'
        
        # Read VIS metadata and append region
        vis_meta = pd.read_csv(self.vis_meta_path, encoding='latin1')
        vis_meta['region'] = 'VIS'
        
        # Combine into a single metadata DataFrame
        self.metadata = pd.concat([mo_meta, vis_meta], ignore_index=True)
        
        # Create full sample ID using region '/' exp_component_name to match expression data
        self.metadata['full_sample_id'] = self.metadata['region'] + '/' + self.metadata['exp_component_name']

        # Integrate the additional cell identity mapping if a path was provided
        if self.cell_id_path:
            cell_id_df = pd.read_csv(
                self.cell_id_path, 
                sep='\t', 
                header=None, 
                names=['major_class', 'subclass_mapped', 'full_sample_id']
            )
            # Merge to bring in the mapped subsets
            self.metadata = self.metadata.merge(cell_id_df, on='full_sample_id', how='left')
            
        self.metadata.set_index('full_sample_id', inplace=True, drop=False)
        
    def _load_expression(self):
        """Internal method to load the gene expression matrix."""
        if not self.expr_path:
            raise ValueError("Expression path was not provided.")
            
        # Expression has columns like `MO/SM-D9CZQ_S96_E1-50.expr.txt`
        # index_col=0 indicates gene_id is the index
        self.expression = pd.read_csv(self.expr_path, sep='\t', index_col=0)
        
        # Rename columns by stripping the '.expr.txt' suffix to align with our full_sample_id
        new_cols = []
        for col in self.expression.columns:
            if col.endswith('.expr.txt'):
                new_cols.append(col[:-len('.expr.txt')])
            elif col.endswith('.txt'):
                new_cols.append(col[:-len('.txt')])
            else:
                new_cols.append(col)
        self.expression.columns = new_cols

    def _load_splicing(self):
        """Internal method to load the splicing events matrix."""
        if not self.cass_path:
            raise ValueError("Splicing matrices path was not provided.")
            
        # Splicing columns usually lack the prefix (e.g. `SM-D9CZQ_S96_E1-50`)
        self.splicing = pd.read_csv(self.cass_path, sep='\t', index_col=0)
        
        # We need to map `SM-...` to `MO/SM-...` using the metadata
        if self.metadata is not None:
            mapping = dict(zip(self.metadata['exp_component_name'], self.metadata['full_sample_id']))
            new_cols = []
            for col in self.splicing.columns:
                if col in mapping:
                    new_cols.append(mapping[col])
                else:
                    new_cols.append(col)
            self.splicing.columns = new_cols

    def get_metadata(self, sample_ids=None):
        """
        Retrieve metadata for a subset of samples or all samples.
        
        Args:
            sample_ids (list): List of sample_ids (`full_sample_id`). Defaults to all.
            
        Returns:
            pd.DataFrame: Sliced metadata DataFrame.
        """
        if self.metadata is None:
            self._load_metadata()
            
        if sample_ids is not None:
            missing = set(sample_ids) - set(self.metadata.index)
            if missing:
                print(f"Warning: {len(missing)} sample IDs not found in metadata.")
            valid_ids = [s for s in sample_ids if s in self.metadata.index]
            return self.metadata.loc[valid_ids]
            
        return self.metadata
        
    def get_expression(self, sample_ids=None, genes=None):
        """
        Retrieve the gene expression matrix, optionally filtered by samples and genes.
        
        Args:
            sample_ids (list): Columns to extract.
            genes (list): Rows (gene_ids) to extract.
            
        Returns:
            pd.DataFrame: The expression matrix.
        """
        if self.expression is None:
            self._load_expression()
        
        df = self.expression
        if sample_ids is not None:
            valid_ids = [s for s in sample_ids if s in df.columns]
            # Assure 'NAME' is kept if present
            subset_cols = ['NAME'] + valid_ids if 'NAME' in df.columns else valid_ids
            df = df[subset_cols]
            
        if genes is not None:
            df = df.loc[df.index.intersection(genes)]
            
        return df

    def get_splicing(self, sample_ids=None, events=None):
        """
        Retrieve the splicing matrix, optionally filtered by samples and events.
        
        Args:
            sample_ids (list): Columns to extract.
            events (list): Rows (event_ids) to extract.
            
        Returns:
            pd.DataFrame: The splicing matrix.
        """
        if self.splicing is None:
            self._load_splicing()
            
        df = self.splicing
        if sample_ids is not None:
            valid_ids = [s for s in sample_ids if s in df.columns]
            subset_cols = ['NAME'] + valid_ids if 'NAME' in df.columns else valid_ids
            df = df[subset_cols]
            
        if events is not None:
            df = df.loc[df.index.intersection(events)]
            
        return df

    def filter_samples(self, **kwargs):
        """
        Query samples by metadata fields dynamically.
        
        Example:
            navigator.filter_samples(region='MO', class='Glutamatergic')
            
        Args:
            **kwargs: Key-value pairs matching metadata attributes.
            
        Returns:
            list: List of `full_sample_id`s that match the criteria.
        """
        if self.metadata is None:
            self._load_metadata()
            
        result = self.metadata
        for col, val in kwargs.items():
            if col in result.columns:
                result = result[result[col] == val]
            else:
                raise ValueError(f"Column '{col}' not found in metadata. Available columns: {list(result.columns)}")
        return result['full_sample_id'].tolist()
        
    def create_anndata(self, modality='expression'):
        """
        Create an AnnData object seamlessly combining counts + annotations.
        Recommended for scRNA-seq downstream analysis in python (scanpy).
        
        Args:
            modality (str): 'expression' or 'splicing'.
            
        Returns:
            anndata.AnnData: Constructed AnnData object.
        """
        try:
            import anndata as ad
        except ImportError:
            raise ImportError("Please install 'anndata' to use this feature (`pip install anndata`).")
            
        if self.metadata is None:
            self._load_metadata()
            
        if modality == 'expression':
            if self.expression is None:
                self._load_expression()
            data_df = self.expression
        elif modality == 'splicing':
            if self.splicing is None:
                self._load_splicing()
            data_df = self.splicing
        else:
            raise ValueError("Modality must be 'expression' or 'splicing'.")
            
        # Isolate true sample columns (excluding metadata-like columns such as 'NAME')
        col_samples = [c for c in data_df.columns if c in self.metadata.index]
        missing_in_meta = [c for c in data_df.columns if c not in self.metadata.index and c != 'NAME']
        
        if missing_in_meta:
            print(f"Warning: {len(missing_in_meta)} columns in {modality} data have no matching metadata and will be dropped.")
            
        # Transpose to shape (samples x features) as required by AnnData
        X = data_df[col_samples].T.values 
        obs = self.metadata.loc[col_samples].copy()
        
        var = pd.DataFrame(index=data_df.index)
        if 'NAME' in data_df.columns:
            var['NAME'] = data_df['NAME']
            
        adata = ad.AnnData(X=X, obs=obs, var=var)
        return adata
