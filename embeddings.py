"""
TF Screening Embedding Pipeline
================================
Extracts protein sequences from gene symbols via UniProt/Entrez APIs,
then generates embeddings using ESM2-150M from HuggingFace.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Lazy imports for heavy dependencies
torch = None
transformers = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProteinRecord:
    """Container for protein metadata and sequence."""
    gene_symbol: str
    uniprot_id: Optional[str] = None
    protein_name: Optional[str] = None
    organism: Optional[str] = None
    sequence: Optional[str] = None
    source: Optional[str] = None  # 'uniprot' or 'entrez'
    error: Optional[str] = None


class SequenceFetcher:
    """Fetches protein sequences from UniProt and NCBI Entrez."""
    
    UNIPROT_API = "https://rest.uniprot.org/uniprotkb"
    NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self, organism: str = "mouse", email: str = None):
        """
        Args:
            organism: Target organism ('mouse', 'human', or taxonomy ID)
            email: Email for NCBI API (recommended for heavy usage)
        """
        self.organism = organism
        self.email = email
        self.organism_map = {
            'mouse': ('10090', 'Mus musculus'),
            'human': ('9606', 'Homo sapiens'),
        }
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'TF-Embedding-Pipeline/1.0'})
    
    def _get_taxonomy_info(self) -> tuple[str, str]:
        """Get taxonomy ID and scientific name for organism."""
        if self.organism.lower() in self.organism_map:
            return self.organism_map[self.organism.lower()]
        return (self.organism, self.organism)
    
    def fetch_from_uniprot(self, gene_symbol: str) -> ProteinRecord:
        """
        Query UniProt for protein sequence by gene symbol.
        Prioritizes reviewed (Swiss-Prot) entries.
        """
        tax_id, org_name = self._get_taxonomy_info()
        record = ProteinRecord(gene_symbol=gene_symbol)
        
        # Search query: gene symbol + organism, prefer reviewed entries
        query = f"(gene:{gene_symbol}) AND (organism_id:{tax_id})"
        params = {
            'query': query,
            'format': 'json',
            'fields': 'accession,id,gene_names,protein_name,organism_name,sequence',
            'size': 10,
        }
        
        try:
            resp = self.session.get(f"{self.UNIPROT_API}/search", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if not data.get('results'):
                record.error = f"No UniProt entries found for {gene_symbol} in {org_name}"
                return record
            
            # Prioritize reviewed (Swiss-Prot) entries
            results = data['results']
            reviewed = [r for r in results if r.get('entryType') == 'UniProtKB reviewed (Swiss-Prot)']
            entry = reviewed[0] if reviewed else results[0]
            
            record.uniprot_id = entry.get('primaryAccession')
            record.protein_name = entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown')
            record.organism = entry.get('organism', {}).get('scientificName', org_name)
            record.sequence = entry.get('sequence', {}).get('value')
            record.source = 'uniprot'
            
            logger.info(f"Found {gene_symbol} -> {record.uniprot_id} ({len(record.sequence)} aa)")
            
        except requests.RequestException as e:
            record.error = f"UniProt API error: {str(e)}"
            logger.warning(record.error)
        
        return record
    
    def fetch_from_entrez(self, gene_symbol: str) -> ProteinRecord:
        """
        Fallback: Query NCBI Entrez for protein sequence.
        Uses gene database to find protein accessions.
        """
        tax_id, org_name = self._get_taxonomy_info()
        record = ProteinRecord(gene_symbol=gene_symbol)
        
        try:
            # Step 1: Search gene database
            search_params = {
                'db': 'gene',
                'term': f"{gene_symbol}[Gene Name] AND {tax_id}[Taxonomy ID]",
                'retmode': 'json',
                'retmax': 1,
            }
            if self.email:
                search_params['email'] = self.email
            
            resp = self.session.get(self.NCBI_ESEARCH, params=search_params, timeout=30)
            resp.raise_for_status()
            search_data = resp.json()
            
            gene_ids = search_data.get('esearchresult', {}).get('idlist', [])
            if not gene_ids:
                record.error = f"No Entrez gene found for {gene_symbol}"
                return record
            
            # Step 2: Get protein accession from gene record
            # Using protein database directly with gene symbol
            protein_search = {
                'db': 'protein',
                'term': f"{gene_symbol}[Gene Name] AND {tax_id}[Taxonomy ID] AND refseq[filter]",
                'retmode': 'json',
                'retmax': 1,
            }
            if self.email:
                protein_search['email'] = self.email
                
            resp = self.session.get(self.NCBI_ESEARCH, params=protein_search, timeout=30)
            resp.raise_for_status()
            protein_data = resp.json()
            
            protein_ids = protein_data.get('esearchresult', {}).get('idlist', [])
            if not protein_ids:
                record.error = f"No protein record found for {gene_symbol}"
                return record
            
            # Step 3: Fetch protein sequence in FASTA format
            time.sleep(0.34)  # NCBI rate limit: 3 requests/second
            fetch_params = {
                'db': 'protein',
                'id': protein_ids[0],
                'rettype': 'fasta',
                'retmode': 'text',
            }
            if self.email:
                fetch_params['email'] = self.email
                
            resp = self.session.get(self.NCBI_EFETCH, params=fetch_params, timeout=30)
            resp.raise_for_status()
            
            # Parse FASTA
            lines = resp.text.strip().split('\n')
            header = lines[0] if lines else ''
            sequence = ''.join(lines[1:]) if len(lines) > 1 else ''
            
            record.uniprot_id = protein_ids[0]  # Actually NCBI accession
            record.protein_name = header[1:80] if header else 'Unknown'
            record.organism = org_name
            record.sequence = sequence
            record.source = 'entrez'
            
            logger.info(f"Found {gene_symbol} via Entrez -> {record.uniprot_id} ({len(record.sequence)} aa)")
            
        except requests.RequestException as e:
            record.error = f"Entrez API error: {str(e)}"
            logger.warning(record.error)
        
        return record
    
    def fetch_sequence(self, gene_symbol: str) -> ProteinRecord:
        """
        Fetch protein sequence, trying UniProt first, then Entrez as fallback.
        """
        # Skip obvious non-genes
        if gene_symbol.lower() in ('control', 'untreated', 'mock', 'empty'):
            return ProteinRecord(gene_symbol=gene_symbol, error="Control/non-gene entry")
        
        # Try UniProt first
        record = self.fetch_from_uniprot(gene_symbol)
        if record.sequence:
            return record
        
        # Fallback to Entrez
        logger.info(f"Trying Entrez fallback for {gene_symbol}")
        time.sleep(0.5)  # Be nice to APIs
        return self.fetch_from_entrez(gene_symbol)
    
    def fetch_batch(self, gene_symbols: list[str], max_workers: int = 4) -> dict[str, ProteinRecord]:
        """
        Fetch sequences for multiple gene symbols with concurrent requests.
        """
        results = {}
        
        # Use ThreadPoolExecutor for I/O-bound tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_gene = {
                executor.submit(self.fetch_sequence, gene): gene 
                for gene in gene_symbols
            }
            
            for future in as_completed(future_to_gene):
                gene = future_to_gene[future]
                try:
                    results[gene] = future.result()
                except Exception as e:
                    results[gene] = ProteinRecord(gene_symbol=gene, error=str(e))
        
        return results


class ESM2Embedder:
    """Generate protein embeddings using ESM2-150M from HuggingFace."""
    
    # MODEL_NAME = "facebook/esm2_t30_150M_UR50D" # 640 embeddings
    # MODEL_NAME = "facebook/esm2_t33_650M_UR50D" # 1280 embeddings
    # MODEL_NAME = "facebook/esm2_t36_3B_UR50D" # 2560 embeddings
    MODEL_NAME = "facebook/esm2_t48_15B_UR50D" # 5120 embeddings
    
    def __init__(self, device: str = None, max_length: int = 1024):
        """
        Args:
            device: 'cuda', 'cpu', or None for auto-detect
            max_length: Maximum sequence length (ESM2 limit is 1024)
        """
        global torch
        if torch is None:
            import torch as _torch
            torch = _torch
            
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Lazy-load the ESM2 model and tokenizer."""
        if self.model is not None:
            return
        
        from transformers import AutoTokenizer, AutoModel
        
        logger.info(f"Loading ESM2-150M on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def embed_sequence(self, sequence: str, pooling: str = 'mean') -> np.ndarray:
        """
        Generate embedding for a single protein sequence.
        
        Args:
            sequence: Amino acid sequence (single-letter codes)
            pooling: 'mean' for mean pooling, 'cls' for CLS token, 'last' for last hidden state
            
        Returns:
            Embedding vector (shape: hidden_dim,)
        """
        self.load_model()
        
        # Truncate if necessary
        if len(sequence) > self.max_length:
            logger.warning(f"Truncating sequence from {len(sequence)} to {self.max_length}")
            sequence = sequence[:self.max_length]
        
        # Tokenize
        inputs = self.tokenizer(
            sequence, 
            return_tensors='pt', 
            truncation=True, 
            max_length=self.max_length,
            padding=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
        
        # Pooling strategy
        if pooling == 'cls':
            # First token (CLS equivalent)
            embedding = hidden_states[0, 0, :]
        elif pooling == 'mean':
            # Mean over sequence (excluding special tokens)
            # ESM2 uses: <cls> seq <eos>
            embedding = hidden_states[0, 1:-1, :].mean(dim=0)
        elif pooling == 'last':
            # Last token before EOS
            embedding = hidden_states[0, -2, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return embedding.cpu().numpy()
    
    def embed_batch(
        self, 
        sequences: dict[str, str], 
        pooling: str = 'mean',
        batch_size: int = 8
    ) -> dict[str, np.ndarray]:
        """
        Generate embeddings for multiple sequences.
        
        Args:
            sequences: Dict mapping identifiers to sequences
            pooling: Pooling strategy
            batch_size: Batch size for processing
            
        Returns:
            Dict mapping identifiers to embedding vectors
        """
        self.load_model()
        embeddings = {}
        
        items = list(sequences.items())
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            
            for gene_id, seq in batch:
                if not seq:
                    logger.warning(f"Skipping {gene_id}: no sequence")
                    continue
                    
                try:
                    embeddings[gene_id] = self.embed_sequence(seq, pooling=pooling)
                    logger.info(f"Embedded {gene_id}: {embeddings[gene_id].shape}")
                except Exception as e:
                    logger.error(f"Failed to embed {gene_id}: {e}")
        
        return embeddings


class TFScreeningPipeline:
    """End-to-end pipeline for TF screening data processing."""
    
    def __init__(self, organism: str = 'mouse', device: str = None):
        self.fetcher = SequenceFetcher(organism=organism)
        self.embedder = ESM2Embedder(device=device)
        self.records: dict[str, ProteinRecord] = {}
        self.embeddings: dict[str, np.ndarray] = {}
    
    def load_well_ids(self, source) -> list[str]:
        """
        Load Well IDs from various sources.
        
        Args:
            source: CSV file path, DataFrame, or list of gene symbols
        """
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source)
            return df['Well ID'].tolist()
        elif isinstance(source, pd.DataFrame):
            return source['Well ID'].tolist()
        elif isinstance(source, list):
            return source
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
    
    def fetch_sequences(self, gene_symbols: list[str]) -> pd.DataFrame:
        """Fetch all protein sequences and return summary DataFrame."""
        logger.info(f"Fetching sequences for {len(gene_symbols)} genes...")
        self.records = self.fetcher.fetch_batch(gene_symbols)
        
        # Build summary
        rows = []
        for gene, record in self.records.items():
            rows.append({
                'gene_symbol': record.gene_symbol,
                'uniprot_id': record.uniprot_id,
                'protein_name': record.protein_name,
                'organism': record.organism,
                'sequence_length': len(record.sequence) if record.sequence else 0,
                'source': record.source,
                'error': record.error,
            })
        
        return pd.DataFrame(rows)
    
    def generate_embeddings(self, pooling: str = 'mean') -> np.ndarray:
        """Generate ESM2 embeddings for all fetched sequences."""
        sequences = {
            gene: record.sequence 
            for gene, record in self.records.items() 
            if record.sequence
        }
        
        logger.info(f"Generating embeddings for {len(sequences)} sequences...")
        self.embeddings = self.embedder.embed_batch(sequences, pooling=pooling)
        
        # Stack into matrix
        genes = list(self.embeddings.keys())
        if genes:
            embedding_matrix = np.stack([self.embeddings[g] for g in genes])
            return genes, embedding_matrix
        return [], np.array([])
    
    def run(
        self, 
        source, 
        output_dir: str = '.', 
        pooling: str = 'mean',
        save_sequences: bool = True
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Run complete pipeline.
        
        Args:
            source: Well ID source (CSV path, DataFrame, or list)
            output_dir: Directory for output files
            pooling: Embedding pooling strategy
            save_sequences: Whether to save fetched sequences to FASTA
            
        Returns:
            Tuple of (summary DataFrame, embedding matrix)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and fetch
        gene_symbols = self.load_well_ids(source)
        summary_df = self.fetch_sequences(gene_symbols)
        
        # Save summary
        summary_path = output_dir / 'sequence_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved sequence summary to {summary_path}")
        
        # Save sequences as FASTA
        if save_sequences:
            fasta_path = output_dir / 'sequences.fasta'
            with open(fasta_path, 'w') as f:
                for gene, record in self.records.items():
                    if record.sequence:
                        f.write(f">{gene}|{record.uniprot_id}|{record.source}\n")
                        # Wrap at 80 chars
                        for i in range(0, len(record.sequence), 80):
                            f.write(record.sequence[i:i+80] + '\n')
            logger.info(f"Saved sequences to {fasta_path}")
        
        # Generate embeddings
        genes, embedding_matrix = self.generate_embeddings(pooling=pooling)
        
        if len(genes) > 0:
            # Save embeddings
            emb_path = output_dir / 'embeddings_5120.npz'
            np.savez(emb_path, genes=genes, embeddings=embedding_matrix)
            logger.info(f"Saved embeddings to {emb_path} (shape: {embedding_matrix.shape})")
            
            # Also save as DataFrame for easy inspection
            emb_df = pd.DataFrame(embedding_matrix, index=genes)
            emb_df.to_csv(output_dir / 'embeddings_5120.csv')
        
        return summary_df, embedding_matrix


# ============================================================================
# Example usage
# ============================================================================

if __name__ == '__main__':
    # Save sample data

    df = pd.read_csv("tf.csv")
    
    # Run pipeline
    pipeline = TFScreeningPipeline(organism='mouse')
    summary, embeddings = pipeline.run(
        source='tf.csv',
        output_dir='./tf_embeddings_output',
        pooling='mean'
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nSequence fetch summary:")
    print(summary[['gene_symbol', 'uniprot_id', 'sequence_length', 'source', 'error']])
    print(f"\nEmbedding matrix shape: {embeddings.shape}")
    print(f"Output files in: ./tf_embeddings_output/")