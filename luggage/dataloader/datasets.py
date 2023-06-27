from pathlib import Path
import pickle
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from ..utils.graph import graphein_to_pytorch_graph, load_protein_as_graph

class ProteinDataset(Dataset):
    """
    torch-geometric Dataset class for loading protein files as graphs.
    """
    def __init__(self, root, 
                 pdb_dir, 
                 graphein_dir, 
                 graph_dir,
                 node_attr_columns: list, 
                 edge_attr_columns: list, 
                 edge_kinds: set,
                 protein_names: list, 
                 label_mapping: dict, 
                 pre_transform=None, 
                 transform=None, 
                 num_workers=1):
        self.pdb_dir = Path(pdb_dir)
        self.graphein_dir = Path(graphein_dir)
        self.graph_dir = Path(graph_dir)
        self.node_attr_columns = node_attr_columns
        self.edge_attr_columns = edge_attr_columns
        self.edge_kinds = edge_kinds
        self.protein_names = protein_names
        self.label_mapping = label_mapping
        self.num_workers = num_workers
        super(ProteinDataset, self).__init__(root, pre_transform=pre_transform, transform=transform)

    @property
    def processed_dir(self) -> str:
        return str(self.graph_dir)    

    @property
    def raw_dir(self) -> str:
        return str(self.graphein_dir)
        
    def download(self):
        for protein_name in tqdm(self.protein_names):
            output = Path(self.raw_dir) / f'{protein_name}.pkl'
            if not output.exists():
                graph = load_protein_as_graph(self.pdb_dir / f"{protein_name}.pdb")
                with open(output, "wb") as f:
                    pickle.dump(graph, f)

    @property
    def raw_file_names(self):
        return [Path(self.raw_dir) / f"{protein_name}.pkl" for protein_name in self.protein_names]
    
    @property
    def processed_file_names(self):
        return [Path(self.processed_dir) / f"{protein_name}.pt" for protein_name in self.protein_names]

    def process(self):
        for protein_name in self.protein_names:
            output = Path(self.processed_dir) / f'{protein_name}.pt'
            if not output.exists():
                with open(Path(self.raw_dir) / f"{protein_name}.pkl", "rb") as f:
                    data = pickle.load(f)
                data = graphein_to_pytorch_graph(data, self.node_attr_columns, self.edge_attr_columns, self.edge_kinds, self.label_mapping[protein_name])
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                torch.save(data, output)

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        return data
    
