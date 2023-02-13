# --- Needed to import modules from other packages
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---

from .abstract_filter import AbstractFilter
from utils import model_names
from models import dataset
from palette_classification import palette
from palette_classification.palettes import mappings
from retrieval import training_and_testing_retrieval
import torch
import open_clip

class RetrievalFilter(AbstractFilter):
    """
    .. description:: 
    Filter for retrieving clothing images from a dataset that satisfy a query string, describing 
    the type of clothing in natural language, and belong to a certain palette (passed as input
    to method execute). The filter returns the paths of all compatible clothing images as a list.
    Moreover, the filter supports execution both on cpu and gpu. The filter doesn't support
    the printing of additional information through verbose parameter of method execute.
    """
    
    def __init__(self, cloth_dataset_path, palette_mappings_dict): 
        """
        .. inputs::
        cloth_dataset_path:     path of dataset of clothing items; the dataset is expected to have a sub-folder
                                for each category of clothing item, each containing a folder 'images' with
                                all images of clothing items of said category inside.
        palette_mappings_dict:  dictionary of n_categories couples (key, value) where each key is a category
                                of clothing item and each value is the mapping dictionary for said category
                                (a dictionary assigning a palette id to each image).
        """

        clip_model = 'ViT-B-32'
        pretrained = model_names.CLIP_MODELS_PRETRAINED[clip_model]
        self.model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained)
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        self.dataset = dataset.DressCodeDataset(
            dataroot_path=cloth_dataset_path,
            preprocess=preprocess,
            phase='test',
            order='unpaired')
        self.dataset_path = cloth_dataset_path
        self.palette_mappings_dict = palette_mappings_dict
        self.query = None
       
    def input_type(self):
        return palette.PaletteRGB

    def output_type(self):
        return list

    def set_query(self, query):
        self.query = query

    def get_query(self):
        return self.query

    def execute(self, input, device=None, verbose=False):
        assert(self.query is not None)

        filtered_cloth_paths = []
        input_palette_id = mappings.DESC_ID_MAPPING[input.description()]
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        cloth_paths = training_and_testing_retrieval.retrieve_clothes(
            device, self.model, self.tokenizer, self.query, self.dataset, k=-1, batch_size=32)
        
        for cloth_path in cloth_paths:
            cloth_path_tokens = cloth_path.split('/')
            category = cloth_path_tokens[-3]
            cloth_filename = cloth_path_tokens[-1]
            cloth_palette_id = self.palette_mappings_dict[category][cloth_filename]

            if input_palette_id == cloth_palette_id:
                filtered_cloth_paths.append(cloth_path)

        return filtered_cloth_paths