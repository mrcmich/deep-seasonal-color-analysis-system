import torch
import open_clip
from models.dataset import DressCodeDataset
from utils import utils, model_names
from retrieval import training_and_testing_retrieval


def main_worker(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = args.clip_model
    pretrained = model_names.CLIP_MODELS_PRETRAINED[clip_model]
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(clip_model)
    # Dataset & Dataloader
    dataset = DressCodeDataset(dataroot_path=args.dataroot,
                               preprocess=preprocess,
                               phase=args.phase,
                               order=args.order)
    
    query = args.query
    images_path = "retrieval/retrieved_images/"
    k = 5
    
    print(f"Retrieving clothes of type '{query}' from DressCode Dataset...")
    training_and_testing_retrieval.retrieve_clothes(device=device, model=model, tokenizer=tokenizer, query=query, k=k,
                                                    dataset=dataset, batch_size=args.batch_size, images_path=images_path)
    print(f"The retrieved clothes have been saved in {images_path}")


if __name__ == '__main__':
    # Get argparser configuration
    args = utils.parse_retrieval_arguments()
    # Call main worker
    main_worker(args)
