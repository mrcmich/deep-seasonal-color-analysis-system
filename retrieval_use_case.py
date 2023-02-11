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

    dataset = DressCodeDataset(dataroot_path=args.dataroot,
                               preprocess=preprocess,
                               phase=args.phase,
                               order=args.order)
    
    batch_size = args.batch_size
    query = args.query
    k = args.k
    images_path = "retrieval/retrieved_images/"
    
    print(f"Retrieving clothes of type '{query}' from DressCode Dataset, using clip model {clip_model} (pretrained on {pretrained})...")
    training_and_testing_retrieval.retrieve_clothes(device=device, model=model, tokenizer=tokenizer, query=query, dataset=dataset,
                                                    k=k, batch_size=batch_size, save_img_path=images_path, verbose=True)
    print(f"The retrieved clothes have been saved in '{images_path}'")


if __name__ == '__main__':
    args = utils.parse_retrieval_arguments()

    main_worker(args)
