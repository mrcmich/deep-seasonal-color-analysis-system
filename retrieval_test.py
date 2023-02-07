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
    
    print(f"Testing clip model {clip_model} (pretrained on '{pretrained}') on the test set of DressCode Dataset...")
    image_text_accuracy, text_image_accuracy = training_and_testing_retrieval.test_retrieval_model(device, model, tokenizer, dataset, batch_size)
    print(f"Image-text accuracy = {(100 * image_text_accuracy):.2f}%")
    print(f"Text-image accuracy = {(100 * text_image_accuracy):.2f}%")


if __name__ == '__main__':
    args = utils.parse_retrieval_arguments()
    
    main_worker(args)
