import yaml
import sys, os
sys.path.append("./")
import argparse
import score_reward_models
import vlm_reward_models
from tqdm import tqdm
from utils.rm_utils import get_pred, get_label
import json
from datasets import load_dataset

def main(args):

    with open(args.config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        reward_models_config = config_data["reward_models"]

    # rm_type = get_parent_label(config_data, args.model)

    rm_type_dict = {}
    for parent_label, sub_models in config_data["reward_models"].items():
        for sub_model in sub_models:
            rm_type_dict[sub_model] = parent_label
    
    device = args.device
    dataset = load_dataset(args.dataset, streaming=True)
    dataset = dataset['validation_unique']

    if os.path.exists(args.local_dataset_buffer):
        root_buffer = args.local_dataset_buffer
        all_images = os.listdir(root_buffer)
        image_dict = {image_dir.split(".jpg")[0]: image_dir for image_dir in all_images}

    if rm_type_dict[args.model] == "score_models":
        model_config = reward_models_config[rm_type_dict[args.model]][args.model]
        reward_model = score_reward_models.Scorer(args.model, model_config["model_path"], model_config["processor_path"], device)
    elif rm_type_dict[args.model] == "opensource_vlm":
        model_config = reward_models_config[rm_type_dict[args.model]][args.model]
        reward_model = vlm_reward_models.Scorer(args.model, model_config["model_path"], model_config["processor_path"], device)
    else:
        raise ValueError(f"Model {args.model} not found in config file")

    data_list = []
    threshold = args.threshold


    for id, example in tqdm(enumerate(dataset)):

        new_item = {}

        image_0_path = os.path.join(root_buffer, image_dict[example["image_0_uid"]])
        image_1_path = os.path.join(root_buffer, image_dict[example["image_1_uid"]])
        caption = example["caption"]

        scores = reward_model.get_score([image_0_path, image_1_path], caption)

        print(f"scores: {scores}")

        label = get_label(example)
        pred = get_pred(scores[0], scores[1], threshold)

        new_item["id"] = id
        new_item["caption"] = caption
        new_item["ranking_id"] = example["ranking_id"]
        new_item["image_0_uid"] = example["image_0_uid"]
        new_item["image_1_uid"] = example["image_1_uid"]
        new_item["score_0"] = scores[0]
        new_item["score_1"] = scores[1]
        new_item["label"] = label
        new_item["pred"] = pred

        data_list.append(new_item)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="clipscore_v1", help="rm model to evaluate")
    parser.add_argument("--config_path", "-c", type=str, default="config/config.yaml", help="config path")
    parser.add_argument("--dataset", type=str, default="yuvalkirstain/pickapic_v1", help="dataset")
    parser.add_argument("--local_dataset_buffer", type=str, default="dataset/pickapic_v1/validation_unique", help="local directory to buffer dataset")
    parser.add_argument("--save_dir", type=str, default="result/test.json", help="save directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold")
    args = parser.parse_args()

    main(args)
