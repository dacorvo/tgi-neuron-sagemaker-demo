import argparse
import boto3
import os
from sagemaker.huggingface import HuggingFaceModel



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The HF Hub model id.")
    parser.add_argument("--batch_size", type=int, default=1, help="The model static batch size.")
    parser.add_argument("--sequence_length", type=int, default=4096, help="The model static sequence length.")
    parser.add_argument("--num_cores", type=int, default=2, help="The number of cores on which the model should be split.")
    parser.add_argument("--auto_cast_type", type=str, default="bf16", choices=["fp16", "bf16"], help="One of fp16, bf16.")
    args = parser.parse_args()
    iam = boto3.client("iam")
    role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

    # Hub Model configuration.
    hub = {
        "HF_MODEL_ID": f"{args.model}",
        "HF_NUM_CORES": f"{args.num_cores}",
        "HF_AUTO_CAST_TYPE": args.auto_cast_type,
        "MAX_BATCH_SIZE": f"{args.batch_size}",
        "MAX_INPUT_TOKENS": f"{args.sequence_length // 2}",
        "MAX_TOTAL_TOKENS": f"{args.sequence_length}",
        "HF_TOKEN": os.environ["HF_TOKEN"],
    }

    region = boto3.Session().region_name
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.2-optimum0.0.27-neuronx-py310-ubuntu22.04"

    # create Hugging Face Model Class
    huggingface_model = HuggingFaceModel(
        image_uri=image_uri,
        env=hub,
        role=role,
    )

    # deploy model to SageMaker Inference
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.inf2.xlarge",
        container_startup_health_check_timeout=1800,
        volume_size=512,
    )

    # send request
    output = predictor.predict(
        {
            "inputs": "When you have a hammer in your hand,",
            "parameters": {
                "do_sample": True,
                "max_new_tokens": 256,
                "temperature": 0.1,
                "top_k": 10,
            }
        }
    )
    print(output)



if __name__ == "__main__":
    main()
