# Sagemaker Hugging Face TGI service deployment demo

## Prerequisites

You need an existing Sagemaker HF TGI image, possibly created from: https://github.com/awslabs/llm-hosting-container.

You also need to export your AWS CLI credentials, namely:

```
export AWS_ACCESS_KEY_ID=<ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<SECRET_ACCESS_KEY>
export AWS_SESSION_TOKEN=<SESSION_TOKEN>
```

And finally you need to set a default region:

```
export AWS_DEFAULT_REGION=<REGION>
```

If you plan to use a gated model, you also need to export your Hugging Face credentials:

```
export HF_TOKEN=<YOUR_TOKEN>
```

## Create a TGI neuronx endpoint

```
python create_endpoint.py --model meta-llama/Llama-3.2-1B-Instruct \
                          --batch_size 1 \
                          --sequence_length 4096 \
                          --num_cores 2 \
                          --auto_cast_type bf16
```

## Test endpoint using a CLI

After having exported your AWS credentials and HF_TOKEN in the environment:

```
python invoke_endpoint huggingface-pytorch-tgi-inference-<ID> \
                       --prompt "The most important thing in life is"
                       --top_k 50 \
                       --top_p 0.7 \
                       --temperature 0.9
```

## Test endpoint in a gradio app

After having exported your AWS credentials and HF_TOKEN in the environment:

```
export SAGEMAKER_ENDPOINT_NAME=huggingface-pytorch-tgi-inference-<ID>
python gradio/app.py
```

