#!/bin/python
import argparse
from sagemaker.huggingface import get_huggingface_llm_image_uri

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("version", type=str)
    args = parser.parse_args()
    uri = get_huggingface_llm_image_uri('huggingface-neuronx', version=args.version)
    print(uri)
