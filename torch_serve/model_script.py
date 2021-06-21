import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Model deploying arguments")
parser.add_argument("--model_name", default="convbert_model")
parser.add_argument("--version", default=1.0)
parser.add_argument("--serialize_model", action="store_true")
parser.add_argument("--deploy_model", action="store_true")
parser.add_argument("--model_path", default="models/checkpoint.pth")
args = parser.parse_args(sys.argv[1:2])

# model = ConvBert(lr=0.01, **{})

# # #TODO insert load of state dict

# script_model = torch.jit.script(model)
# script_model.save('torch_serve/' + str(args.model_name) + '.pt')
os.makedirs("model_store/", exist_ok=True)
if args.serialize_model:
    ss = (
        "torch-model-archiver --model-name "
        + str(args.model_name)
        + " --version "
        + str(args.version)
        + " --serialized-file "
    )
    ss = (
        ss
        + "torch_serve/"
        + str(args.model_name)
        + ".pt"
        + " --export-path model_store"
        + " --extra-files torch_serve/index_to_name.json"
        + " --handler text_classifier"
    )
    os.system(ss)

if args.deploy_model:
    ss = (
        "torchserve --start --ncs --model-store model_store --models "
        + str(args.model_name)
        + "="
        + str(args.model_name)
        + ".mar"
    )
    os.system(ss)
