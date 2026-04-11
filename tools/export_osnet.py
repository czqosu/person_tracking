"""Export OSNet x0.25 (msmt17) to ONNX for NvDeepSORT re-ID."""
import torch
import torchreid
from torchreid.reid.utils import load_pretrained_weights
import gdown
import os

OUT = "models/osnet_x0_25_msmt17.onnx"
WEIGHTS = "/tmp/osnet_x0_25_msmt17.pt"
os.makedirs("models", exist_ok=True)

# Download msmt17 re-ID pretrained weights (NOT imagenet)
if not os.path.exists(WEIGHTS):
    gdown.download(
        "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
        WEIGHTS, quiet=False
    )

model = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1041,   # msmt17 has 1041 IDs
    pretrained=False,
)
load_pretrained_weights(model, WEIGHTS)
model.eval()

dummy = torch.zeros(1, 3, 256, 128)
torch.onnx.export(
    model, dummy, OUT,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11,
)
print(f"Exported: {OUT}")

# quick sanity check
import onnx
m = onnx.load(OUT)
onnx.checker.check_model(m)
print("ONNX check passed")
print(f"Input : {m.graph.input[0].type.tensor_type.shape}")
print(f"Output: {m.graph.output[0].type.tensor_type.shape}")
