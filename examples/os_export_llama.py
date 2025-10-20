
import sys
from pathlib import Path

import onnx_ir as ir

# Add the src folder to Python path so it can find torch_onnx_models
src_folder = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_folder))

from onnx_models._exporter import convert_hf_model

models = {
    "llama-3_2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    # "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
    # "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}

folder = r"C:\Data\osbm\\llama"
for name, model_id in models.items():
    print(f"Exporting {model_id} to ONNX...")
    onnx_model = convert_hf_model(model_id, load_weights=True)
    onnx_model.display()
    # TODO: Show progress bar
    output_path = Path(folder) / f"{name}.onnx"
    # print(f"Saving ONNX model to {output_path} ...")
    # ir.save(onnx_model, output_path) # external_data="data.onnx")

print("Done!")
