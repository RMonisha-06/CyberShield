import zipfile
import json
import shutil
import os

model_path = "models/baseline_bilstm.keras"
temp_path = "models/baseline_bilstm_fixed.keras"

with zipfile.ZipFile(model_path, "r") as zin:
    config = json.loads(zin.read("config.json"))

    for layer in config["config"]["layers"]:
        if layer.get("class_name") == "Embedding":
            layer["config"].pop("quantization_config", None)

    with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "config.json":
                zout.writestr(
                    item,
                    json.dumps(config, separators=(",", ":"))
                )
            else:
                zout.writestr(item, zin.read(item.filename))

print("Fixed model created:", temp_path)
