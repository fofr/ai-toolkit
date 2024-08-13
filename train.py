# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BaseModel, Input, Path, Secret
import os
import subprocess
from zipfile import ZipFile
from huggingface_hub import HfApi


class TrainingOutput(BaseModel):
    weights: Path


# Run in lucataco/sandbox2
def train(
    images: Path = Input(
        description="A zip/tar file containing the images that will be used for training. File names must be their captions: a_photo_of_TOK.png, etc. Min 12 images required."
    ),
    model_name: str = Input(
        description="Model name", default="black-forest-labs/FLUX.1-dev"
    ),
    hf_token: Secret = Input(
        description="HuggingFace token to use for accessing model"
    ),
    steps: int = Input(
        description="Number of training steps. Recommended range 500-4000",
        ge=10,
        le=4000,
        default=1000,
    ),
    learning_rate: float = Input(description="Learning rate", default=4e-4),
    batch_size: int = Input(description="Batch size", default=1),
    resolution: str = Input(
        description="Image resolutions for training", default="512,768,1024"
    ),
    lora_linear: int = Input(description="LoRA linear value", default=16),
    lora_linear_alpha: int = Input(description="LoRA linear alpha value", default=16),
    repo_id: str = Input(
        description="Enter HuggingFace repo id to upload LoRA to HF. Will return zip file if left empty.Ex: lucataco/flux-dev-lora",
        default=None,
    ),
) -> TrainingOutput:
    """Run a single prediction on the model"""
    print("Starting prediction")
    # Cleanup previous runs
    os.system("rm -rf output")

    # Set huggingface token via huggingface-cli login
    os.system(f"huggingface-cli login --token {hf_token.get_secret_value()}")

    # replace the 'name_or_path: 'black-forest-labs/FLUX.1-dev' value in the config file
    os.system(
        f"sed -i 's/name_or_path: 'black-forest-labs/FLUX.1-dev'/name_or_path: {model_name}/' config/replicate.yml"
    )
    # replace the 'steps: 1000' value in the config file
    os.system(f"sed -i 's/steps: 1000/steps: {steps}/' config/replicate.yml")
    # replace the 'save_every: 1001' value in the config file
    os.system(
        f"sed -i 's/save_every: 1001/save_every: {steps+1}/' config/replicate.yml"
    )
    # replace the 'lr: 4e-4' value in the config file
    os.system(f"sed -i 's/lr: 4e-4/lr: {learning_rate}/' config/replicate.yml")
    # replace the 'batch_size: 1' value in the config file
    os.system(
        f"sed -i 's/batch_size: 1/batch_size: {batch_size}/' config/replicate.yml"
    )
    # replace the 'resolution: [ 512, 768, 1024 ]' value in the config file
    os.system(
        f"sed -i 's/resolution: [ 512, 768, 1024 ]/resolution: [{resolution}]/' config/replicate.yml"
    )
    # replace the 'linear: 16' value in the config file
    os.system(f"sed -i 's/linear: 16/linear: {lora_linear}/' config/replicate.yml")
    # replace the 'linear_alpha: 16' value in the config file
    os.system(
        f"sed -i 's/linear_alpha: 16/linear_alpha: {lora_linear_alpha}/' config/replicate.yml"
    )

    # Unzip images from input images file to the input_images folder
    input_dir = "input_images"
    input_images = str(images)
    if input_images.endswith(".zip"):
        print("Detected zip file")
        os.makedirs(input_dir, exist_ok=True)
        with ZipFile(input_images, "r") as zip_ref:
            zip_ref.extractall(input_dir + "/")
        print("Extracted zip file")
    elif input_images.endswith(".tar"):
        print("Detected tar file")
        os.makedirs(input_dir, exist_ok=True)
        os.system(f"tar -xvf {input_images} -C {input_dir}")
        print("Extracted tar file")

    # Run - bash train.sh
    subprocess.check_call(["python", "run.py", "config/replicate.yml"], close_fds=False)

    # Zip up the output folder
    output_lora = "output/flux_train_replicate"
    # copy license file to output folder
    os.system(f"cp lora-license.md {output_lora}/README.md")
    output_zip_path = "/tmp/output.zip"
    os.system(f"zip -r {output_zip_path} {output_lora}")

    # cleanup input_images folder
    os.system(f"rm -rf {input_dir}")

    if hf_token is not None and repo_id is not None:
        api = HfApi()
        api.upload_folder(
            repo_id=repo_id,
            folder_path=output_lora,
            repo_type="model",
            use_auth_token=hf_token,
        )
    return TrainingOutput(weights=Path(output_zip_path))
