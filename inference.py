import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

# Example: Using transformers pipeline for VQA (replace with your model as needed)
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Load model and processor, move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")#change path
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base",torch_dtype=torch.float16).to(device)
    adapter_path = "weights"  # Change if needed
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            # encoding = processor(image, question, return_tensors="pt").to(device)
            inputs = processor(images=image, text=question, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs,max_new_tokens=10)
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            # print(e)
            answer = "error"
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()