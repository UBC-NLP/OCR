from transformers import VisionEncoderDecoderModel
from PIL import Image
from transformers import TrOCRProcessor

def predict(image_dir,model_path):
    image = Image.open(image_dir).convert("RGB")
    processor = TrOCRProcessor.from_pretrained(model_path)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    print(pixel_values.shape)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

if __name__ == "__main__":
    predict("/project/6005442/DataBank/ocr-project/new-khatt/OnlineKHATT_A0001_1_10_1.jpg","/home/gagan30/scratch/arocr/results/ADAB-ARBERT-deit-base-distilled-patch16-224")