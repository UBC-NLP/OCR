from json import load
from transformers import VisionEncoderDecoderModel
from PIL import Image
from transformers import TrOCRProcessor 
from io import BytesIO
import base64
import glob
import pandas as pd
from datasets import load_dataset, load_metric
from tqdm import tqdm


def predict(image_dir,model_path):
    #print(image_dir)
    image = image_dir.convert("RGB")
    processor = TrOCRProcessor.from_pretrained(model_path)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    #print(pixel_values.shape)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    #print(generated_text)

    return image, generated_text
    

def table_results(dataset_pred, save_dir,dataset_name="ADAB"):
    def image_base64(im):
        with BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()

    def image_formatter(im):
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

    ht = dataset_pred.to_html(
        formatters={'image': image_formatter}, escape=False)
    text_file = open(save_dir+"/"+"pred_{}.html".format(dataset_name), "w")
    text_file.write(ht)
    text_file.close()


def run(dataset_name, model_name, save_dir):
    cer_metric = load_metric("cer.py")
    wer_metric = load_metric("wer.py")
    #model_name = "/home/gagan30/scratch/arocr/results/ADAB-ARBERT-deit-base-distilled-patch16-224"
    print(dataset_name)
    dataset = load_dataset(
        "/home/gagan30/scratch/arocr/AraOCR_dataset", dataset_name, split="test")
    dataset_pred = pd.DataFrame(dataset)
    dataset_pred = dataset_pred.sample(n=100, random_state=42)
    print(dataset_pred.shape)
    texts = []
    images = []
    for image_jp in tqdm(dataset_pred['image'].to_list()):
        image, text = predict(image_jp, model_name)
        images.append(image)
        texts.append(text)

    dataset_pred_new = pd.DataFrame(
        {"image": dataset_pred['image'].to_list(), "pred": texts, "gt": dataset_pred['text'].to_list()})
    cer = cer_metric.compute(
        predictions=dataset_pred_new['pred'], references=dataset_pred_new['gt'])
    wer = wer_metric.compute(
        predictions=dataset_pred_new['pred'], references=dataset_pred_new['gt'])

    print("Dataset: {}".format(dataset_name))
    print("Char Error Rate: {}".format(cer))
    print("Word Error Rate: {}".format(wer))
    table_results(dataset_pred_new, save_dir, dataset_name)


def run_pixel(model_name, save_dir):
    data_files = glob.glob()
    texts = []
    images = []
    for image_jp in data_files:
        image, text = predict(image_jp, model_name)
        images.append(image)
        texts.append(text)

if __name__ == "__main__":
    datasets = ['IDPL-PFOD', 'OnlineKhatt',
                'ADAB', 'alexuw', 'MADBase', 'AHCD']
    for i in datasets:
        run(i, "/home/gagan30/scratch/arocr/models/trocr-base", "/home/gagan30/scratch/arocr/results")