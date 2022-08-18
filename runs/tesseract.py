from datasets import load_dataset
import pandas as pd
from datasets import load_metric
import pytesseract
from io import BytesIO
import base64
from transformers import VisionEncoderDecoderModel
from PIL import Image
from transformers import TrOCRProcessor
from tqdm import tqdm

def table_results(dataset_pred, results, save_dir, dataset_name="ADAB"):
    def image_base64(im):
        with BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()

    def image_formatter(im):
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

    res = results.to_html(escape=False)
    ht = dataset_pred.to_html(
        formatters={'image': image_formatter}, escape=False)
    text_file = open(save_dir+"/"+"pred_{}.html".format(dataset_name), "w")
    text_file.write(res)
    text_file.write(ht)
    text_file.close()


def predict(image_dir, model_path):
    #print(image_dir)
    image = image_dir.convert("RGB")
    processor = TrOCRProcessor.from_pretrained(model_path)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    #print(pixel_values.shape)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    # print(generated_text)

    return generated_text

def main(dataset_name):
    dataset = load_dataset(
        "/home/gagan30/scratch/arocr/AraOCR_dataset", dataset_name)

    df_pred = pd.DataFrame(dataset['test'])
    # df_pred = df_pred.sample(n=3, random_state=42)
    cer = load_metric("cer.py")
    wer = load_metric("wer.py")

    custom_config = r'--tessdata-dir "/home/gagan30/scratch/arocr/code/tessdata_best" --oem 3 --psm 6 -c page_separator=""'

    output_list = []
    texts = []
    texts_pred = []
    for img in tqdm(df_pred['image'].to_list()):
        output = pytesseract.image_to_string(img, config=custom_config, lang='ara')
        output = output.replace('\n', '')
        output_list.append(output)
        text = predict(img, "../models/trocr-base")
        texts.append(text)
        text_pred = predict(img, "../results/ADAB-ARBERT-deit-base-distilled-patch16-224")
        texts_pred.append(text_pred)

    df_pred['pred_tesseract'] = output_list
    df_pred['pred_trocr'] = texts
    df_pred['pred_ours'] = texts_pred
    cer_score_tess = cer.compute(
        predictions=df_pred['pred_tesseract'].to_list(), references=df_pred['text'].to_list())
    wer_score_tess = wer.compute(
        predictions=df_pred['pred_tesseract'].to_list(), references=df_pred['text'].to_list())
    cer_score_trocr = cer.compute(
        predictions=df_pred['pred_trocr'].to_list(), references=df_pred['text'].to_list())
    wer_score_trocr = wer.compute(
        predictions=df_pred['pred_trocr'].to_list(), references=df_pred['text'].to_list())
    cer_score_our = cer.compute(
        predictions=df_pred['pred_ours'].to_list(), references=df_pred['text'].to_list())
    wer_score_our = wer.compute(
        predictions=df_pred['pred_ours'].to_list(), references=df_pred['text'].to_list())
    
    print("Tesseract: CER: {}, WER: {}".format(cer_score_tess, wer_score_tess))
    print("Trocr: CER: {}, WER: {}".format(cer_score_trocr, wer_score_trocr))
    print("Ours: CER: {}, WER: {}".format(cer_score_our, wer_score_our))
    results = {
        "Tesseract": {"CER": cer_score_tess, "WER": wer_score_tess},
        "Trocr": {"CER": cer_score_trocr, "WER": wer_score_trocr},
        "Ours": {"CER": cer_score_our, "WER": wer_score_our}
    }
    df_results = pd.DataFrame.from_dict(results).T
    table_results(
        df_pred.head(15), df_results, "/home/gagan30/scratch/arocr/results", dataset_name)
    
if __name__ == "__main__":
    
