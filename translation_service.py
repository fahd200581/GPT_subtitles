from abc import ABC, abstractmethod
import whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from googletrans import Translator as GoogleTranslator
import time
from translate_gpt import translate_with_gemini
from tqdm import tqdm
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def batch_text(result, gs=32):
    """split list into small groups of group size `gs`."""
    segs = result['segments']
    length = len(segs)
    mb = length // gs
    text_batches = []
    for i in range(mb):
        text_batches.append([s['text'] for s in segs[i * gs:(i + 1) * gs]])
    if mb * gs != length:
        text_batches.append([s['text'] for s in segs[mb * gs:length]])
    return text_batches

class ITranslationService(ABC):
    @abstractmethod
    def translate(self, text, src_lang, tr_lang):
        pass

class GoogleTranslateService(ITranslationService):
    def translate(self, result, src_lang='en', tr_lang='zh-cn'):
        if tr_lang == 'zh':
            tr_lang = 'zh-cn'
        translator = GoogleTranslator()
        batch_texts = batch_text(result, gs=25)
        translated = []
        
        for texts in tqdm(batch_texts):
            batch_translated = []
            for text in texts:
                inference_not_done = True
                while inference_not_done:
                    try:
                        translation = translator.translate(text, src=src_lang, dest=tr_lang)
                        inference_not_done = False
                    except Exception as e:
                        print(f"Waiting 15 seconds")
                        print(f"Error was: {e}")
                        time.sleep(15)
    
                batch_translated.append(translation.text)
            translated += batch_translated
        return translated

class M2M100TranslateService(ITranslationService):
    def translate(self, result, src_lang='en', tr_lang='zh'):
        model_name = "facebook/m2m100_418M"
        model = M2M100ForConditionalGeneration.from_pretrained(model_name).to('cuda')
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        tokenizer.src_lang = src_lang
        translated = []
        batch_texts = batch_text(result, gs=32)
        for texts in tqdm(batch_texts):
            batch_translated = []
            for text in texts:
                encoded = tokenizer(text, return_tensors="pt", padding=True).to('cuda')
                generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tr_lang))
                translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                batch_translated += translated_text
            translated += batch_translated
        return translated

class GeminiTranslateService(ITranslationService):
    def translate(self, result, src_lang='en', tr_lang='zh'):
        model = genai.GenerativeModel('gemini-1.5-flash')
        translated = []
        batch_texts = batch_text(result, gs=32)
        
        for texts in tqdm(batch_texts):
            batch_translated = []
            for text in texts:
                response = model.generate_content(text)
                batch_translated.append(response.text)
            translated += batch_translated
        return translated

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Translate subtitles using different translation services")
    parser.add_argument("input_file", type=str, help="Path to the input subtitle file")
    parser.add_argument("output_file", type=str, help="Path to the output subtitle file")
    parser.add_argument("service", type=str, choices=["google", "m2m100", "gemini"], help="Translation service to use")
    parser.add_argument("-s", "--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("-t", "--tr_lang", type=str, default="zh", help="Target language")

    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        result = {'segments': [{'text': line.strip()} for line in f.readlines()]}

    if args.service == "google":
        service = GoogleTranslateService()
    elif args.service == "m2m100":
        service = M2M100TranslateService()
    elif args.service == "gemini":
        service = GeminiTranslateService()

    translated_text = service.translate(result, src_lang=args.src_lang, tr_lang=args.tr_lang)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(translated_text))
