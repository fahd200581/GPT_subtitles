import google.generativeai as genai
import ujson
import os
from tqdm import tqdm
import re
import argparse
import time
from dotenv import load_dotenv
import logging

def check_for_errors(log_file_path, starting_line):
    if not os.path.exists(log_file_path):
        print(f"No log file found at {log_file_path}")
        return False
    
    error_occurred = False
    with open(log_file_path, 'r') as log_file:
        for _ in range(starting_line):
            next(log_file, None) 
        
        for line in log_file:
            if "- ERROR -" in line:
                error_occurred = True
                break
    return error_occurred
    
def count_log_lines(log_file_path):
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            return sum(1 for line in file)
    else:
        return 0

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def count_token(str):
    return len(genai.GenerativeModel.count_tokens(str).total_tokens)

class Subtitle:
    def __init__(self, file_path):
        self.file_path = file_path
        self.content = self.load_subtitles()

    def load_subtitles(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def save_subtitles(self, file_path, content):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def split_subtitles(self, batch_size):
        subtitle_blocks = self.content.strip().split('\n\n')
        batches = []

        for i in range(0, len(subtitle_blocks), batch_size):
            batch = '\n\n'.join(subtitle_blocks[i:i + batch_size])
            batches.append(batch)

        return batches

    def process_subtitles(self, subtitles):
        lines = subtitles.split('\n')
        processed_lines = []
        timestamps = []

        for line in lines:
            if re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
                timestamps.append(line)
            else:
                processed_lines.append(line)

        return '\n'.join(processed_lines), timestamps

    def get_processed_batches_and_timestamps(self, batch_size):
        subtitle_batches = self.split_subtitles(batch_size)
        processed_batches = []
        timestamps_batches = []
        for batch in subtitle_batches:
            processed_batch, timestamps = self.process_subtitles(batch)
            processed_batches.append(processed_batch)
            timestamps_batches.append(timestamps)
        return processed_batches, timestamps_batches

class TranslationMapping:
    def __init__(self, max_size):
        self.max_size = max_size
        self.mapping_dict = {}
        self.translations = set()
        self.all_mappings = []
        self.current_index = 0

    def add_mapping(self, new_mapping, translations):
        for subtitle in translations:
            if not isinstance(subtitle["index"], int):
                continue
                
            index = subtitle["index"]
            self.current_index = max(self.current_index, index)
            translation = subtitle["translation"]
            original_text = subtitle["original_text"]

            words = re.findall(r'\b\w+\b', original_text)

            for word in words:
                if word in self.mapping_dict:
                    self.mapping_dict[word]['frequency'] += 1
                    self.mapping_dict[word]['index'] = index
                    self.calculate_score(word)

        for term, translation in new_mapping.items():
            proper_noun = term.lower().strip()

            if proper_noun not in self.mapping_dict and translation not in self.translations:
                if len(self.mapping_dict) == self.max_size:
                    proper_noun_to_remove = min(self.mapping_dict, key=lambda x: self.mapping_dict[x]['score'])
                    removed_translation = self.mapping_dict[proper_noun_to_remove]['translation']
                    del self.mapping_dict[proper_noun_to_remove]
                    self.translations.remove(removed_translation)

                self.mapping_dict[proper_noun] = {'translation': translation, 'frequency': 1, 'index': self.current_index, 'score': 0}
                self.translations.add(translation)

            self.all_mappings.append((proper_noun, translation))

        self.mapping_dict = dict(sorted(self.mapping_dict.items(), key=lambda item: item[0]))
        self.all_mappings = sorted(self.all_mappings, key=lambda item: item[0])

    def calculate_score(self, proper_noun):
        frequency_score = self.mapping_dict[proper_noun]['frequency']
        index_difference = self.current_index - self.mapping_dict[proper_noun]['index'] + 1
        recency_score = 1 / index_difference
        self.mapping_dict[proper_noun]['score'] = frequency_score * recency_score

    def get_mappings(self):
        return {proper_noun: mapping['translation'] for proper_noun, mapping in self.mapping_dict.items()}

    def get_all_mappings(self):
        unique_mappings = list(set(self.all_mappings))
        sorted_mappings = sorted(unique_mappings, key=lambda item: item[0])
        return "\n".join(f"{proper_noun} : {translation}" for proper_noun, translation in sorted_mappings)

    def get_current_mappings(self):
        sorted_mappings = sorted(self.mapping_dict.items(), key=lambda item: item[0])
        return "\n".join(f"{proper_noun} : {mapping['translation']}" for proper_noun, mapping in sorted_mappings)

def merge_subtitles_with_timestamps(translated_subtitles, timestamps):
    translated_lines = translated_subtitles.split('\n')
    merged_lines = []

    timestamp_idx = 0
    for line in translated_lines:
        if re.match(r'\d+\s*$', line):
            merged_lines.append(line)
            merged_lines.append(timestamps[timestamp_idx])
            timestamp_idx += 1
        else:
            merged_lines.append(line)

    return '\n'.join(merged_lines)

def count_blocks(subtitle_string):
    if not subtitle_string.endswith('\n'):
        subtitle_string += '\n'
    return len(re.findall(r'(\d+\n(?:.+\n)+)', subtitle_string))

def check_response(input_subtitles, translated_subtitles):
    if not translated_subtitles.endswith('\n'):
        translated_subtitles += '\n'

    input_blocks = re.findall(r'(\d+\n(?:.+\n)+)', input_subtitles)
    translated_blocks = re.findall(r'(\d+\n(?:.+\n)+)', translated_subtitles)
    additional_content = re.sub(r'\d+\n(?:.+\n)+', '', translated_subtitles).strip()

    problematic_blocks = []
    for i, (input_block, translated_block) in enumerate(zip(input_blocks, translated_blocks)):
        input_lines = input_block.strip().split('\n')
        translated_lines = translated_block.strip().split('\n')

        if len(input_lines) != len(translated_lines):
            problematic_blocks.append((i, translated_block))
            continue

        input_line_number = int(input_lines[0])
        translated_line_number = int(translated_lines[0])

        if input_line_number != translated_line_number:
            problematic_blocks.append((i, translated_block))

    return len(translated_blocks), additional_content, problematic_blocks

class Translator:
    def __init__(self, model='gemini-1.5-flash', batch_size=40, target_language='zh', source_language='en', titles='Video Title not found', video_info=None, input_path=None, no_translation_mapping=False, load_from_tmp=False):
        self.model = model
        self.batch_size = batch_size
        self.target_language = target_language
        self.source_language = source_language
        self.titles = titles
        self.video_info = video_info
        self.input_path = input_path
        
        self.translation_mapping = TranslationMapping(max_size=40)
        self.no_translation_mapping = no_translation_mapping
        self.load_from_tmp = load_from_tmp
        self.translate_max_retry = 2
        
        with open('few_shot_examples.json', 'r') as f:
            few_shot_examples = ujson.load(f)
        
        try:
            self.few_shot_examples = few_shot_examples[f"{self.source_language}-to-{self.target_language}"]
        except KeyError:
            print("No few shot examples found for this language pair. Please add some examples to few_shot_examples.json. Use default examples (en-to-zh)")
            self.few_shot_examples = few_shot_examples["en-to-zh"]
            
        if target_language == "zh":
            self.target_language = "Simplified Chinese"
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(os.path.join(os.path.dirname(input_path), 'translator.log'))
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.openai_logger = logging.getLogger('Gemini_Response')
        self.openai_logger.setLevel(logging.DEBUG)

        openai_file_handler = logging.FileHandler(os.path.join(os.path.dirname(input_path), 'response.log'))
        openai_file_handler.setLevel(logging.DEBUG)
        openai_formatter = logging.Formatter('%(message)s')
        openai_file_handler.setFormatter(openai_formatter)
        self.openai_logger.addHandler(openai_file_handler)

    def process_line(self, line):
        subtitles = []
        lines = line.split("\n")
        
        i = 0
        while i < len(lines):
            if lines[i].strip() == "":
                i += 1
                continue
            
            number = int(lines[i])
            i += 1
            original_text = lines[i]
            
            subtitles.append({"index": number, "original_text": original_text})
            
            i += 2
            
        return subtitles
            
    def send_to_gemini(self, subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, subtitles_length, warning_message=None, prev_response=None):
        model = genai.GenerativeModel(self.model)
        
        user_input = self.process_user_input(subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, warning_message)
        
        messages = [{"role": "system", "content": self.system_content()}]
        for example in self.few_shot_examples["examples"]:
            messages.append({"role": "user", "content": ujson.dumps(example["input"], ensure_ascii=False, indent=2)})
            messages.append({"role": "assistant", "content": ujson.dumps(example["output"], ensure_ascii=False, indent=2)})
        messages.append({"role": "user", "content": ujson.dumps(user_input, ensure_ascii=False, indent=2)})
        
        self.logger.info("========Messages========\n")
        self.logger.info(messages)
        self.logger.info("========End of Messages========\n")
        
        max_retries = self.translate_max_retry
        retry_count = 0
        translated_subtitles = ''
        while retry_count < max_retries:
            try:
                response = model.generate_content(user_input["current_batch_subtitles"], stream=True)
                response.resolve()
                translated_subtitles = response.text
                
                self.openai_logger.info(translated_subtitles)
                
                data = ujson.loads(translated_subtitles)
                output_string = ""
                for subtitle in data["current_batch_subtitles_translation"]:
                    index = subtitle["index"]
                    translation = subtitle["translation"]
                    output_string += f"{index}\n{translation}\n\n"
                
                translation_mapping = data["translation_mapping"]
                self.translation_mapping.add_mapping(translation_mapping, data["current_batch_subtitles_translation"])
                
                return output_string

            except ujson.JSONDecodeError as e:
                retry_count += 1
                self.logger.error(f"An error occurred while parsing JSON: {e}. Retrying {retry_count} of {max_retries}.")
                warning_message = f"Your response is not in a valid JSON format. Please double-check your answer. Error:{e}"
                if warning_message:
                    user_input["Warning_message"] = f"In a previous request sent to Gemini, the response is problematic. Please double-check your answer. Warning message: {warning_message} Retry count: {retry_count} of {max_retries}"
                messages[-1]["content"] = ujson.dumps(user_input)
                time.sleep(10)
            
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                return translated_subtitles

        self.logger.error("Max retries reached. Unable to get valid JSON response.")
        return translated_subtitles

    def process_user_input(self, subtitles, prev_subtitle, next_subtitle, prev_translated_subtitle, warning_message):
        user_input = {}
        
        if prev_subtitle:
            previous_subtitles = self.process_line(prev_subtitle)
            if prev_translated_subtitle:
                translated_subtitles = self.process_line(prev_translated_subtitle)
                index_to_translation = {item["index"]: item["original_text"] for item in translated_subtitles}
                for item in previous_subtitles:
                    if item["index"] in index_to_translation:
                        item["translation"] = index_to_translation[item["index"]]
                    else:
                        self.logger.info(f"Error: index {item['index']} not found in translated subtitles")
                    user_input["previous_batch_subtitles"] = previous_subtitles
        
        if subtitles:
            user_input["current_batch_subtitles"] = self.process_line(subtitles)
        
        if next_subtitle:
            user_input["next_batch_subtitles"] = self.process_line(next_subtitle)
            
        if warning_message:
            user_input["Warning_message"] = f"In a previous request sent to Gemini, the response is problematic. Please double-check your answer. Warning message: {warning_message}"
        
        if len(self.translation_mapping.get_mappings()) != 0 and not self.no_translation_mapping:
            user_input["translation_mapping"] = self.translation_mapping.get_mappings()
            
        return user_input

    def system_content(self):
        return f"""You are a program responsible for translating subtitles. Your task is to translate the current batch of subtitles into {self.target_language} for the video titled '{self.titles}' and follow the guidelines below.
Guidelines:
- Keep in mind that each index should correspond exactly with the original text, and your translation should be faithful to the context of each sentence.
- Translate with informal slang if necessary, ensuring that the translation is accurate and reflects the context and terminology. Please do not output any text other than the translation. 
- You will also receive some additional information for your reference only, such as the previous batch of subtitles, the translation of the previous batch, the next batch of subtitle, and maybe error messages. 
- Please ensure that each line in the current batch has a corresponding translated line. 
- If the last sentence in the current batch is incomplete, you may ignore the last sentence. If the first sentence in the current batch is incomplete, you may combine the last sentence in the last batch to make the sentence complete. 
- Please only output the translation of the current batch of subtitles (current_batch_subtitles_translation).
- Do not put the translation of the next whole sentence in the current sentence.
- Each index in the current batch of subtitles must correspond to the exact original text and translation. Do not combine sentences from different indices.
- Ensure that the number of lines in the current batch of subtitles is the same as the number of lines in the translation.
- You may translate with conversational language if the original text is informal.
- Additional information for the video: {self.video_info}
- Please ensure that the translation and the original text are matched correctly.
- You may receive original text in other languages, but please only output {self.target_language} translation.

- Please translate the following subtitles and summarize all the proper nouns that appear to generate a mapping.
- Please only output the proper nouns and their translation that appear in the current batch of subtitles, do not repeat from the input
- You may receive translation_mapping as input, which is a mapping of proper nouns to their translation in {self.target_language}. 
- Please follow this mapping to translate the subtitles to improve translation consistency.

- Target language: {self.target_language}

- Please output proper JSON with this format:
{{
    "current_batch_subtitles_translation": [
        {{
            "index": <int>,
            "original_text": <str>,
            "translation": <str>
        }}
    ],
    "translation_mapping": {{
        "proper nouns": <translation in target language>
    }}
}}"""

    def batch_translate(self, subtitle_batches, timestamps_batches):
        translated = []
        prev_translated_subtitle = None
        
        def extract_line(text, num_lines, is_next=False):
            if text is None:
                return None
            entries = text.split('\n\n')
            if is_next:
                selected_entries = '\n\n'.join(entries[:num_lines])
            else:
                selected_entries = '\n\n'.join(entries[-num_lines:])
            return selected_entries
            
        tmp_file = os.path.join(os.path.dirname(self.input_path), 'tmp_subtitles.json')
        skip_length = 0
        if os.path.exists(tmp_file):
            with open(tmp_file, 'r') as f:
                previous_subtitles = ujson.load(f)
            if self.load_from_tmp:
                translated = previous_subtitles
                skip_length = len(translated)
        
        for i, t in enumerate(tqdm(subtitle_batches)):
            if skip_length > 0:
                skip_length -= 1
                continue
                
            prev_subtitle = subtitle_batches[i - 1] if i > 0 else None
            next_subtitle = subtitle_batches[i + 1] if i < len(subtitle_batches) - 1 else None
            prev_subtitle = extract_line(prev_subtitle, 2)
            next_subtitle = extract_line(next_subtitle, 2, is_next=True)

            tt = self.send_to_gemini(t, prev_subtitle, next_subtitle, prev_translated_subtitle, len(t))
            prev_translated_subtitle = tt
            tt_merged = merge_subtitles_with_timestamps(tt, timestamps_batches[i])
            self.logger.info("========Batch summary=======\n")
            self.logger.info(t)
            self.logger.info(tt_merged)
            self.logger.info("========End of Batch summary=======\n")
            translated.append(tt_merged)
            
            with open(tmp_file, 'w') as f:
                ujson.dump(translated, f, ensure_ascii=False, indent=2)

        translated = ''.join(translated)

        self.logger.info(translated)
        self.logger.info("========translation mapping=======\n")
        self.logger.info(self.translation_mapping.get_all_mappings())
        
        return translated

def translate_with_gemini(input_file, target_language='zh', source_language='en', batch_size=40, model='gemini-1.5-flash', video_info=None, no_translation_mapping=False, load_from_tmp=False):
    log_file_path = os.path.join(os.path.dirname(input_file), 'translator.log')
    starting_line = count_log_lines(log_file_path)
    
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    
    subtitle = Subtitle(input_file)
    translator = Translator(model=model, batch_size=batch_size, target_language=target_language, source_language=source_language, 
        titles=file_name, video_info=video_info, input_path=input_file, no_translation_mapping=no_translation_mapping, load_from_tmp=load_from_tmp)

    subtitle_batches, timestamps_batches = subtitle.get_processed_batches_and_timestamps(batch_size)
    translated_subtitles = translator.batch_translate(subtitle_batches, timestamps_batches)

    output_file = os.path.join(os.path.dirname(input_file), f"{os.path.splitext(os.path.basename(input_file))[0]}_{target_language}_gemini.srt")
    subtitle.save_subtitles(output_file, translated_subtitles)
    
    if check_for_errors(log_file_path, starting_line):
        print("An error was logged. Please search '- ERROR -' in translator.log for more details.")
    
def main():
    parser = argparse.ArgumentParser(description='Translate subtitles using Gemini')
    parser.add_argument('-i', '--input_file', help='The path to the input subtitle file.', type=str, required=True)
    parser.add_argument('-b', '--batch_size', help='The number of subtitles to process in a batch.', type=int, default=12)
    parser.add_argument('-l', '--target_language', help='The target language for translation.', default='zh')
    parser.add_argument('-s', '--source_language', help='The source language for translation.', default='en')
    parser.add_argument('-v', "--video_info", type=str, default="", help="Additional information about the video.")
    parser.add_argument('-m', '--model', default='gemini-1.5-flash', help='Model for Gemini API', type=str)
    parser.add_argument('-um', "--no_mapping", action='store_true', help="don't use translation mapping as input to the model")
    parser.add_argument('-lt', "--load_tmp_file", action='store_true', help="load the previous translated subtitles, assume previous tmp file generated with the same setting as the current run")
    
    args = parser.parse_args()

    translate_with_gemini(args.input_file, args.target_language, args.source_language, 
                args.batch_size , args.model, args.video_info, args.no_mapping, args.load_tmp_file)

if __name__ == "__main__":
    main()
