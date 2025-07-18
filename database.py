from pymorphy2 import MorphAnalyzer
import zipfile
import requests
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pytesseract
from PIL import Image
import io
import re
import pdfplumber
#from pdf2image import convert_from_bytes
import csv
#from collections import defaultdict
#import pypdf
#from pypdf import PdfReader
#import pdfminer
from pdfminer.high_level import extract_text as pdfminer_extract_text
from PIL import Image
#from paddleocr import PaddleOCR
import cv2
import numpy as np

morph = MorphAnalyzer()
lemmatizer = WordNetLemmatizer()

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


zip_url = 'https://drive.google.com/uc?export=download&id=1EcoqCzZU4MYXWTx4Himtso0hT8Xhq37E'
classificator_url = 'http://193.232.208.58:9001/filewatcher/get_classificator'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_term(text):
    words = text.split()
    lemmatized = []

    for word in words:
        lang = '1' if any(cyr in word.lower() for cyr in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя') else '2'
        word_lower = word.lower()
        
        if lang == '1':
            parsed = morph.parse(word_lower)[0]
            lemma = parsed.normal_form
            if word.isupper():
                lemma = word
        else:
            lemma = lemmatizer.lemmatize(word_lower, get_wordnet_pos(word_lower))
            if word.isupper():
                lemma = word
        
        lemmatized.append(lemma)
    
    return ' '.join(lemmatized) if lemmatized else None

def get_terms(terms_list): 
    terms_dict = {}
    if isinstance(terms_list, dict):
        for language in ['1', '2']:   # 1 - русский, 2 - англ
            if language in terms_list:
                if isinstance(terms_list[language], str):
                    original_term = terms_list[language]
                    if original_term.isupper():
                        terms_dict[original_term] = original_term
                    else:
                        lemmatized = lemmatize_term(original_term)
                        terms_dict[lemmatized] = original_term

        for val in terms_list.values():
            terms_dict.update(get_terms(val))

    elif isinstance(terms_list, list):
        for val in terms_list:
            terms_dict.update(get_terms(val))

    return terms_dict


def get_text(article):
    text = pdfminer_extract_text(io.BytesIO(article))
    
    with pdfplumber.open(io.BytesIO(article)) as pdf:
        for page in pdf.pages:
            if page.images:
                try:
                    img = page.to_image(resolution=300).original
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_pil = Image.open(img_bytes)
                    
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    ocr_text = pytesseract.image_to_string(binary, lang='rus+eng')
                    new_text = ""
                    
                    for line in ocr_text.split('\n'):
                        if line.strip() not in text:
                            new_text += line + " "
                    
                    text += " " + new_text.strip()
                except:
                    continue
    
    return re.sub(r'\s+', ' ', text).strip()


def find_terms_in_text(text, terms):
    found_terms = []
    word_pattern = re.compile(r'\b\w+\b')

    for term_lemmatized, original_term in terms.items():
        term_words = term_lemmatized.split()
        
        if len(term_words) > 1:
            if any(cyr in term_lemmatized.lower() for cyr in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
                patterns = []
                for word in term_words:
                    if word.isupper():
                        patterns.append(re.escape(word))
                    else:
                        parsed = morph.parse(word.lower())[0]
                        forms = {f.word for f in parsed.lexeme} | {word.lower()}
                        patterns.append('(' + '|'.join(re.escape(f) for f in forms) + ')')
                pattern = re.compile(r'\b' + r'\s+'.join(patterns) + r'\b', re.IGNORECASE)
            else:
                flags = re.IGNORECASE if not term_lemmatized.isupper() else 0
                pattern = re.compile(r'\b' + r'\s+'.join(re.escape(w) for w in term_words) + r'\b', flags)
            
            found_terms.extend({
                'term': original_term,
                'position': match.start(),
                'matched_text': match.group()
            } for match in pattern.finditer(text))
    
    for match in word_pattern.finditer(text):
        word = match.group()
        lemma = lemmatize_term(word)
        
        if lemma in terms:
            found_terms.append({
                'term': terms[lemma],
                'position': match.start(),
                'matched_text': word
            })
    
    return found_terms


res_classificator = requests.get(classificator_url)

terms_list = res_classificator.json()
terms = get_terms(terms_list['classificator'])

res_zip = requests.get(zip_url)
zip_content = io.BytesIO(res_zip.content)

#articles = []
csv_data = []

with zipfile.ZipFile(zip_content) as zip_file:
    for file_info in zip_file.infolist():
        if file_info.is_dir():
            continue
        if not file_info.filename.lower().endswith('.pdf'):
            continue
            
        with zip_file.open(file_info) as f:
            content = f.read()
        
            text = get_text(content)
        
            path_parts = file_info.filename.split('/')
            folder_name = path_parts[-2] if len(path_parts) > 1 else ''
            file_name = path_parts[-1]

            articles_terms = find_terms_in_text(text, terms)

            for term in articles_terms:
                csv_data.append({
                    'Термин': term['term'],
                    'Место встречи термина': term['position'],
                    'Название публикации': file_name,
                    'Название папки статьи': folder_name
    })
            


with open('terms.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    fieldnames = ['Термин', 'Место встречи термина', 'Название публикации', 'Название папки статьи']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

