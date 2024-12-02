import streamlit as st
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from model import KoBARTConditionalGeneration
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

@st.cache_resource()
def load_model():
    import torch
    model = KoBARTConditionalGeneration.load_from_checkpoint(
        "./YourModelPath.ckpt",  # 체크포인트 파일이 같은 디렉토리에 있는 경우
        map_location=torch.device('cpu')
    )
    return model

@st.cache_resource()
def load_tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    return tokenizer

@st.cache_resource()
def load_model1():
    model1 = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
    return model1

@st.cache_resource()
def load_tokenizer1():
    tokenizer1 = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    return tokenizer1

summarization_model = load_model()
summarization_tokenizer = load_tokenizer()
model1 = load_model1()
tokenizer1 = load_tokenizer1()

@st.cache_resource
def initialize_translator():
    text_translator = Translator()
    return text_translator

text_translator = initialize_translator()

# 원문과 요약문의 길이, 주요 키워드 포함률 등을 측정하여 요약의 품질을 평가
def evaluate_summary(original_article, generated_summary):
    # 원문과 요약문의 단어 수 계산
    original_word_count = len(original_article.split())
    summary_word_count = len(generated_summary.split())
    
    # 요약 압축률 계산
    compression_ratio = (summary_word_count / original_word_count) * 100

    # 키워드를 추출해 공통된 키워드를 찾음
    original_keywords_set = set(re.findall(r'\b\w+\b', original_article.lower()))
    summary_keywords_set = set(re.findall(r'\b\w+\b', generated_summary.lower()))
    common_keywords_set = original_keywords_set.intersection(summary_keywords_set)
    
    # 주요 키워드 포함률 계산
    keyword_inclusion_rate = (len(common_keywords_set) / len(original_keywords_set)) * 100

    return original_word_count, summary_word_count, compression_ratio, keyword_inclusion_rate

def compute_similarity(original_text, summarized_text):
    # 원문과 요약문을 TF-IDF로 벡터화
    tfidf_matrix = TfidfVectorizer().fit_transform([original_text, summarized_text])
    vector_array = tfidf_matrix.toarray()
    
    # 두 벡터 간의 코사인 유사도를 계산하여 반환
    cosine_sim = cosine_similarity(vector_array)[0, 1]
    return cosine_sim

st.title("KoBART 요약 Test")

# 선택 가능한 언어들
language_options = {
    "한국어": "ko",
    "영어": "en",
    "일본어": "ja",
    "중국어(간체)": "zh-cn",
    "프랑스어": "fr",
    "독일어": "de"
}

# 사용자가 요약할 텍스트와 결과물의 언어를 선택
input_language_selection = st.selectbox("요약할 텍스트의 언어를 선택하세요:", list(language_options.keys()))
output_language_selection = st.selectbox("요약 결과물의 언어를 선택하세요:", list(language_options.keys()))

# 선택한 언어의 코드를 가져옴 (한국어, 영어 등 -> ko, en 등)
input_language_code = language_options[input_language_selection]
output_language_code = language_options[output_language_selection]

# 뉴스 기사 입력
news_text = st.text_area("뉴스 입력:")

if news_text:
    # 입력된 뉴스 기사의 언어를 감지하여 사용자가 선택한 언어와 일치하는지 확인
    detected_input_language = text_translator.detect(news_text).lang

    # 입력된 뉴스 기사의 언어가 선택한 언어와 다를 경우 오류 메시지 출력 및 요약 중단
    if detected_input_language != input_language_code:
        st.error(f"입력된 텍스트의 언어가 선택한 언어와 다릅니다. 입력된 텍스트의 언어: {detected_input_language}")
    else:
        # 입력된 텍스트의 줄바꿈 제거 (필요에 따라 정리)
        cleaned_text = news_text.replace('\n', '')

        # 한국어로 번역된 뉴스 원문 출력 (기본)
        if input_language_code != "ko":
            korean_translated_text = text_translator.translate(cleaned_text, src=input_language_code, dest='ko').text
            st.markdown("## 번역된 뉴스 원문 (한국어)")
            st.write(korean_translated_text)
        else:
            korean_translated_text = cleaned_text
            st.markdown("##  뉴스 원문")
            st.write(cleaned_text)

        # 한국어로 입력 및 출력 선택한 경우, 두 모델의 요약 및 품질 비교 수행
        if input_language_code == "ko" and output_language_code == "ko":
            with st.spinner('KoBART 요약 처리 중...'):
                # 제공받은 모델 요약
                input_ids1 = tokenizer1.encode(korean_translated_text, return_tensors="pt")
                output1 = model1.generate(input_ids1, eos_token_id=1, max_length=512, num_beams=5)
                summary_text1 = tokenizer1.decode(output1[0], skip_special_tokens=True)

                # 직접 학습한 모델 요약
                input_ids2 = summarization_tokenizer.encode(korean_translated_text, return_tensors="pt")
                output2 = summarization_model.model.generate(input_ids2, eos_token_id=1, max_length=512, num_beams=5)
                summary_text2 = summarization_tokenizer.decode(output2[0], skip_special_tokens=True)

            # 결과 출력
            st.markdown("## KoBART 요약 결과")
            st.write(f"digit82/kobart-summarization: {summary_text1}")
            st.write(f"학습한 모델: {summary_text2}")

            # 요약 품질 평가
            original_word_count1, summary_word_count1, compression_ratio1, keyword_coverage_ratio1 = evaluate_summary(korean_translated_text, summary_text1)
            original_word_count2, summary_word_count2, compression_ratio2, keyword_coverage_ratio2 = evaluate_summary(korean_translated_text, summary_text2)
            similarity_score = compute_similarity(summary_text1, summary_text2)

            # 품질 평가 결과 출력
            st.markdown("### digit82/kobart-summarization 요약 품질 평가:")
            st.write(f"- 원문 길이: {original_word_count1} 단어")
            st.write(f"- 요약 길이: {summary_word_count1} 단어")
            st.write(f"- 요약 압축률: {compression_ratio1:.2f}%")
            st.write(f"- 주요 키워드 포함률: {keyword_coverage_ratio1:.2f}%")

            st.markdown("### 학습한 모델 요약 품질 평가:")
            st.write(f"- 원문 길이: {original_word_count2} 단어")
            st.write(f"- 요약 길이: {summary_word_count2} 단어")
            st.write(f"- 요약 압축률: {compression_ratio2:.2f}%")
            st.write(f"- 주요 키워드 포함률: {keyword_coverage_ratio2:.2f}%")

            st.write(f"두 요약문 간의 유사도: {similarity_score:.2f}")

        # 다른 언어로 번역된 경우, 직접 학습한 모델로 요약 및 품질 평가 수행
        else:
            # KoBART 모델을 이용해 뉴스 원문을 요약
            with st.spinner('KoBART 요약 처리 중...'):
                model_input_ids = summarization_tokenizer.encode(korean_translated_text, return_tensors="pt")
                summary_output = summarization_model.model.generate(model_input_ids, eos_token_id=1, max_length=512, num_beams=5)
                summarized_text_korean = summarization_tokenizer.decode(summary_output[0], skip_special_tokens=True)

            # 한국어로 된 요약 결과를 출력
            st.markdown("## KoBART 요약 결과 (한국어)")
            st.write(summarized_text_korean)

            # 사용자가 선택한 언어로 원문을 번역 후 출력
            if output_language_code != "ko":
                original_translated_output = text_translator.translate(cleaned_text, src=input_language_code, dest=output_language_code).text
                st.markdown(f"## 번역된 뉴스 원문 ({output_language_selection})")
                st.write(original_translated_output)

                # 사용자가 선택한 언어로 요약 결과를 번역 후 출력
                summarized_translated_text = text_translator.translate(summarized_text_korean, src='ko', dest=output_language_code).text
                st.markdown(f"## KoBART 요약 결과 ({output_language_selection})")
                st.write(summarized_translated_text)

            # 요약 품질 평가 결과 출력 (직접 학습한 모델 기준)
            original_word_count, summary_word_count, compression_ratio, keyword_inclusion_rate = evaluate_summary(korean_translated_text, summarized_text_korean)
            st.markdown("## 요약 품질 평가")
            st.write(f"- 원문 길이: {original_word_count} 단어")
            st.write(f"- 요약 길이: {summary_word_count} 단어")
            st.write(f"- 요약 압축률: {compression_ratio:.2f}%")
            st.write(f"- 주요 키워드 포함률: {keyword_inclusion_rate:.2f}%")

            # 유사도
            cosine_similarity_value = compute_similarity(korean_translated_text, summarized_text_korean)
            st.write(f"- 원문과 요약문 간의 유사도: {cosine_similarity_value:.2f}")
