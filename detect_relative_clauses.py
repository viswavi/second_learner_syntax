'''
python detect_relative_clauses.py \
    --corpus-type ets \
    --data-index /projects/ogma2/users/vijayv/extra_storage/ETS_Corpus_of_Non-Native_Written_English/data/text/index.csv \
    --data-dir /projects/ogma2/users/vijayv/extra_storage/ETS_Corpus_of_Non-Native_Written_English/data/text/responses/original/ \
    --language-code en

python detect_relative_clauses.py \
    --corpus-type cedel2 \
    --data-dir /projects/ogma2/users/vijayv/extra_storage/cedel2 \
    --language-code es
'''

import argparse
import csv
import numpy as np
import os
import pickle
from scipy.stats import ttest_ind
import stanza
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--corpus-type', type=str,
                    choices=["ets", "cedel2"],
                    help='Path to corpus index')
parser.add_argument('--data-index', type=str,
                    default=None,
                    help='Path to corpus index')
parser.add_argument('--data-dir', type=str,
                    default="/projects/ogma2/users/vijayv/extra_storage/ETS_Corpus_of_Non-Native_Written_English/data/text/responses/original/",
                    help='Path to corpus directory of learner English.')
parser.add_argument('--language-code', type=str, default="en",
                    help='Language code for loading Stanza models')
parser.add_argument('--paragraph-level-statistics', action="store_true",
                    help='Whether to count the incidence of relative clauses at a paragraph-level or sentence-level.')

def load_data(corpus_type, data_index, data_dir, language_group, score_level):
    '''
    - Load text data in paragraphs.
    - Return list of paragraphs.
    '''
    matching_texts = []
    language_codes = []
    if corpus_type == "ets":
        with open(data_index) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["Language"] in language_group and row["Score Level"] == score_level:
                    text_file = os.path.join(data_dir, row["Filename"])
                    matching_texts.append(open(text_file).read())
                    language_codes.append(row["Language"])
    else:
        assert corpus_type == "cedel2"
        for file in os.listdir(data_dir):
            language_code = file.split("_")[0]
            if language_code in language_group:
                text_file = os.path.join(data_dir, file)
                matching_texts.append(open(text_file).read())
                language_codes.append(language_code)
    return matching_texts, language_codes


def load_model(language_code):
    '''
    - Load Stanza model for dependency parsing in language code.
    '''
    stanza.download(lang=language_code)
    nlp = stanza.Pipeline(lang=language_code, processors='tokenize,pos,lemma,depparse')
    return nlp


def detect_relative_clause_on_sentence(model, sentence):
    '''
    - Detect if a sentence has a relative clause.
    '''
    raise NotImplementedError


def parse_dependencies(model, paragraphs, cache_file):
    if os.path.exists(cache_file):
        parses = pickle.load(open(cache_file, 'rb'))
    else:
        parses = []
        for p in tqdm(paragraphs):
            parse = model(p)
            parses.append(parse)
        pickle.dump(parses, open(cache_file, 'wb'))
    return parses


def detect_relative_clauses(parsed_paragraphs, use_paragraph_level_counts=False):
    '''
    - Check for relative clause in all sentences in all paragraphs,
      and return a list of list of relative clause labels (for each sentence
      in each paragraph).s
    '''
    relative_pronoun_counts = []
    for paragraph in parsed_paragraphs:
        paragraph_relative_pronoun_labels = []
        for sentence in paragraph.to_dict():
            relative_pronoun = False
            for token in sentence:
                if "feats" not in token:
                    continue
                token_features = [feature.strip() for feature in token["feats"].split('|')]
                pronoun_features = [feat for feat in token_features if feat.startswith("PronType=")]
                assert len(pronoun_features) <= 1
                if len(pronoun_features) == 1:
                    pronoun_feature_values = pronoun_features[0][len("PronType="):].split(',')
                    if "Rel" in pronoun_feature_values:
                        relative_pronoun = True
            paragraph_relative_pronoun_labels.append(int(relative_pronoun))
            fraction_of_sentences_with_relative_pronoun = sum(paragraph_relative_pronoun_labels) / len(paragraph_relative_pronoun_labels)
        relative_pronoun_counts.append(fraction_of_sentences_with_relative_pronoun)
    return relative_pronoun_counts


def compute_statistics(relative_pronoun_labels_a, relative_pronoun_labels_b, clause_labels_by_language):
    # Perform a Welch's t-test (https://en.wikipedia.org/wiki/Welch%27s_t-test) to test the hypothesis that
    # the frequency of relation pronoun constructions is greater in group A than group B without assuming 
    # equal variances of the two populations.
    t_statistic, p_value = ttest_ind(relative_clause_labels_a, relative_clause_labels_b, equal_var=False, alternative="less")
    summary_statistics_per_language = {lang: (np.mean(values), np.std(values)) for lang, values in clause_labels_by_language.items()}
    breakpoint()
    raise NotImplementedError


def display_statistics(statistics):
    raise NotImplementedError


if __name__ == "__main__":
    args = parser.parse_args()

    if args.corpus_type == "ets":
        language_group_a = ["KOR", "JPN", "ZHO"]
        language_group_b = ["DEU", "ARA", "FRA", "SPA", "HIN", "ITA"]
        score_level = "medium"
        PARSES_CACHE = "/projects/ogma2/users/vijayv/extra_storage/ETS_Corpus_of_Non-Native_Written_English/parses_cache/"
        # Unclear whether Telugu or Turkish have a frequently-used relative pronoun construction.
    else:
        assert args.corpus_type == "cedel2"
        language_group_a = ["jp", "cn"]
        language_group_b = ["de", "en", "fr", "gr", "it", "nl", "pt", "ru"]
        score_level = None
        PARSES_CACHE = "/projects/ogma2/users/vijayv/extra_storage/cedel2/parses_cache/"

    data_group_a, language_codes_a = load_data(args.corpus_type, args.data_index, args.data_dir, language_group_a, score_level)
    data_group_b, language_codes_b = load_data(args.corpus_type, args.data_index, args.data_dir, language_group_b, score_level)
    model = load_model(args.language_code)

    os.makedirs(PARSES_CACHE, exist_ok = True)
    cache_file_a = os.path.join(PARSES_CACHE, "dependencies_group_a.pkl")
    dependency_parses_a = parse_dependencies(model, data_group_a, cache_file_a)
    cache_file_b = os.path.join(PARSES_CACHE, "dependencies_group_b.pkl")
    dependency_parses_b = parse_dependencies(model, data_group_b, cache_file_b)

    relative_clause_labels_a = detect_relative_clauses(dependency_parses_a, args.paragraph_level)
    relative_clause_labels_b = detect_relative_clauses(dependency_parses_b, args.paragraph_level)

    combined_language_codes = language_codes_a + language_codes_b
    combined_relative_clause_labels = relative_clause_labels_a + relative_clause_labels_b
    clause_labels_by_language = {}
    for lang_code, relative_clause_labels in zip(combined_language_codes, combined_relative_clause_labels):
        if lang_code not in clause_labels_by_language:
            clause_labels_by_language[lang_code] = []
        clause_labels_by_language[lang_code].append(relative_clause_labels)

    statistics = compute_statistics(relative_clause_labels_a, relative_clause_labels_b, clause_labels_by_language)
    display_statistics(statistics)
