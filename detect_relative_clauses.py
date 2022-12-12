import argparse
import csv
import os
import pickle
import stanza
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    default="/projects/ogma2/users/vijayv/extra_storage/ETS_Corpus_of_Non-Native_Written_English/data/text/responses/original/",
                    help='Path to corpus directory of learner English.')
parser.add_argument('--data-index', type=str,
                    default="/projects/ogma2/users/vijayv/extra_storage/ETS_Corpus_of_Non-Native_Written_English/data/text/index.csv",
                    help='Path to corpus index')
parser.add_argument('--language-code', type=str, default="en",
                    help='Language code for loading Stanza models')

PARSES_CACHE = "/projects/ogma2/users/vijayv/extra_storage/second_learner_syntax/ETS_Corpus_of_Non-Native_Written_English/parses_cache/"

def load_data(data_index, data_dir, language_group, score_level):
    '''
    - Load text data in paragraphs.
    - Return list of paragraphs.
    '''
    matching_texts = []
    with open(data_index) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["Language"] in language_group or row["Score Level"] == score_level:
                text_file = os.path.join(data_dir, row["Filename"])
                matching_texts.append(open(text_file).read())
    return matching_texts


def load_model(language_code):
    '''
    - Load Stanza model for dependency parsing in language code.
    '''
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
    return parses


def detect_relative_clauses(parsed_paragraphs):
    '''
    - Check for relative clause in all sentences in all paragraphs,
      and return a list of list of relative clause labels (for each sentence
      in each paragraph).s
    '''
    relative_pronoun_labels = []
    for paragraph in parsed_paragraphs:
        paragraph_relative_pronoun_labels = []
        for sentence in paragraph.to_dict():
            relative_pronoun = False
            for token in sentence:
                if "PronType=Rel" in token["feats"].split():
                    relative_pronoun = True
            paragraph_relative_pronoun_labels.append(int(relative_pronoun))
        relative_pronoun_labels.append(paragraph_relative_pronoun_labels)
    return relative_pronoun_labels


def compute_bootstrap_t_test(relative_pronoun_labels_a, relative_pronoun_labels_b):
    raise NotImplementedError


def display_statistics(statistics):
    raise NotImplementedError


if __name__ == "__main__":
    args = parser.parse_args()

    language_group_a = ["KOR", "JPN", "ZHO"]
    language_group_b = ["DEU", "ARA", "FRA", "SPA", "HIN", "ITA"]
    score_level = "medium"
    # Unclear whether Telugu or Turkish have a frequently-used relative pronoun construction.
 
    data_group_a = load_data(args.data_index, args.data_dir, language_group_a, score_level)
    data_group_b = load_data(args.data_index, args.data_dir, language_group_b, score_level)
    model = load_model(args.language_code)

    cache_file_a = os.path.join(PARSES_CACHE, "dependencies_group_a.pkl")
    dependency_parses_a = parse_dependencies(model, data_group_a, cache_file_a)
    cache_file_b = os.path.join(PARSES_CACHE, "dependencies_group_b.pkl")
    dependency_parses_b = parse_dependencies(model, data_group_b, cache_file_b)

    relative_clause_labels_a = detect_relative_clauses(dependency_parses_a)
    relative_clause_labels_b = detect_relative_clauses(dependency_parses_b)

    statistics = compute_bootstrap_t_test(relative_clause_labels_a, relative_clause_labels_b)
    display_statistics(statistics)
