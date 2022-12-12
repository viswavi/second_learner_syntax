import argparse
import csv
import os
import stanza

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    default="/projects/ogma2/users/vijayv/extra_storage/ETS_Corpus_of_Non-Native_Written_English/data/text/responses/original/",
                    help='Path to corpus directory of learner English.')
parser.add_argument('--data-index', type=str,
                    default="/projects/ogma2/users/vijayv/extra_storage/ETS_Corpus_of_Non-Native_Written_English/data/text/index.csv",
                    help='Path to corpus index')
parser.add_argument('--language-code', type=str, default="en",
                    help='Language code for loading Stanza models')

PARSES_CACHE = "/projects/ogma2/users/vijayv/extra_storage/second_learner_syntax/ETS_Corpus_of_Non-Native_Written_English/parses_cache"

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


def load_model(language_code, language):
    '''
    - Load Stanza model for dependency parsing in language code.
    '''
    nlp = stanza.Pipeline('en')
    return nlp


def detect_relative_clause_on_sentence(model, sentence):
    '''
    - Detect if a sentence has a relative clause.
    '''
    raise NotImplementedError


def parse_dependencies(model, sentence):
    raise NotImplementedError


def detect_relative_clause(model, paragraphs):
    '''
    - Run the relative clause detector on all sentences in all paragraphs,
      and return a list of list of relative clause labels (for each sentence
      in each paragraph).s
    '''
    raise NotImplementedError


if __name__ == "__main__":
    args = parser.parse_args()

    language_group_a = ["KOR", "JPN", "ZHO"]
    language_group_b = ["DEU", "ARA", "FRA", "SPA", "HIN", "ITA"]
    score_level = "high"
    # Unclear whether Telugu or Turkish have a frequently-used relative pronoun construction.
 
    data_group_a = load_data(args.data_index, args.data_dir, language_group_a, score_level)
    data_group_b = load_data(args.data_index, args.data_dir, language_group_b, score_level)
    breakpoint()
    model = load_model(args.language_code)
