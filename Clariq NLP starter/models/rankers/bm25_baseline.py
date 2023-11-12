
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import os

nltk.data.path = [os.path.join(os.getcwd(), 'models/rankers/nltk_data')]

def stem_tokenize(text, remove_stopwords=True):
  stemmer = PorterStemmer()
  tokens = [word for sent in nltk.sent_tokenize(text) \
                                      for word in nltk.word_tokenize(sent)]
  tokens = [word for word in tokens if word not in \
          nltk.corpus.stopwords.words('english')]
  return [stemmer.stem(word) for word in tokens]

class BM25Ranker:
    def __init__(self):
        pass

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def rank_questions(self, instruction, gridworld_state, question_bank):
        """
        Implements the ranking function for a given instruction
        Inputs:
            instruction - Single instruction string, may or may not need any clarifying question
                          The evaluator may pass questions that don't need clarification, 
                          But only questions requiring clarifying questions will be scored

            gridworld_state - Internal state from the iglu-gridworld simulator corresponding to the instuction
                              NOTE: The state will only contain the "avatarInfo" and "worldEndingState"

            question_bank - List of clarifying questions to rank

        Outputs:
            ranks - A sorted list of questions from the question bank
                    Such that the first index corresponds to the best ranked question

        """

        tokenized_questions = [stem_tokenize(q) for q in question_bank]
        token_question_map = {' '.join(tq): q for q, tq in zip(question_bank, tokenized_questions)}
        bm25 = BM25Okapi(tokenized_questions)
        tokenized_instruction = stem_tokenize(instruction, True)
        bm25_ranked_tokenized_questions = bm25.get_top_n(tokenized_instruction, tokenized_questions, n=len(tokenized_questions))
        ranked_joined_sentences = [' '.join(tq) for tq in bm25_ranked_tokenized_questions]
        ranked_question_list = [token_question_map[sent] for sent in ranked_joined_sentences]
        return ranked_question_list
