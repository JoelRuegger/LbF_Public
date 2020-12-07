import unittest

from preprocessing_pipeline import clean_reviews
from preprocessing_pipeline import count_elongated_words
from preprocessing_pipeline import findNegationContexts

class TestCleanReviews(unittest.TestCase):
    def test_clean_review(self):
        """
        Test that it can clean review
        """
        review = "This is a<br /> test."
        result = clean_reviews(review)
        self.assertEqual(result, "This is a test.")



class TestCountElongatedWords(unittest.TestCase):
    def test_count_elongated_words(self):
        """
        Test how many elongated words are in a review
        """
        review = "Hiiii how aare you todaaay?"
        result = count_elongated_words(review)
        self.assertEqual(result, 2)


class TestCountPosTags(unittest.TestCase):
    def count_posTags(self):
        """
        Test how many pos tags there are
        """
        pos_list = ['NUM', 'AUX', 'ADV', 'DET', 'NOUN', 'ADP', 'ADJ', 'NOUN', 'VERB', 'NOUN', 'PUNCT', 'PUNCT']
        result = count_posTags(pos_list)
        self.assertEqual(result, (3, 1, 1, 1, 2))



class TestFindNegationContexts(unittest.TestCase):

    def checkNegationException(sentence, i):
        # if word in the first three places of sentence
        if i in (0, 1, 2):
            return True
        # if word in the first three places of a sentence
        elif sentence[i-1].pos_ == "PUNCT":
            return True
        elif sentence[i-2].pos_ == "PUNCT":
            return True
        elif sentence[i-3].pos_ == "PUNCT":
            return True
        else:
            return False


    def findNegationContexts(self, get_option=checkNegationException):
        """
        Test how many negationContexts are found
        """
        #sentence = "No wonder has he been sick, he has not been well for ages."
        sentence1 = "No wonder has he been ill, he was outside all evening."
        result = findNegationContexts(sentence)
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)

