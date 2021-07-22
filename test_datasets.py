
import unittest
from datasets import ESC
from collections import Counter

class test_datasets(unittest.TestCase):

    def test_ESC(self):
        dataset = ESC()
        self.assertEqual(Counter(dataset.labels)['animal/dog'], 40)


if __name__ == '__main__':
    unittest.main()

