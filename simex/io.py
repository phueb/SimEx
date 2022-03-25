from typing import List

from simex import configs


def load_tokens(corpus_name: str,
                ) -> List[str]:

    p = configs.Dirs.corpora / f'{corpus_name}.txt'
    text = p.read_text()
    text = text.replace('\n', ' ')
    res = text.split()

    return res
