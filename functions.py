import re
from functools import reduce
# import random
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    '''
    argument: a sentence string
    returns a list of tokens(words)
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def word_index():
    wi = {'.': 1, '?': 2, 'Daniel': 3, 'John': 4, 'Mary': 5, 'Sandra': 6, 'Where': 7, 'back': 8, 'bathroom': 9,
          'bedroom': 10,
          'garden': 11, 'hallway': 12, 'is': 13, 'journeyed': 14, 'kitchen': 15, 'moved': 16, 'office': 17, 'the': 18,
          'to': 19, 'travelled': 20,
          'went': 21}
    return wi


def indx_word():
    id = {1: '.',
          2: '?',
          3: 'Daniel',
          4: 'John',
          5: 'Mary',
          6: 'Sandra',
          7: 'Where',
          8: 'back',
          9: 'bathroom',
          10: 'bedroom',
          11: 'garden',
          12: 'hallway',
          13: 'is',
          14: 'journeyed',
          15: 'kitchen',
          16: 'moved',
          17: 'office',
          18: 'the',
          19: 'to',
          20: 'travelled',
          21: 'went'}
    return id


def parse_stories(lines):
    '''
    - Parse stories provided in the bAbI tasks format
    - A story starts from line 1 to line 15. Every 3rd line,
      there is a question &amp;amp;amp;amp;amp; answer.
    - Function extracts sub-stories within a story and
      creates tuples
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            # reset story when line ID=1 (start of new story)
            story = []
        if '\t' in line:
            # this line is tab separated Q, A &amp;amp;amp;amp;amp; support fact ID
            q, a, supporting = line.split('\t')
            # tokenize the words of question
            q = tokenize(q)
            # Provide all the sub-stories till this question
            substory = [x for x in story if x]
            # A story ends and is appended to global story data-set
            data.append((substory, q, a))
            story.append('')
        else:
            # this line is a sentence of story
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f):
    '''
    argument: filename
    returns list of all stories in the argument data-set file
    '''
    # read the data file and parse 10k stories
    data = parse_stories(f.readlines())
    # lambda func to flatten the list of sentences into one list
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    # creating list of tuples for each story
    data = [(flatten(story), q, answer) for story, q, answer in data]
    return data


def vectorize_context(data, word_idx, story_maxlen, query_maxlen):
    # story vector initialization
    X = []
    # query vector initialization
    Xq = []
    # answer vector intialization
    Y = []
    for story, query in data:
        # creating list of story word indices
        x = [word_idx[w] for w in story]
        # creating list of query word indices
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        X.append(x)
        Xq.append(xq)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen))
