import json
from evaluation import PTBTokenizer
from tqdm import tqdm
import pdb
import spacy

special_cases = {
    "there are two guys looking at a lot of bikes.": "there are two guys looking at a lot of bikes",
    "a bathroom with a vanity mirror sink. and toilet": "a bathroom with a vanity mirror sink and toilet",
    "a white bus sitting parked next to another bus.": "a white bus sitting parked next to another bus",
    "a bus driving down a street near down town la.": "a bus driving down a street near down town la",
    "a man dressed in white sits on a cart filled with sticks pulled by two oxen.": "a man dressed in white sits on a cart filled with sticks pulled by two oxen",
    "woman standing underneath stop sign in the middle of a city :--rrb-": "woman standing underneath stop sign in the middle of a city",
    "a silver piece of luggage sitting on an airport runway.": "a silver piece of luggage sitting on an airport runway",
    "an umbrella with cats all over it is open and leaning against a small herb garden with a buddha in it.": "an umbrella with cats all over it is open and leaning against a small herb garden with a buddha in it",
    "a woman holds a child that has something in their hand.": "a woman holds a child that has something in their hand",
    "a pile of fruit sitting on top of a blue and brown tray.": "a pile of fruit sitting on top of a blue and brown tray",
    "a man riding a skateboard down a tree covered street.": "a man riding a skateboard down a tree covered street",
    "a calico cat belly up on a laptop asleep.": "a calico cat belly up on a laptop asleep",
    "a room full of beds sitting next to each other.": "a room full of beds sitting next to each other",
    "a person looking at a kite they are flying.": "a person looking at a kite they are flying",
    "a booth in a restaurant with a wooden decoration on the wall behind it.": "a booth in a restaurant with a wooden decoration on the wall behind it",
    "a man standing in front of a table with a computer on it.": "a man standing in front of a table with a computer on it",
    "a woman looks like she is drunk and is talking on a cell phone.": "a woman looks like she is drunk and is talking on a cell phone",
    "a man holding a tennis racquet on top of a tennis court.": "a man holding a tennis racquet on top of a tennis court"
}


def second_tokenize(dataset):
    count = 0
    for item in dataset['annotations']:
        res = special_cases.get(item['caption'], None)
        if res is not None:
            item['caption'] = res
            count += 1
    assert count == len(special_cases)


def spacy_tokenize(s):
    spacy_en = spacy.load('en_core_web_md')
    return [tok.text for tok in spacy_en.tokenizer(s)]


def tokenize(dataset):
    captions = []
    for item in tqdm(dataset['annotations']):
        captions.append(item['caption'])
    print('tokenizing...')
    tokenized_corpus = PTBTokenizer.tokenize(captions)
    # pdb.set_trace()
    count = 0
    for item, value in zip(dataset['annotations'], tokenized_corpus.values()):
        item['caption'] = value[0]

    print(count)


train = json.load(open('annotation/captions_train2014.json'))
val = json.load(open('annotation/captions_val2014.json'))

print('train tokenize')
tokenize(train)

print('train map')
second_tokenize(train)
json.dump(train, open('annotation/captions_train2014_tokenized.json', 'w'))

# print('validate train')
# tokenize(train, True)

print('val tokenize')
tokenize(val)
json.dump(val, open('annotation/captions_val2014_tokenized.json', 'w'))
