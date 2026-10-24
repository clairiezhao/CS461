#! /usr/bin/python3

import argparse
import itertools
import numpy as np
import string
import sys
import yaml
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="Path to a text file.")
    parser.add_argument(
        "order",
        type=int,
        help="Number of past characters to look at. 0 for unigrams.")
    parser.add_argument(
        "outconfig",
        type=str,
        help="Path to save grammar transitions.")

    args = parser.parse_args()

    try:
        with open(args.path, 'r', encoding='utf-8') as corpus:
            text = corpus.read()
    except FileNotFoundError:
        print(f"Error: The file '{args.path}' was not found.")
        sys.exit(1)

    # Change everything to lowercase and remove all non alphabetic characters except for spaces.
    text = text.lower().replace('\n', ' ').replace('\t', ' ')
    text = ''.join(c for c in text if (c.isalpha() or c == ' '))

    # Initialize the ngrams
    # The indices of the frequency list will be all possible ngrams
    ngram_tokens = list(string.ascii_lowercase) + [' ']

    # Draw permutations equal to whatever was requested
    if args.order > 0:
        ngram_tokens = [''.join(token) for token in itertools.product(ngram_tokens, repeat = args.order+1)]

    # Observations
    # We could initialize this to 0, but, depending upon the ngram length, some occurences may not occur at all
    ngram_counts = dict()

    # Now loop through the text
    for i in range(args.order, len(text)):
        ngram = text[i-args.order:i+1]
        if ngram not in ngram_counts:
            ngram_counts[ngram] = 0
        ngram_counts[ngram] = ngram_counts[ngram] + 1

    total_ngrams = len(text) - args.order
    print(f"Collected {total_ngrams} total ngrams")

    # Now print out the normalized frequences for each ngram
    column_names = ["t"+str(x-args.order) for x in range(args.order)] + ["t"]
    normalized_ngrams = {}
    # Header row
    print(f"{','.join(column_names)}, ngram")
    for token in ngram_tokens:
        if token in ngram_counts:
            print(','.join(token) + f",{ngram_counts[token]/total_ngrams}")
            normalized_ngrams[token] = ngram_counts[token]/total_ngrams

    # Open the config, modify it with the grammar, and resave it
    if os.path.exists(args.outconfig):
        with open(args.outconfig, 'r') as infile:
            config = yaml.safe_load(infile)
    else:
        config = {}

    config[f"{args.order}gram"] = normalized_ngrams
    with open(args.outconfig, 'w') as outfile:
        yaml.dump(config, outfile)

