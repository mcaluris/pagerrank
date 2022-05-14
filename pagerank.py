import os
import random
import re
import sys
from collections import defaultdict
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    

def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    base_probability = (1 - damping_factor) / len(corpus)
    links = corpus[page]
    number_of_links = len(links)
    number_of_pages = len(corpus)
    d = {page : 0.00 for page in corpus}
    if links:
        link_factor = (1 / number_of_links) * damping_factor
        for link in links:
            d[link] = link_factor
        
        for page in corpus:
            d[page] = d[page] + base_probability
    else:
        equal_probability = 1 / number_of_pages
        for page in corpus:
            d[page] = equal_probability
    return d

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    d = dict()
    for key in corpus:
        d[key] = 0.00
    page = random.choice(list(corpus))

    for i in range(0, n):
        d[page] += (1/n)
        probability_dic = transition_model(corpus, page, damping_factor)
        weights = []
        pages = []
        for k, v in probability_dic.items():
            pages.append(k)
            weights.append(v)
        page = random.choices(pages, weights=weights, k=1)[0]
    return d     
    
def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to .
    """
    result = {page : 1 / len(corpus) for page in corpus}
    is_difference = True
    number_of_pages = len(corpus)
    base_probability = (1 - damping_factor) / number_of_pages
    threshold = .001
    
    while is_difference:
        is_difference = False
        for page in corpus:
            current = 0
            links = []
            for link in corpus:
                if page in corpus[link]:
                    links.append(link)
            if links:
                current = get_sum_of_links(corpus, links, damping_factor, result) + base_probability
                is_difference = is_difference or (abs(result[page] - current) > .001)
                result[page] = current
            else:
                current = base_probability
                is_difference = is_difference or (abs(result[page] - current) > .001)
                result[page] = current
    total = sum(result.values())
    for page in result:
        result[page] = result[page] * 1 / total
    return result
                                
def get_sum_of_links(corpus, links, damping_factor, result):
    sum = 0
    for link in links:
        sum += (result[link] / (len(corpus[link])))
    sum *= damping_factor
    return sum         

if __name__ == "__main__":
    main()