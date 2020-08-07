import re
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def tweet_cleaner(tweet):
    # Remove Hyperlinks and tags, remove non latin alphabet/numeric characters
    punctuation = string.punctuation.replace("&", "")
    tweet =  re.sub(
        r"(@[A-Za-z0-9]+)|(https?:\/\/.*[\r\n]*)|([^a-zA-Z0-9{} ])".format(
            re.escape(punctuation)
        ),
        "",
        tweet,
    )
    
    # Replace multiple spaces with a single space
    tweet = re.sub(r"[ ]+", " ", tweet)
    
    # If the tweet now only contains punctuation or numbers, replace with empty string to be filtered later
    pattern = re.compile("[\d{}]+$".format(re.escape(punctuation)))
    if pattern.match(tweet):
        return ""
    
    return tweet
    

def formatplot(fig=None):
    """
    Formats a seaborn distplot to look sharper.
    Passing a specific figure can be useful if 
    plotting multiple lots at the same time. 
    
    args
    -----
    fig, matplotlib.pyplot.Figure
        A figure you'd like to format
    """
    if fig:
        ax = fig.gca()
    else:
        ax = plt.gca()
    
    # Remove borders
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.margins(x=0)
    
    ax.grid(axis='y')

def distplot(values=None, subject="", kwargs={}):
    """
    A convenience wrapper around seaborn distplot to
    make it more informative.
    
    args
    ----
    values, numpy.ndarray, list
        An array of samples values from the distibution
        
    
    """
    fig, ax = plt.subplots()
    sns.distplot(values, hist=True, ax=ax, **kwargs)
    ax.set_title(
        "Distribution of the {} per tweet | Mean {:.1f}, Median {:.1f}".format(
            subject, np.median(values), np.mean(values)
        ).title(),
        fontsize=20,
    )
    ax.set_xlabel(subject.title(), fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    
    formatplot(fig)
    return fig

