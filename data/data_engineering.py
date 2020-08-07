from glob import glob
import zipfile, json

OUTPUT_PATH = "/home/edent/Projects/Demos/TrumpTweet/data/"
OUT_NAME = "archive.json"
INPUT_PATH = OUTPUT_PATH + "trump_tweet_data_archive/"

def zip2text(filepath):
    """
    This function reads a single zipped json file and returns the
    content as a string.

    Args
    ----
    filepath: str, the path of the zipfile

    Return
    ------
    json: str, a string of the contents of the json file
    """
    with zipfile.ZipFile(filepath, 'r') as zf:
        with zf.open(name=zf.namelist()[0]) as jsn:
            bts = jsn.read()
    return bts.decode('utf-8')

# Only use condensed data files
files = glob(INPUT_PATH + "condensed_*")
flatten = lambda l: [item for sublist in l for item in sublist]

with open(OUTPUT_PATH + OUT_NAME, "w") as outfile:
    data = [json.loads(zip2text(f)) for f in files]
    json.dump(flatten(data), outfile)