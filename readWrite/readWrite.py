import sys


def readToList(path):
    """
        Reads file from given path and stores results in list
    """
    ll = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            ll.append(line)

    return ll


def savePandasDFtoFile(df, path):

    if path.endswith(".parquet"):
        df.to_parquet(path, engine="pyarrow")
    elif path.endswith(".csv"):
        df.to_csv(path, sep=";")
    else:
        print("ERROR: Unable to save result. Unknown file extension. Supported formats: .parquet, .csv")
        sys.exit()
