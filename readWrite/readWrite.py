


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
