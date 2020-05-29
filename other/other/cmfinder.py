import os

def execute_cmfinder04(path="cmfinder-0.4.1.18/bin/cmfinder04",
                       data="data"):
    """Executes the cmfinder program to calculate the yao-score
       with all samples in the given data folder.
    Args:
      path(String): The path for the cmfinder04 tool.
      data(String): The path of the datafolder.
    """
    output = data + "/yaoscores"
    os.system("mkdir -p " + output)
    for directory in os.listdir(data):
        if not os.path.isdir(directory):
            continue
        if directory == "yaoscores":
            continue
        if os.path.exists(output + "/" + directory + ".json"):
            print(output + "/" + directory + ".json" + " already exists.")
            continue
        with open(output + "/" + directory + ".json", "w") as f:
            f.write("{")
        for filename in os.listdir(data + "/" + directory):
            os.system("""%s --summarize --summarize-gsc \
                      --summarize-no-de-tag --fragmentary %s/%s/%s | grep -oP \
                      '(?<=yaoFormulaScore=)[0-9]+.[0-9]+' | \
                      awk '{print "\\"%s/%s/%s\\": "$1", "}' >> %s/%s.json"""
                      %(path, data, directory, filename, data, directory,
                        filename, output, directory))
        os.system("truncate -s-3 %s/%s.json" %(output, directory))
        with open(output + "/" + directory + ".json", "a") as f:
            f.write("}")


if __name__ == "__main__":
    parent = "../"
    path = parent + "cmfinder-0.4.1.18/bin/cmfinder04"
    data = parent + "data"
    execute_cmfinder04(path, data)
