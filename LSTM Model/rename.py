import os

os.getcwd()

path = "../../../../ownCloud/NeVRo/Data/EEG/10_SSD/"
# os.path.exists(path)

for fol in os.listdir(path)[1:]:
    path_sub = path + fol

    for fo in os.listdir(path_sub)[1:]:
        path_ssub = path_sub + "/SBA/"

        for f in os.listdir(path_ssub)[1:]:

            path_sssub = path_ssub + f + "/"

            print(path_sssub)

            for filename in os.listdir(path_sssub):
                if filename.endswith(".csv") and "filt" in filename:

                    newname = filename.split("SSD")[0] + "{}_SSD_cmp.csv".format(
                        "narrow" if "narrow" in path_sssub else "broad")

                    os.rename(path_sssub + filename, path_sssub + newname)

            print("Update done\n")