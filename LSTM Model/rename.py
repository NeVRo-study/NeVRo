import os

os.getcwd()

path = "../../../../ownCloud/NeVRo/Data/EEG/10_SSD/"

if not os.path.exists(path):  # changes depending on root folder
    path = "../../ownCloud/NeVRo/Data/EEG/10_SSD/"

if not os.path.exists(path):
    raise FileNotFoundError("Path does not exist:", path)

# # Change filenames from ..._1_... to ..._nomov_... etc

# for fol in os.listdir(path)[1:]:
#     path_sub = path + fol
#
#     for fo in os.listdir(path_sub)[1:]:
#         path_ssub = path_sub + "/SBA/"
#
#         for f in os.listdir(path_ssub)[1:]:
#
#             path_sssub = path_ssub + f + "/"
#
#             print(path_sssub)
#
#             for filename in os.listdir(path_sssub):
#                 if filename.endswith(".csv") and "filt" in filename:
#
#                     newname = filename.split("SSD")[0] + "{}_SSD_cmp.csv".format(
#                         "narrow" if "narrow" in path_sssub else "broad")
#
#                     os.rename(path_sssub + filename, path_sssub + newname)
#
#             print("Update done\n")


# # Change filenames from ..._1_... to ..._nomov_... etc

# for fol in os.listdir(path):
#     path_sub = path + fol
#
#     for fo in os.listdir(path_sub):
#         path_ssub = path_sub + "/{}/".format(fo)
#
#         for f in os.listdir(path_ssub):
#
#             path_sssub = path_ssub + f + "/"
#
#             print(path_sssub)
#
#             for filename in os.listdir(path_sssub):
#                 if "_2_" in filename:
#                     newname = filename.split("_2_")[0] + "_mov_" + filename.split("_2_")[1]
#                 elif "_1_" in filename:
#                     newname = filename.split("_1_")[0] + "_nomov_" + filename.split("_1_")[1]
#                 else:
#                     print("No change of naming", filename)
#
#                 os.rename(path_sssub + filename, path_sssub + newname)
#
#             print("Update done\n")
