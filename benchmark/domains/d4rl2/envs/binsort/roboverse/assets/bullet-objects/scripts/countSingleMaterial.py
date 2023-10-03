import os
dirName = "/home/albert/mice-bucket/albert/ShapeNetCore_v2/ShapeNetCore.v2"

def countMaterials(fileName):
    f = open(fileName, 'r')
    lines = f.readlines()
    return sum([1 if "newmtl" in line else 0 for line in lines])


with open("obj_ids.txt", "w") as f_out:
    numOneMaterials = {}
    for root, dirs, files in os.walk(dirName):
        for d in dirs:
            numOneMaterialInDir = 0
            for modelroot, modeldirs, modelfiles in os.walk(os.path.join(root, d)):
                for f in modelfiles:
                    fileName = os.path.join(modelroot, f)
                    if fileName.endswith(".mtl"):
                        if countMaterials(fileName) == 1:
                            numOneMaterialInDir += 1
                            path_list = fileName.split("/")
                            # print("path_list", path_list)
                            start_idx = path_list.index("ShapeNetCore.v2")
                            end_idx = path_list.index("models")
                            # print(path_list)
                            obj_id = path_list[start_idx + 1: end_idx]
                            # print(obj_id)
                            obj_id = str("/".join(obj_id))
                            print(obj_id)
                            f_out.write(obj_id + "\n")
                            # print(fileName)
            numOneMaterials[d] = numOneMaterialInDir
        break

print(numOneMaterials)

