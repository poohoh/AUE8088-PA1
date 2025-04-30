import torchvision.models as tvm
lst = tvm.list_models()

for e in lst:
    print(e)