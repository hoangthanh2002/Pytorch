from lib import *

torch.manual_seed(1234) # cố định random seed để mỗi lần chạy code sẽ cho ra kết quả giống nhau
np.random.seed(1234)
random.seed(1234)

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225) 

num_epochs = 2