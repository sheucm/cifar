from dataset import CifarDataManager
from utils import display_cifar

data = CifarDataManager().load_data()
display_cifar(data.test.images, size=10)