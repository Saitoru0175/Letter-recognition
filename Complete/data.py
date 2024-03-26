import pandas as pd

data = pd.read_csv(r"E:\a-z alphabets\A_Z Handwritten Data.csv").astype('float32')
n = 5  # Số hàng bạn muốn hiển thị
print(data.head(n))