# @Time    : 2023.04.16 下午 9:32
# @Author  : Tianfei Xu
# @Email   : tianfei8@outlook.com
# @Software: PyCharm
# @Project : program
# @File    : test001


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('digit-recognizer/test.csv')

# Get the image data and label
images = data.iloc[:,:].values.astype(np.uint8)
labels = data.iloc[:,0].values.astype(np.uint8)

# Display the first 10 images
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].reshape(28,28), cmap='gray')
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
plt.show()
