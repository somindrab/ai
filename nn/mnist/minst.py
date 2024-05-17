from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

X, y = fetch_openml('mnist_784',
                    version=1,
                    return_X_y=True)

# fetch_openml will return pandas DataFrames
# .values will return the underlying numpy arrays
X = X.values
y = y.astype(int).values

print(X.shape)
print(y.shape)


#### 70000 images with 784 pixels each. The value of a pixel is betwen 0 and 255
#### there are 70000 class labels, i.e., what numbers does the data represent
### (70000, 784)
### (70000,)
###
####

# So about the 0 to 255. Eventually, we will use gradient based optimization techniques (SGD for example), and so for those things, it is useful if the inputs are normalized across a range such as -1 to +1. The easiest way to do so

X = ((X / 255. - 0.5) * 2)

## min value of 0 becomes -1, max value of 255 becomes +1, the midpoint is 0

fig,ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

ax=ax.flatten()

for i in range(10):
    img = X[y == i][0].reshape(28,28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

