"""Input preprocessing shared by training and serving.

This lives in its own module so the saved model refers to
`app.preprocessing.scale_pixels`, which the API can import when it loads
the model. (If it were defined in train.py, running `python -m app.train`
would save it as `__main__.scale_pixels` and the API could not load it.)
"""


def scale_pixels(X):
    """Map pixel values from 0..16 to 0..1.

    A StandardScaler is not used on purpose: a few corner pixels are almost
    always 0 in training, so the scaler blows any ink there up to a huge value.
    Distorted images then end up far from all training data, and the SVC
    labelled almost all of them as the same digit with high confidence.
    """
    return X / 16.0
