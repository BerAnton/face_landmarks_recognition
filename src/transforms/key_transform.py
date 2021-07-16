class TransformByKeys(object):
    """Allows to aply transforms to dataset by key."""

    def __init__(self, transform, names):
        """:args:
                - transform - torch transform or custom transform to apply.
                - names - part of dataset which will be transformed
           :returns:
                - TransformByKeys object"""
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        """:args:
                - sample - image to transform.
           :returns:
                - sample - transformed image."""
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample
