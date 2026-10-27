import numpy as np
from .als import ALS


class PersonalALS(ALS):
    def targets(self, data):
        usual_rating = data.groupby('user').rating.transform('mean').to_numpy()
        ratings = data.rating.to_numpy()
        return (ratings >= 4) * np.clip(1 + (ratings - usual_rating) / 4, .5, 1.5)
