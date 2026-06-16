"""SVM eye-only baseline method."""

from __future__ import annotations

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from task.train import EstimatorMethod


class SVMMethod(EstimatorMethod):
    method_type = "Eye-only"
    name = "SVM"
    year = "2014"
    feature_name = "eye_at_dt"

    def estimator(self):
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=8.0, gamma="scale", random_state=self.seed),
        )
