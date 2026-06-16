"""DES decision-level fusion baseline method.

Dynamic ensemble selection estimates the local competence of each modality
classifier on nearby validation samples and selects the classifier with higher
local competence for each test instance.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from task.train import BaseMethod, ExperimentContext


class DESMethod(BaseMethod):
    method_type = "Fusion"
    name = "DES"
    year = "2026"

    def _classifier(self):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=700, C=1.0, n_jobs=-1, random_state=self.seed),
        )

    def predict(self, ctx: ExperimentContext) -> np.ndarray:
        train_idx = np.arange(len(ctx.y_train))
        if len(np.unique(ctx.y_train)) > 1 and len(ctx.y_train) >= 30:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=self.seed)
            fit_local, val_local = next(splitter.split(np.zeros_like(ctx.y_train), ctx.y_train))
        else:
            fit_local, val_local = train_idx, train_idx

        x_eye = ctx.features.eye
        x_eeg = ctx.features.eeg
        tr = ctx.train_idx
        te = ctx.test_idx

        fit_idx = tr[fit_local]
        val_idx = tr[val_local]

        eye_model = self._classifier()
        eeg_model = self._classifier()
        eye_model.fit(x_eye[fit_idx], ctx.y_train[fit_local])
        eeg_model.fit(x_eeg[fit_idx], ctx.y_train[fit_local])

        eye_prob = eye_model.predict_proba(x_eye[te])
        eeg_prob = eeg_model.predict_proba(x_eeg[te])
        eye_val_pred = eye_model.predict(x_eye[val_idx])
        eeg_val_pred = eeg_model.predict(x_eeg[val_idx])
        y_val = ctx.y_train[val_local]

        scaler = StandardScaler()
        joint_train = np.concatenate([x_eye[fit_idx], x_eeg[fit_idx]], axis=1)
        joint_val = np.concatenate([x_eye[val_idx], x_eeg[val_idx]], axis=1)
        joint_test = np.concatenate([x_eye[te], x_eeg[te]], axis=1)
        scaler.fit(joint_train)
        joint_val = scaler.transform(joint_val)
        joint_test = scaler.transform(joint_test)

        k = min(15, len(joint_val))
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(joint_val)
        neighbor_ids = neighbors.kneighbors(joint_test, return_distance=False)

        chosen = []
        for i, ids in enumerate(neighbor_ids):
            eye_competence = np.mean(eye_val_pred[ids] == y_val[ids])
            eeg_competence = np.mean(eeg_val_pred[ids] == y_val[ids])
            if eye_competence >= eeg_competence:
                chosen.append(eye_prob[i])
            else:
                chosen.append(eeg_prob[i])
        return np.asarray(chosen).argmax(axis=1)
