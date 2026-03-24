import xgboost as xgb


class XGBoostPredictor:
    def __init__(self):
        self.models = {}  # 为每个网格训练一个模型（或者训练一个全局模型）
        self.pre_len = None

    def fit(self, X_train, Y_train):
        # 将输入展平
        Samples, seq_len, H, W = X_train.shape
        _, pre_len, _, _ = Y_train.shape
        X_flat = X_train.reshape(Samples, -1)
        Y_flat = Y_train.reshape(Samples, -1)
        self.pre_len = pre_len

        self.model = xgb.XGBRegressor(n_estimators=50, max_depth=5, objective='reg:squarederror')
        self.model.fit(X_flat, Y_flat)

    def predict(self, X_test):
        Samples, seq_len, H, W = X_test.shape
        X_flat = X_test.reshape(Samples, -1)
        pred_flat = self.model.predict(X_flat)
        return pred_flat.reshape(Samples, self.pre_len, H, W)
