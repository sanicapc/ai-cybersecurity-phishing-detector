from sklearn.ensemble import RandomForestClassifier

def get_model():
    model = RandomForestClassifier(n_estimators=100)
    return model
