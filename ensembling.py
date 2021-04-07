from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier,  
models = [
    ('rf', RandomForestClassifier, {})
]

def generate_meta_datasets(model_list):
