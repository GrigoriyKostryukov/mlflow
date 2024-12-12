from src1.models.svm import get_svm_model

params = {"kernel": ("linear", "rbf"), "C": [1, 10]}
model = get_svm_model(parameters=params)
