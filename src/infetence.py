def predict(model, img):
    return model(img).item()
