def predict_single(customer, model):
    y_pred = model.predict(customer)
    return (y_pred[0])