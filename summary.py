from keras.models import load_model # type: ignore

# Load the model
loaded_model = load_model('mnist_improved_v2.h5')

# Print model summary
loaded_model.summary()
