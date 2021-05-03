import tensorflow as tf
import tensorflow.keras as keras

def test_model(model):
    (_, _), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    test_images = test_images / 255.0
    testing_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)
    
    _, test_acc = model.evaluate(testing_dataset)
    print('\nTest accuracy:', test_acc)


def main():
    filepath = "tf-models/cifar10-classifier-"
    file_name = input("Enter the model accuracy: ")
    filepath += file_name
    model = keras.models.load_model(filepath, compile=True)

    test_model(model)


if __name__ == "__main__":
    main()