import matplotlib.pyplot as plt
import numpy as np

from cloth_classificator import ClothClassificator


def plot_image(predictions, label, img, class_names):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions)
    color = 'blue' if predicted_label == label else 'red'
    plt.xlabel(f"{class_names[predicted_label]} {100 * np.max(predictions):2.0f}% ({class_names[label]})",
               color=color)


def plot_value_array(predictions, label):
    plt.grid(False)
    plt.xticks(range(len(predictions)))
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.ylim([0, 1])
    bars = plt.bar(range(len(predictions)), predictions, color="#777777")

    predicted_label = np.argmax(predictions)
    bars[predicted_label].set_color('red')
    bars[label].set_color('blue')


def plot_images_labels_predictions(images, labels, predictions, class_names, num_rows=5, num_cols=3):
    plt.figure(figsize=(4 * num_cols, 2 * num_rows))
    num_images = num_rows * num_cols
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(predictions[i], labels[i], images[i], class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(predictions[i], labels[i])
    plt.tight_layout()
    plt.show()


def plot_first_images(images, labels, class_names, num_rows=5, num_cols=3):
    plt.figure(figsize=(2 * num_cols, 2 * num_rows))
    num_images = num_rows * num_cols
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    classificator = ClothClassificator()
    (train_images, train_labels), (test_images, test_labels) = classificator.load_training_data()
    plot_first_images(train_images, train_labels, classificator.class_names)

    prepared_train_images = ClothClassificator.prepare_data(train_images)
    prepared_test_images = ClothClassificator.prepare_data(test_images)

    classificator.compile_model()
    classificator.train_model(prepared_train_images, train_labels, enable_tensorboard=True)

    test_loss, test_acc = classificator.evaluate_accuracy(prepared_test_images, test_labels)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    test_predictions = classificator.predict(prepared_test_images)
    plot_images_labels_predictions(test_images, test_labels, test_predictions, classificator.class_names)
