import os


def main():
    folders = ['./test', './train', './valid']

    for folder in folders:
        image_path = folder + '/images'
        labels_path = folder + '/labels'
        images = os.scandir(image_path)
        labels = os.scandir(labels_path)
        for image, label in zip(sorted(images, key=lambda x: x.name), sorted(labels, key=lambda x: x.name)):
            with open(labels_path + '/' + label.name) as label_file:
                final_image_path = image_path + '/' + image.name
                final_label_path = labels_path + '/' + label.name
                if len(label_file.readlines()) == 0:
                    print('no labels for image ' + final_image_path)
                    os.remove(final_image_path)
                    os.remove(final_label_path)


if __name__ == "__main__":
    main()
